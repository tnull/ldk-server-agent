pub mod protocol;

use std::sync::atomic::{AtomicU64, Ordering};

use std::time::Duration;

use anyhow::{Context, bail};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};

use protocol::{InitializeResult, JsonRpcRequest, JsonRpcResponse, ToolCallResult, ToolDefinition};

const PROTOCOL_VERSION: &str = "2024-11-05";
const CLIENT_NAME: &str = "ldk-server-agent";
const CLIENT_VERSION: &str = "0.1.0";

/// Timeout for individual MCP tool calls.
const TOOL_CALL_TIMEOUT: Duration = Duration::from_secs(30);

pub struct McpClient {
    child: Child,
    stdin: tokio::process::ChildStdin,
    reader: BufReader<tokio::process::ChildStdout>,
    next_id: AtomicU64,
    tools: Vec<ToolDefinition>,
}

impl McpClient {
    /// Spawns the MCP server process and performs the initialization handshake.
    pub async fn connect(
        mcp_binary: &str,
        env_vars: Vec<(String, String)>,
    ) -> anyhow::Result<Self> {
        let mut cmd = Command::new(mcp_binary);
        cmd.stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        for (key, value) in &env_vars {
            cmd.env(key, value);
        }

        let mut child = cmd.spawn().context("Failed to spawn MCP server process")?;

        let stdin = child
            .stdin
            .take()
            .context("Failed to capture MCP server stdin")?;
        let stdout = child
            .stdout
            .take()
            .context("Failed to capture MCP server stdout")?;
        let reader = BufReader::new(stdout);

        let mut client = Self {
            child,
            stdin,
            reader,
            next_id: AtomicU64::new(1),
            tools: Vec::new(),
        };

        client.initialize().await?;
        client.discover_tools().await?;

        Ok(client)
    }

    async fn initialize(&mut self) -> anyhow::Result<InitializeResult> {
        let resp = self
            .send_request(
                "initialize",
                json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {
                        "name": CLIENT_NAME,
                        "version": CLIENT_VERSION
                    }
                }),
            )
            .await?;

        let init_result: InitializeResult =
            serde_json::from_value(resp).context("Failed to parse initialize response")?;

        if init_result.protocol_version != PROTOCOL_VERSION {
            bail!(
                "Protocol version mismatch: expected {}, got {}",
                PROTOCOL_VERSION,
                init_result.protocol_version
            );
        }

        // Send initialized notification
        self.send_notification("notifications/initialized").await?;

        Ok(init_result)
    }

    async fn discover_tools(&mut self) -> anyhow::Result<()> {
        let resp = self.send_request("tools/list", json!({})).await?;

        let tools_value = resp
            .get("tools")
            .context("Missing 'tools' field in tools/list response")?;
        self.tools =
            serde_json::from_value(tools_value.clone()).context("Failed to parse tools list")?;

        Ok(())
    }

    /// Returns the list of available tools discovered from the MCP server.
    pub fn tools(&self) -> &[ToolDefinition] {
        &self.tools
    }

    /// Calls a tool on the MCP server with the given arguments.
    ///
    /// Times out after [`TOOL_CALL_TIMEOUT`] to prevent indefinite hangs
    /// (e.g. if the upstream LDK Server is unreachable).
    pub async fn call_tool(
        &mut self,
        name: &str,
        arguments: Value,
    ) -> anyhow::Result<ToolCallResult> {
        let fut = self.send_request(
            "tools/call",
            json!({ "name": name, "arguments": arguments }),
        );

        let resp = tokio::time::timeout(TOOL_CALL_TIMEOUT, fut)
            .await
            .map_err(|_| {
                anyhow::anyhow!(
                    "Tool call '{}' timed out after {}s (is the LDK Server reachable?)",
                    name,
                    TOOL_CALL_TIMEOUT.as_secs()
                )
            })??;

        let result: ToolCallResult =
            serde_json::from_value(resp).context("Failed to parse tool call result")?;

        Ok(result)
    }

    async fn send_request(&mut self, method: &str, params: Value) -> anyhow::Result<Value> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let request = JsonRpcRequest::new(id, method, params);
        let line = serde_json::to_string(&request)?;

        self.stdin.write_all(line.as_bytes()).await?;
        self.stdin.write_all(b"\n").await?;
        self.stdin.flush().await?;

        let response = self.recv_response().await?;

        if let Some(error) = response.error {
            bail!("JSON-RPC error {}: {}", error.code, error.message);
        }

        response
            .result
            .context("Missing result in JSON-RPC response")
    }

    async fn send_notification(&mut self, method: &str) -> anyhow::Result<()> {
        let notification = JsonRpcRequest::notification(method);
        let line = serde_json::to_string(&notification)?;

        self.stdin.write_all(line.as_bytes()).await?;
        self.stdin.write_all(b"\n").await?;
        self.stdin.flush().await?;

        Ok(())
    }

    async fn recv_response(&mut self) -> anyhow::Result<JsonRpcResponse> {
        let mut line = String::new();
        self.reader
            .read_line(&mut line)
            .await
            .context("Failed to read from MCP server stdout")?;

        if line.is_empty() {
            bail!("MCP server closed stdout unexpectedly");
        }

        serde_json::from_str(line.trim()).context("Failed to parse JSON-RPC response")
    }

    /// Shuts down the MCP server process.
    pub async fn shutdown(&mut self) -> anyhow::Result<()> {
        let _ = self.child.kill().await;
        let _ = self.child.wait().await;
        Ok(())
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        // Best-effort kill on drop. The process will be cleaned up when the
        // child handle is dropped regardless, but we attempt explicit kill.
        let _ = self.child.start_kill();
    }
}
