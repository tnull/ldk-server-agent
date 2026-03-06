mod cli;
mod config;
mod conversation;
mod llm;
mod markdown;
mod mcp;
mod prompt;
mod safety;
mod tool_call;

use std::path::PathBuf;
use std::process;

use config::Config;
use conversation::Conversation;
use llm::LlmEngine;
use mcp::McpClient;

const DEFAULT_CONFIG_PATH: &str = "config.toml";

#[tokio::main]
async fn main() {
    let config_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_CONFIG_PATH));

    let config = match Config::load(&config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!(
                "Failed to load config from '{}': {}",
                config_path.display(),
                e
            );
            eprintln!("Usage: ldk-server-agent [config.toml]");
            process::exit(1);
        }
    };

    eprintln!("Connecting to MCP server...");
    let mcp = match McpClient::connect(
        config.mcp.binary_path.to_str().unwrap_or("ldk-server-mcp"),
        config.mcp.env_pairs(),
    )
    .await
    {
        Ok(client) => client,
        Err(e) => {
            eprintln!("Failed to connect to MCP server: {}", e);
            process::exit(1);
        }
    };

    let tool_count = mcp.tools().len();
    eprintln!("MCP server connected. Discovered {} tools.", tool_count);

    eprintln!("Loading LLM...");
    let llm = match LlmEngine::load(
        &config.model.model_path,
        config.model.lora_path.as_deref(),
        config.model.context_size,
        config.model.max_generation_tokens,
        config.model.gpu_layers,
        config.model.threads,
    ) {
        Ok(engine) => engine,
        Err(e) => {
            eprintln!("Failed to load LLM: {}", e);
            process::exit(1);
        }
    };

    eprintln!("LLM loaded successfully.\n");

    let conversation = Conversation::new(mcp.tools().to_vec());

    if let Err(e) = cli::run_repl(llm, mcp, conversation).await {
        eprintln!("Fatal error: {}", e);
        process::exit(1);
    }
}
