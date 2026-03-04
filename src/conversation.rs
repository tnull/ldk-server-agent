use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::Context;

use crate::llm::{ChatMessage, ChatRole, LlmEngine};
use crate::mcp::McpClient;
use crate::mcp::protocol::ToolDefinition;
use crate::prompt::build_system_prompt;
use crate::safety::{self, ToolSafety};
use crate::tool_call::{self, ToolCall};

/// Maximum number of tool-call round trips before forcing a text response.
const MAX_TOOL_ROUNDS: usize = 10;

/// Manages the conversation state and orchestrates tool-use loops.
pub struct Conversation {
    messages: Vec<ChatMessage>,
    tools: Vec<ToolDefinition>,
}

impl Conversation {
    /// Creates a new conversation with the system prompt built from
    /// the given tool definitions.
    pub fn new(tools: Vec<ToolDefinition>) -> Self {
        let system_prompt = build_system_prompt(&tools);
        let messages = vec![ChatMessage {
            role: ChatRole::System,
            content: system_prompt,
        }];
        Self { messages, tools }
    }

    /// Processes a user message through the full tool-use loop:
    /// 1. Add user message to history
    /// 2. Generate LLM response
    /// 3. If response contains tool calls, execute them and loop
    /// 4. Return final text response
    ///
    /// The `confirm_fn` callback is called for mutating tools and should
    /// return `true` if the user confirms the action.
    pub async fn send_message(
        &mut self,
        user_input: &str,
        llm: &LlmEngine,
        mcp: &mut McpClient,
        on_token: &mut dyn FnMut(&str),
        confirm_fn: &mut dyn FnMut(&str, &str, &serde_json::Value) -> bool,
    ) -> anyhow::Result<String> {
        self.messages.push(ChatMessage {
            role: ChatRole::User,
            content: user_input.to_string(),
        });

        let mut rounds = 0;

        loop {
            if rounds >= MAX_TOOL_ROUNDS {
                eprintln!(
                    "\n[Warning: reached maximum tool call rounds ({}), forcing response]",
                    MAX_TOOL_ROUNDS
                );
                break;
            }

            eprintln!("[Round {}: generating...]", rounds + 1);

            let generated = llm
                .generate(&self.messages, &self.tools, on_token)
                .context("LLM generation failed")?;

            eprintln!(
                "\n[Round {} raw output ({} chars): {}]",
                rounds + 1,
                generated.len(),
                truncate_for_log(&generated, 300)
            );

            let (tool_calls, text) = tool_call::parse_tool_calls(&generated);

            if tool_calls.is_empty() {
                // No tool calls — this is the final response
                self.messages.push(ChatMessage {
                    role: ChatRole::Assistant,
                    content: generated.clone(),
                });
                return Ok(text);
            }

            eprintln!(
                "[Round {}: {} tool call(s), remaining text: {}]",
                rounds + 1,
                tool_calls.len(),
                if text.is_empty() {
                    "(none)".to_string()
                } else {
                    truncate_for_log(&text, 100)
                }
            );

            // The model wants to call tools
            self.messages.push(ChatMessage {
                role: ChatRole::Assistant,
                content: generated,
            });

            for call in &tool_calls {
                let result = self.execute_tool_call(call, mcp, confirm_fn).await?;
                self.messages.push(ChatMessage {
                    role: ChatRole::Tool,
                    content: result,
                });
            }

            rounds += 1;
        }

        // If we exhausted tool rounds, do one final generation without checking
        // for tool calls
        let final_response = llm
            .generate(&self.messages, &self.tools, on_token)
            .context("Final LLM generation failed")?;
        self.messages.push(ChatMessage {
            role: ChatRole::Assistant,
            content: final_response.clone(),
        });
        let (_, text) = tool_call::parse_tool_calls(&final_response);
        Ok(text)
    }

    async fn execute_tool_call(
        &self,
        call: &ToolCall,
        mcp: &mut McpClient,
        confirm_fn: &mut dyn FnMut(&str, &str, &serde_json::Value) -> bool,
    ) -> anyhow::Result<String> {
        let safety = safety::classify_tool(&call.name);

        if safety == ToolSafety::Mutating {
            let description = safety::describe_action(&call.name, &call.arguments);
            if !confirm_fn(&call.name, &description, &call.arguments) {
                return Ok(format!(
                    "Tool call '{}' was denied by the user. The user chose not to proceed with: {}",
                    call.name, description
                ));
            }
        }

        // Debug: show tool name and arguments
        let args_compact = serde_json::to_string(&call.arguments).unwrap_or_default();
        eprintln!("\n[Calling tool: {} | args: {}]", call.name, args_compact);

        // Start a timer thread to show elapsed time while the tool call runs
        let tool_name = call.name.clone();
        let timer_stop = Arc::new(AtomicBool::new(false));
        let timer_stop_clone = Arc::clone(&timer_stop);
        let timer_start = Instant::now();
        let timer_handle = std::thread::spawn(move || {
            // Wait a moment before showing the timer so quick calls don't flicker
            std::thread::sleep(std::time::Duration::from_millis(500));
            while !timer_stop_clone.load(Ordering::Relaxed) {
                let elapsed = timer_start.elapsed().as_secs_f32();
                eprint!("\r[tool: {} | {:.1}s] ", tool_name, elapsed);
                let _ = std::io::stderr().flush();
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        });

        let result = mcp
            .call_tool(&call.name, call.arguments.clone())
            .await
            .context(format!("Failed to call tool '{}'", call.name));

        // Stop the timer
        timer_stop.store(true, Ordering::Relaxed);
        let _ = timer_handle.join();
        let elapsed = timer_start.elapsed();

        // Clear the timer line
        eprint!("\r\x1b[2K");

        let result = result?;

        let text = result
            .content
            .into_iter()
            .map(|c| c.text)
            .collect::<Vec<_>>()
            .join("\n");

        if result.is_error == Some(true) {
            eprintln!(
                "[Tool {} returned error after {:.1}s: {}]",
                call.name,
                elapsed.as_secs_f32(),
                truncate_for_log(&text, 200)
            );
            Ok(format!("Tool '{}' returned an error: {}", call.name, text))
        } else {
            eprintln!(
                "[Tool {} completed in {:.1}s | response: {}]",
                call.name,
                elapsed.as_secs_f32(),
                truncate_for_log(&text, 200)
            );
            Ok(text)
        }
    }

    /// Clears the conversation history (keeping only the system prompt).
    pub fn clear(&mut self) {
        let system = self.messages.remove(0);
        self.messages.clear();
        self.messages.push(system);
    }
}

/// Truncates a string for log output, adding "..." if trimmed.
fn truncate_for_log(s: &str, max_len: usize) -> String {
    let s = s.replace('\n', " ");
    if s.len() <= max_len {
        s
    } else {
        format!("{}...", &s[..max_len])
    }
}
