use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::Context;

use crate::llm::{ChatMessage, ChatRole, GenerationStats, LlmEngine};
use crate::mcp::McpClient;
use crate::mcp::protocol::ToolDefinition;
use crate::prompt::build_system_prompt;
use crate::safety::{self, ToolSafety};
use crate::tool_call::{self, ToolCall};
use crate::ui::UserInterface;

/// Maximum number of tool-call round trips before forcing a text response.
const MAX_TOOL_ROUNDS: usize = 10;

/// Fraction of the context window (in characters, assuming ~4 chars/token)
/// that a single tool result is allowed to occupy.
///
/// `context_size / 16` means each tool result gets at most 1/16th of the
/// context window.  With up to ~8 tool results, the system prompt, user
/// messages, assistant messages, and generation budget all still fit
/// comfortably.
///
/// Examples at different context sizes:
///   32k ctx  →  ~8 000 chars  (~2 000 tokens) per tool result
///   65k ctx  →  ~16 000 chars (~4 000 tokens) per tool result
///  128k ctx  →  ~32 000 chars (~8 000 tokens) per tool result
const TOOL_RESULT_CONTEXT_DIVISOR: u32 = 4;

/// Manages the conversation state and orchestrates tool-use loops.
pub struct Conversation {
    messages: Vec<ChatMessage>,
    tools: Vec<ToolDefinition>,
    /// Maximum character length for a single tool result before truncation.
    max_tool_result_chars: usize,
}

impl Conversation {
    /// Creates a new conversation with the system prompt built from
    /// the given tool definitions.
    ///
    /// `context_size` is used to derive the per-tool-result truncation
    /// limit so that it scales with the available context window.
    pub fn new(tools: Vec<ToolDefinition>, context_size: u32) -> Self {
        let system_prompt = build_system_prompt(&tools);
        let messages = vec![ChatMessage {
            role: ChatRole::System,
            content: system_prompt,
        }];
        // ~4 chars per token, so context_size / DIVISOR tokens ≈
        // context_size * 4 / DIVISOR chars.
        let max_tool_result_chars =
            (context_size as usize * 4) / TOOL_RESULT_CONTEXT_DIVISOR as usize;
        Self {
            messages,
            tools,
            max_tool_result_chars,
        }
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
        stats: &GenerationStats,
        on_token: &mut dyn FnMut(&str),
        on_round_complete: &mut dyn FnMut(),
        confirm_fn: &mut dyn FnMut(&str, &str, &serde_json::Value) -> bool,
        ui: &Arc<dyn UserInterface>,
    ) -> anyhow::Result<String> {
        self.messages.push(ChatMessage {
            role: ChatRole::User,
            content: user_input.to_string(),
        });

        let mut rounds = 0;

        loop {
            if rounds >= MAX_TOOL_ROUNDS {
                ui.show_warning(&format!(
                    "reached maximum tool call rounds ({}), forcing response",
                    MAX_TOOL_ROUNDS
                ));
                break;
            }

            let generated = llm
                .generate(&self.messages, &self.tools, stats, on_token, ui.as_ref())
                .context("LLM generation failed")?;

            let (tool_calls, text) = tool_call::parse_tool_calls(&generated);

            if tool_calls.is_empty() {
                // No tool calls — this is the final response
                self.messages.push(ChatMessage {
                    role: ChatRole::Assistant,
                    content: generated.clone(),
                });
                return Ok(text);
            }

            // The model wants to call tools
            self.messages.push(ChatMessage {
                role: ChatRole::Assistant,
                content: generated,
            });

            // Signal that a generation round with tool calls has finished.
            // This allows the caller to flush/reset streaming render state
            // (e.g. the markdown renderer's tool-call suppression flag) so
            // that the next generation round's output is not swallowed.
            on_round_complete();

            for call in &tool_calls {
                let result = self
                    .execute_tool_call(call, mcp, confirm_fn, &Arc::clone(ui))
                    .await?;
                let result = truncate_tool_result(&result, self.max_tool_result_chars);
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
            .generate(&self.messages, &self.tools, stats, on_token, ui.as_ref())
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
        ui: &Arc<dyn UserInterface>,
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

        // Start a timer thread to show elapsed time while the tool call runs
        let tool_name = call.name.clone();
        let timer_stop = Arc::new(AtomicBool::new(false));
        let timer_stop_clone = Arc::clone(&timer_stop);
        let timer_start = Instant::now();
        let timer_ui = Arc::clone(ui);
        let timer_handle = std::thread::spawn(move || {
            // Wait a moment before showing the timer so quick calls don't flicker
            std::thread::sleep(std::time::Duration::from_millis(500));
            while !timer_stop_clone.load(Ordering::Relaxed) {
                let elapsed = timer_start.elapsed();
                timer_ui.show_tool_progress(&tool_name, elapsed);
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
        ui.clear_tool_progress();

        let result = result?;

        let text = result
            .content
            .into_iter()
            .map(|c| c.text)
            .collect::<Vec<_>>()
            .join("\n");

        if result.is_error == Some(true) {
            ui.show_tool_result(&call.name, elapsed, &truncate_for_log(&text, 200), true);
            Ok(format!("Tool '{}' returned an error: {}", call.name, text))
        } else {
            ui.show_tool_result(&call.name, elapsed, &truncate_for_log(&text, 200), false);
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

/// Truncates a tool result that would otherwise consume too much context.
///
/// Preserves valid UTF-8 by finding the nearest char boundary.
fn truncate_tool_result(s: &str, max_chars: usize) -> String {
    if s.len() <= max_chars {
        return s.to_string();
    }

    // Find a clean UTF-8 boundary at or before the limit.
    let boundary = s.floor_char_boundary(max_chars);
    format!(
        "{}\n... [truncated — {} chars of {} total]",
        &s[..boundary],
        boundary,
        s.len()
    )
}

/// Truncates a string for log output, adding "..." if trimmed.
fn truncate_for_log(s: &str, max_len: usize) -> String {
    let s = s.replace('\n', " ");
    if s.len() <= max_len {
        s
    } else {
        let boundary = s.floor_char_boundary(max_len);
        format!("{}...", &s[..boundary])
    }
}
