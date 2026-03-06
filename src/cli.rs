use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::conversation::Conversation;
use crate::llm::{GenerationStats, LlmEngine};
use crate::markdown::StreamRenderer;
use crate::mcp::McpClient;
use crate::ui::{InputAction, UserInterface};

/// Runs the interactive REPL loop.
pub async fn run_repl(
    llm: LlmEngine,
    mut mcp: McpClient,
    mut conversation: Conversation,
    ui: Arc<dyn UserInterface>,
) -> anyhow::Result<()> {
    ui.display_banner();

    loop {
        let action = ui.read_user_input();

        match action {
            InputAction::Empty => continue,
            InputAction::Interrupt => continue,
            InputAction::Eof => break,
            InputAction::Error(err) => {
                ui.show_error(&format!("Readline error: {}", err));
                break;
            }
            InputAction::Message(input) => {
                if input.starts_with('/') {
                    match handle_command(&input, &mut conversation, &*ui) {
                        CommandResult::Continue => continue,
                        CommandResult::Quit => break,
                    }
                }

                ui.confirm_user_input(&input);

                let start = Instant::now();
                let got_first_token = Arc::new(AtomicBool::new(false));
                let stats = GenerationStats::new();

                // Spawn a timer thread that shows a live status line.
                // Before the first token it overwrites the current line;
                // once streaming begins it renders below the cursor.
                let timer_stop = Arc::new(AtomicBool::new(false));
                let timer_ui = Arc::clone(&ui);
                let timer_stop_clone = Arc::clone(&timer_stop);
                let got_first_clone = Arc::clone(&got_first_token);
                let stats_clone = Arc::clone(&stats);
                let timer_start = start;
                let timer_handle = std::thread::spawn(move || {
                    while !timer_stop_clone.load(Ordering::Relaxed) {
                        let elapsed = timer_start.elapsed().as_secs_f32();
                        let prompt_tok = stats_clone.prompt_tokens.load(Ordering::Relaxed);
                        let ctx = stats_clone.context_size.load(Ordering::Relaxed);
                        let msgs = stats_clone.message_count.load(Ordering::Relaxed);
                        let gen_tok = stats_clone.generated_tokens.load(Ordering::Relaxed);

                        let status = if prompt_tok > 0 {
                            format!(
                                "[{:.1}s | {} prompt + {} generated tokens | {} ctx | {} messages]",
                                elapsed, prompt_tok, gen_tok, ctx, msgs,
                            )
                        } else {
                            format!("[{:.1}s | preparing...]", elapsed)
                        };

                        let below = got_first_clone.load(Ordering::Relaxed);
                        timer_ui.update_status(&status, below);
                        std::thread::sleep(std::time::Duration::from_millis(100));
                    }
                });

                let got_first_for_cb = Arc::clone(&got_first_token);
                let mut first_token_time = None;
                let md_renderer = std::cell::RefCell::new(StreamRenderer::new());

                let result = {
                    let md = &md_renderer;
                    let ftt = &mut first_token_time;
                    let got_first_ref = &got_first_for_cb;
                    let ui_ref = &ui;
                    let mut on_token = |token: &str| {
                        if !got_first_ref.load(Ordering::Relaxed) {
                            got_first_ref.store(true, Ordering::Relaxed);
                            *ftt = Some(start.elapsed());
                            ui_ref.begin_assistant_response();
                        }
                        md.borrow_mut().push(token, &mut |rendered| {
                            ui_ref.stream_assistant_token(rendered);
                        });
                    };

                    let mut on_round_complete = || {
                        md_renderer
                            .borrow_mut()
                            .reset_tool_call_state(&mut |rendered| {
                                ui.stream_assistant_token(rendered);
                            });
                    };

                    let mut confirm_fn =
                        |_name: &str, description: &str, _args: &serde_json::Value| -> bool {
                            ui.confirm_action(description)
                        };

                    conversation
                        .send_message(
                            &input,
                            &llm,
                            &mut mcp,
                            &stats,
                            &mut on_token,
                            &mut on_round_complete,
                            &mut confirm_fn,
                            &ui,
                        )
                        .await
                };

                // Flush any remaining partial line from the renderer.
                md_renderer.borrow_mut().flush(&mut |rendered| {
                    ui.stream_assistant_token(rendered);
                });

                ui.end_assistant_response();

                // Stop the timer thread and clear the status line.
                timer_stop.store(true, Ordering::Relaxed);
                let _ = timer_handle.join();
                if got_first_token.load(Ordering::Relaxed) {
                    ui.clear_status();
                }

                let total = start.elapsed();
                match result {
                    Ok(_response) => {
                        ui.show_timing(total, first_token_time);
                    }
                    Err(e) => {
                        ui.show_error(&format!(
                            "[Error after {:.1}s: {:#}]\n",
                            total.as_secs_f32(),
                            e
                        ));
                    }
                }
            }
        }
    }

    ui.goodbye();
    let _ = mcp.shutdown().await;

    Ok(())
}

enum CommandResult {
    Continue,
    Quit,
}

/// Handles a slash-command, returning whether the REPL should continue or quit.
fn handle_command(
    input: &str,
    conversation: &mut Conversation,
    ui: &dyn UserInterface,
) -> CommandResult {
    match input {
        "/quit" | "/exit" | "/q" => CommandResult::Quit,
        "/clear" | "/reset" => {
            conversation.clear();
            ui.show_command_output("[Conversation cleared]\n");
            CommandResult::Continue
        }
        "/help" => {
            ui.show_command_output("Commands:");
            ui.show_command_output("  /quit, /exit, /q  - Exit the assistant");
            ui.show_command_output("  /clear, /reset    - Clear conversation history");
            ui.show_command_output("  /help             - Show this help message");
            ui.show_command_output("");
            CommandResult::Continue
        }
        _ => {
            ui.show_command_output(&format!(
                "[Unknown command: {}. Type /help for available commands.]\n",
                input
            ));
            CommandResult::Continue
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::protocol::ToolDefinition;
    use crate::ui::{MockUi, UiEvent};

    /// Helper: create a minimal `Conversation` with no tools.
    fn test_conversation() -> Conversation {
        Conversation::new(Vec::<ToolDefinition>::new(), 8192)
    }

    // ── Slash commands ──────────────────────────────────────────────

    #[test]
    fn test_quit_command() {
        let mut conv = test_conversation();
        let ui = MockUi::new(vec![]);
        let result = handle_command("/quit", &mut conv, &ui);
        assert!(matches!(result, CommandResult::Quit));
        // /quit produces no UI output
        assert!(ui.events().is_empty());
    }

    #[test]
    fn test_exit_command() {
        let mut conv = test_conversation();
        let ui = MockUi::new(vec![]);
        let result = handle_command("/exit", &mut conv, &ui);
        assert!(matches!(result, CommandResult::Quit));
    }

    #[test]
    fn test_q_command() {
        let mut conv = test_conversation();
        let ui = MockUi::new(vec![]);
        let result = handle_command("/q", &mut conv, &ui);
        assert!(matches!(result, CommandResult::Quit));
    }

    #[test]
    fn test_clear_command() {
        let mut conv = test_conversation();
        let ui = MockUi::new(vec![]);
        let result = handle_command("/clear", &mut conv, &ui);
        assert!(matches!(result, CommandResult::Continue));

        let events = ui.events();
        assert_eq!(events.len(), 1);
        assert!(
            matches!(&events[0], UiEvent::CommandOutput(msg) if msg.contains("Conversation cleared"))
        );
    }

    #[test]
    fn test_reset_command() {
        let mut conv = test_conversation();
        let ui = MockUi::new(vec![]);
        let result = handle_command("/reset", &mut conv, &ui);
        assert!(matches!(result, CommandResult::Continue));

        let events = ui.events();
        assert!(
            matches!(&events[0], UiEvent::CommandOutput(msg) if msg.contains("Conversation cleared"))
        );
    }

    #[test]
    fn test_help_command() {
        let mut conv = test_conversation();
        let ui = MockUi::new(vec![]);
        let result = handle_command("/help", &mut conv, &ui);
        assert!(matches!(result, CommandResult::Continue));

        let events = ui.events();
        // /help outputs multiple lines: header + 3 commands + blank
        assert_eq!(events.len(), 5);
        assert!(matches!(&events[0], UiEvent::CommandOutput(msg) if msg.contains("Commands:")));
        assert!(matches!(&events[1], UiEvent::CommandOutput(msg) if msg.contains("/quit")));
        assert!(matches!(&events[2], UiEvent::CommandOutput(msg) if msg.contains("/clear")));
        assert!(matches!(&events[3], UiEvent::CommandOutput(msg) if msg.contains("/help")));
    }

    #[test]
    fn test_unknown_command() {
        let mut conv = test_conversation();
        let ui = MockUi::new(vec![]);
        let result = handle_command("/foobar", &mut conv, &ui);
        assert!(matches!(result, CommandResult::Continue));

        let events = ui.events();
        assert_eq!(events.len(), 1);
        assert!(
            matches!(&events[0], UiEvent::CommandOutput(msg) if msg.contains("Unknown command: /foobar"))
        );
    }

    // ── MockUi input / confirm behaviour ────────────────────────────

    #[test]
    fn test_mock_ui_empty_input() {
        let ui = MockUi::new(vec!["".into()]);
        let action = ui.read_user_input();
        assert!(matches!(action, InputAction::Empty));
    }

    #[test]
    fn test_mock_ui_message_input() {
        let ui = MockUi::new(vec!["hello world".into()]);
        let action = ui.read_user_input();
        assert!(matches!(action, InputAction::Message(ref s) if s == "hello world"));
    }

    #[test]
    fn test_mock_ui_eof_when_exhausted() {
        let ui = MockUi::new(vec![]);
        let action = ui.read_user_input();
        assert!(matches!(action, InputAction::Eof));
    }

    #[test]
    fn test_mock_ui_confirm_action_default_deny() {
        let ui = MockUi::new(vec![]);
        // No confirms scripted → default deny
        assert!(!ui.confirm_action("do something dangerous"));

        let events = ui.events();
        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            UiEvent::ConfirmAction { description, result }
            if description == "do something dangerous" && !result
        ));
    }

    #[test]
    fn test_mock_ui_confirm_action_scripted_approve() {
        let ui = MockUi::with_confirms(vec![], vec![true]);
        assert!(ui.confirm_action("open channel"));

        let events = ui.events();
        assert!(matches!(
            &events[0],
            UiEvent::ConfirmAction { result, .. } if *result
        ));
    }

    #[test]
    fn test_mock_ui_records_streaming_events() {
        let ui = MockUi::new(vec![]);
        ui.display_banner();
        ui.begin_assistant_response();
        ui.stream_assistant_token("Hello ");
        ui.stream_assistant_token("world");
        ui.end_assistant_response();
        ui.goodbye();

        let events = ui.events();
        assert_eq!(
            events,
            vec![
                UiEvent::Banner,
                UiEvent::BeginAssistantResponse,
                UiEvent::AssistantToken("Hello ".into()),
                UiEvent::AssistantToken("world".into()),
                UiEvent::EndAssistantResponse,
                UiEvent::Goodbye,
            ]
        );
    }

    #[test]
    fn test_mock_ui_records_status_events() {
        let ui = MockUi::new(vec![]);
        ui.update_status("processing...", false);
        ui.update_status("still going...", true);
        ui.clear_status();

        let events = ui.events();
        assert_eq!(events.len(), 3);
        assert!(matches!(
            &events[0],
            UiEvent::UpdateStatus { status, below_output }
            if status == "processing..." && !below_output
        ));
        assert!(matches!(
            &events[1],
            UiEvent::UpdateStatus { status, below_output }
            if status == "still going..." && *below_output
        ));
        assert!(matches!(&events[2], UiEvent::ClearStatus));
    }

    #[test]
    fn test_mock_ui_records_tool_events() {
        use std::time::Duration;

        let ui = MockUi::new(vec![]);
        ui.show_tool_progress("get_balances", Duration::from_secs_f32(1.5));
        ui.clear_tool_progress();
        ui.show_tool_result(
            "get_balances",
            Duration::from_secs_f32(2.0),
            "balance: 100000 sats",
            false,
        );

        let events = ui.events();
        assert_eq!(events.len(), 3);
        assert!(matches!(
            &events[0],
            UiEvent::ToolProgress { name, .. } if name == "get_balances"
        ));
        assert!(matches!(&events[1], UiEvent::ClearToolProgress));
        assert!(matches!(
            &events[2],
            UiEvent::ToolResult { name, is_error, .. }
            if name == "get_balances" && !is_error
        ));
    }

    #[test]
    fn test_mock_ui_records_warning_and_error() {
        let ui = MockUi::new(vec![]);
        ui.show_warning("generation truncated");
        ui.show_error("something went wrong");
        ui.show_info("just FYI");

        let events = ui.events();
        assert_eq!(
            events,
            vec![
                UiEvent::Warning("generation truncated".into()),
                UiEvent::Error("something went wrong".into()),
                UiEvent::Info("just FYI".into()),
            ]
        );
    }
}
