use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::Context;
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;

use crate::conversation::Conversation;
use crate::llm::LlmEngine;
use crate::markdown::StreamRenderer;
use crate::mcp::McpClient;

const HISTORY_FILE: &str = ".ldk-agent-history";

/// Runs the interactive REPL loop.
pub async fn run_repl(
    llm: LlmEngine,
    mut mcp: McpClient,
    mut conversation: Conversation,
) -> anyhow::Result<()> {
    let mut rl = DefaultEditor::new().context("Failed to initialize readline")?;

    let history_path = dirs_path()
        .map(|p| p.join(HISTORY_FILE))
        .unwrap_or_else(|| HISTORY_FILE.into());

    let _ = rl.load_history(&history_path);

    println!("Lightning Node Assistant");
    println!(
        "Type your questions about your Lightning node. Type /quit to exit, /clear to reset.\n"
    );

    loop {
        let readline = rl.readline("you> ");
        match readline {
            Ok(line) => {
                let input = line.trim();
                if input.is_empty() {
                    continue;
                }

                let _ = rl.add_history_entry(input);

                if input.starts_with('/') {
                    match handle_command(input, &mut conversation) {
                        CommandResult::Continue => continue,
                        CommandResult::Quit => break,
                    }
                }

                let start = Instant::now();
                let got_first_token = Arc::new(AtomicBool::new(false));

                // Spawn a timer thread that shows elapsed seconds while waiting
                let timer_stop = Arc::new(AtomicBool::new(false));
                let timer_stop_clone = Arc::clone(&timer_stop);
                let got_first_clone = Arc::clone(&got_first_token);
                let timer_start = start;
                let timer_handle = std::thread::spawn(move || {
                    while !timer_stop_clone.load(Ordering::Relaxed) {
                        if !got_first_clone.load(Ordering::Relaxed) {
                            let elapsed = timer_start.elapsed().as_secs_f32();
                            eprint!("\rassistant> [processing {:.1}s] ", elapsed);
                            let _ = std::io::stderr().flush();
                        }
                        std::thread::sleep(std::time::Duration::from_millis(100));
                    }
                });

                let got_first_for_cb = Arc::clone(&got_first_token);
                let mut first_token_time = None;
                let mut md_renderer = StreamRenderer::new();

                let result = {
                    let md = &mut md_renderer;
                    let ftt = &mut first_token_time;
                    let mut on_token = |token: &str| {
                        if !got_first_for_cb.load(Ordering::Relaxed) {
                            got_first_for_cb.store(true, Ordering::Relaxed);
                            *ftt = Some(start.elapsed());
                            eprint!("\r\x1b[2K");
                            print!("assistant> ");
                        }
                        md.push(token, &mut |rendered| {
                            print!("{}", rendered);
                            let _ = std::io::stdout().flush();
                        });
                    };

                    let mut confirm_fn =
                        |_name: &str, description: &str, _args: &serde_json::Value| -> bool {
                            confirm_action(description)
                        };

                    conversation
                        .send_message(input, &llm, &mut mcp, &mut on_token, &mut confirm_fn)
                        .await
                };

                // Flush any remaining partial line from the renderer.
                md_renderer.flush(&mut |rendered| {
                    print!("{}", rendered);
                    let _ = std::io::stdout().flush();
                });

                // Stop the timer thread
                timer_stop.store(true, Ordering::Relaxed);
                let _ = timer_handle.join();

                let total = start.elapsed();
                match result {
                    Ok(_response) => {
                        if let Some(ttft) = first_token_time {
                            eprintln!(
                                "\n[{:.1}s total, {:.1}s to first token]",
                                total.as_secs_f32(),
                                ttft.as_secs_f32()
                            );
                        } else {
                            eprintln!("\n[{:.1}s total]", total.as_secs_f32());
                        }
                        println!();
                    }
                    Err(e) => {
                        eprint!("\r\x1b[2K");
                        eprintln!("[Error after {:.1}s: {:#}]\n", total.as_secs_f32(), e);
                    }
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("^C");
                continue;
            }
            Err(ReadlineError::Eof) => {
                break;
            }
            Err(err) => {
                eprintln!("Readline error: {}", err);
                break;
            }
        }
    }

    let _ = rl.save_history(&history_path);
    let _ = mcp.shutdown().await;

    println!("Goodbye.");
    Ok(())
}

enum CommandResult {
    Continue,
    Quit,
}

fn handle_command(input: &str, conversation: &mut Conversation) -> CommandResult {
    match input {
        "/quit" | "/exit" | "/q" => CommandResult::Quit,
        "/clear" | "/reset" => {
            conversation.clear();
            println!("[Conversation cleared]\n");
            CommandResult::Continue
        }
        "/help" => {
            println!("Commands:");
            println!("  /quit, /exit, /q  - Exit the assistant");
            println!("  /clear, /reset    - Clear conversation history");
            println!("  /help             - Show this help message");
            println!();
            CommandResult::Continue
        }
        _ => {
            println!(
                "[Unknown command: {}. Type /help for available commands.]\n",
                input
            );
            CommandResult::Continue
        }
    }
}

/// Prompts the user to confirm a mutating action.
fn confirm_action(description: &str) -> bool {
    println!("\n[Action required] {}", description);
    print!("Proceed? (y/N): ");
    let _ = std::io::stdout().flush();

    let mut input = String::new();
    if std::io::stdin().read_line(&mut input).is_err() {
        return false;
    }

    matches!(input.trim().to_lowercase().as_str(), "y" | "yes")
}

fn dirs_path() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME").map(std::path::PathBuf::from)
}
