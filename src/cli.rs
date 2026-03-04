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

// ANSI color constants.
const GREY: &str = "\x1b[90m";
const BRIGHT: &str = "\x1b[97m";
const RESET: &str = "\x1b[0m";

/// Returns the terminal width in columns, or 80 as a fallback.
fn term_width() -> usize {
    #[cfg(unix)]
    {
        unsafe {
            let mut ws: libc::winsize = std::mem::zeroed();
            if libc::ioctl(libc::STDOUT_FILENO, libc::TIOCGWINSZ, &mut ws) == 0 && ws.ws_col > 0 {
                return ws.ws_col as usize;
            }
        }
    }
    80
}

/// Draw the top border: `┌─ you ───...─┐`
fn draw_box_top(width: usize, label: &str, solid: bool) -> String {
    let color = if solid { BRIGHT } else { GREY };
    // "┌─ " (4 display cols) + label + " " (1) + "─...─┐" (remaining)
    let prefix = format!("┌─ {} ", label);
    // prefix display width: 4 + label.len() + 1
    let prefix_display = 4 + label.len() + 1;
    let suffix = "─┐";
    let suffix_display = 2;
    let fill = width.saturating_sub(prefix_display + suffix_display);
    format!("{}{}{}{}{}", color, prefix, "─".repeat(fill), suffix, RESET,)
}

/// Draw the bottom border: `└───...───┘`
fn draw_box_bottom(width: usize, solid: bool) -> String {
    let color = if solid { BRIGHT } else { GREY };
    // "└" (1) + "─...─" (fill) + "┘" (1) = width
    let fill = width.saturating_sub(2);
    format!("{}└{}┘{}", color, "─".repeat(fill), RESET)
}

/// Return the colored left-border prompt for rustyline: `│ `
fn box_prompt(solid: bool) -> String {
    let color = if solid { BRIGHT } else { GREY };
    format!("{}│{} ", color, RESET)
}

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
        let width = term_width();

        // Print the grey top border and set up the prompt.
        println!("{}", draw_box_top(width, "you", false));
        let prompt = box_prompt(false);

        let readline = rl.readline(&prompt);
        match readline {
            Ok(line) => {
                let input = line.trim();
                if input.is_empty() {
                    // Erase the partial box: move up 2 lines (top border + input line) and clear.
                    print!("\x1b[1A\x1b[2K\x1b[1A\x1b[2K");
                    let _ = std::io::stdout().flush();
                    continue;
                }

                let _ = rl.add_history_entry(input);

                if input.starts_with('/') {
                    // Close the box with a grey bottom border, then handle the command.
                    println!("{}", draw_box_bottom(width, false));
                    match handle_command(input, &mut conversation) {
                        CommandResult::Continue => continue,
                        CommandResult::Quit => break,
                    }
                }

                // Reprint the box in bright: cursor-up over the input lines + top border.
                // Calculate how many terminal rows the input line occupies.
                // The prompt is "│ " (2 display cols), so content starts at col 3.
                let prompt_display_width = 2; // "│ "
                let content_len = input.len() + prompt_display_width;
                let input_rows = content_len.saturating_sub(1) / width + 1;
                let lines_up = input_rows + 1; // +1 for top border

                // Move up, reprint top border in bright
                print!(
                    "\x1b[{}A\x1b[2K{}",
                    lines_up,
                    draw_box_top(width, "you", true)
                );
                // Move down and reprint each input row in bright
                let bright_prompt = box_prompt(true);
                for _ in 0..input_rows {
                    print!("\n\x1b[2K");
                }
                // Re-position cursor: move back up to the input line, print it
                print!("\x1b[{}A", input_rows);
                print!("\r\x1b[2K{}{}", bright_prompt, input);
                // Move to the end
                for _ in 1..input_rows {
                    print!("\n");
                }
                println!();

                // Print bright bottom border
                println!("{}", draw_box_bottom(width, true));

                let _ = std::io::stdout().flush();

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
                // Ctrl-C: close the box with grey bottom border
                println!();
                println!("{}", draw_box_bottom(width, false));
                continue;
            }
            Err(ReadlineError::Eof) => {
                // Ctrl-D: close the box with grey bottom border
                println!();
                println!("{}", draw_box_bottom(width, false));
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
