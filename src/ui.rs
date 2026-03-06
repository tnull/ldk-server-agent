//! Abstracts all user-facing I/O behind a trait so that the REPL, conversation
//! loop, and LLM engine can be tested without a real terminal.

use std::sync::Mutex;
use std::time::Duration;

/// The result of reading user input.
pub enum InputAction {
    /// The user submitted a non-empty message.
    Message(String),
    /// The user submitted an empty line (no-op).
    Empty,
    /// The user pressed Ctrl-C (interrupt).
    Interrupt,
    /// The user pressed Ctrl-D (EOF / quit).
    Eof,
    /// A fatal readline error occurred.
    Error(String),
}

/// Abstraction over all user-facing I/O.
///
/// Every method takes `&self` so that implementations can be shared across
/// threads (e.g. the timer thread) via `Arc<dyn UserInterface>`.  Mutable
/// state (e.g. readline, recorded output in the mock) is managed through
/// interior mutability.
pub trait UserInterface: Send + Sync {
    // ── Startup ──────────────────────────────────────────────────────

    /// Display the welcome banner when the REPL starts.
    fn display_banner(&self);

    // ── User input ───────────────────────────────────────────────────

    /// Read user input (blocking).  Implementations handle prompting,
    /// drawing input boxes, history, etc.
    fn read_user_input(&self) -> InputAction;

    /// Highlight / "solidify" the most recently entered input.
    ///
    /// In the terminal UI this reprints the input box in bright white.
    /// Implementations that don't need this can leave it as a no-op.
    fn confirm_user_input(&self, input: &str);

    /// Ask the user to confirm a mutating tool action.
    /// Returns `true` if the user approves.
    fn confirm_action(&self, description: &str) -> bool;

    // ── Assistant response streaming ─────────────────────────────────

    /// Called once before the first token of an assistant response.
    /// (e.g. prints the `assistant> ` prefix.)
    fn begin_assistant_response(&self);

    /// Stream a rendered chunk of the assistant's response.
    fn stream_assistant_token(&self, rendered: &str);

    /// Called after the assistant response is fully streamed.
    fn end_assistant_response(&self);

    // ── Status / progress lines ──────────────────────────────────────

    /// Show or update a status line (e.g. elapsed time, token counts).
    ///
    /// `below_output` indicates whether the status should be rendered
    /// below the streaming output (true) or on the current line (false).
    fn update_status(&self, status: &str, below_output: bool);

    /// Erase the current status line.
    fn clear_status(&self);

    /// Show a live tool-execution progress indicator.
    fn show_tool_progress(&self, name: &str, elapsed: Duration);

    /// Clear the tool-execution progress indicator.
    fn clear_tool_progress(&self);

    /// Display a summary line after a tool call completes.
    fn show_tool_result(&self, name: &str, elapsed: Duration, summary: &str, is_error: bool);

    // ── Informational / diagnostic output ────────────────────────────

    /// Display a timing summary after generation completes.
    fn show_timing(&self, total: Duration, time_to_first_token: Option<Duration>);

    /// Display a warning (e.g. generation truncated).
    fn show_warning(&self, msg: &str);

    /// Display a general informational message.
    fn show_info(&self, msg: &str);

    /// Display an error message.
    fn show_error(&self, msg: &str);

    /// Display command output (e.g. `/help`, `/clear` feedback).
    fn show_command_output(&self, msg: &str);

    // ── Lifecycle ────────────────────────────────────────────────────

    /// Called when the REPL is about to exit.
    fn goodbye(&self);
}

// ═══════════════════════════════════════════════════════════════════════
//  Terminal implementation
// ═══════════════════════════════════════════════════════════════════════

use std::io::Write;

// ANSI escape sequences.
const GREY: &str = "\x1b[90m";
const BRIGHT: &str = "\x1b[97m";
const YELLOW: &str = "\x1b[33m";
const RESET: &str = "\x1b[0m";

/// The real terminal-based UI using ANSI escapes and `rustyline`.
pub struct TerminalUi {
    editor: Mutex<rustyline::DefaultEditor>,
    history_path: std::path::PathBuf,
}

impl TerminalUi {
    pub fn new() -> anyhow::Result<Self> {
        let editor = rustyline::DefaultEditor::new()?;
        let history_path = std::env::var_os("HOME")
            .map(std::path::PathBuf::from)
            .unwrap_or_default()
            .join(".ldk-agent-history");
        let mut ed = editor;
        let _ = ed.load_history(&history_path);
        Ok(Self {
            editor: Mutex::new(ed),
            history_path,
        })
    }

    /// Save readline history on shutdown.
    pub fn save_history(&self) {
        if let Ok(mut ed) = self.editor.lock() {
            let _ = ed.save_history(&self.history_path);
        }
    }
}

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
    let prefix = format!("┌─ {} ", label);
    let prefix_display = 3 + label.len() + 1;
    let suffix = "─┐";
    let suffix_display = 2;
    let fill = width.saturating_sub(prefix_display + suffix_display);
    format!("{}{}{}{}{}", color, prefix, "─".repeat(fill), suffix, RESET)
}

/// Return the colored left-border prompt for rustyline: `│ `
fn box_prompt(solid: bool) -> String {
    let color = if solid { BRIGHT } else { GREY };
    format!("{}│{} ", color, RESET)
}

impl UserInterface for TerminalUi {
    fn display_banner(&self) {
        println!("Lightning Node Assistant");
        println!(
            "Type your questions about your Lightning node. Type /quit to exit, /clear to reset.\n"
        );
    }

    fn read_user_input(&self) -> InputAction {
        let width = term_width();

        // Reserve vertical space: top border + input line.
        const BOTTOM_PADDING: usize = 1;
        let reserve = BOTTOM_PADDING + 2;
        print!("{}\x1b[{}A", "\n".repeat(reserve), reserve);
        let _ = std::io::stdout().flush();

        println!("{}", draw_box_top(width, "you", false));
        print!("{}", box_prompt(false));
        let _ = std::io::stdout().flush();

        let prompt = box_prompt(false);
        let readline = {
            let mut ed = self.editor.lock().unwrap();
            ed.readline(&prompt)
        };

        match readline {
            Ok(line) => {
                let trimmed = line.trim().to_string();
                if trimmed.is_empty() {
                    // Erase the box (top + input = 2 lines).
                    print!("\x1b[1A\x1b[2K\x1b[1A\x1b[2K");
                    let _ = std::io::stdout().flush();
                    InputAction::Empty
                } else {
                    if let Ok(mut ed) = self.editor.lock() {
                        let _ = ed.add_history_entry(&trimmed);
                    }
                    InputAction::Message(trimmed)
                }
            }
            Err(rustyline::error::ReadlineError::Interrupted) => InputAction::Interrupt,
            Err(rustyline::error::ReadlineError::Eof) => InputAction::Eof,
            Err(err) => InputAction::Error(err.to_string()),
        }
    }

    fn confirm_user_input(&self, input: &str) {
        let width = term_width();
        let prompt_display_width = 2; // "│ "
        let content_len = input.len() + prompt_display_width;
        let input_rows = content_len.saturating_sub(1) / width + 1;

        // Move up to the top border and reprint in bright.
        print!(
            "\x1b[{}A\r\x1b[2K{}",
            input_rows,
            draw_box_top(width, "you", true)
        );
        print!("\n\x1b[2K{}{}", box_prompt(true), input);
        println!();
        let _ = std::io::stdout().flush();
    }

    fn confirm_action(&self, description: &str) -> bool {
        println!("\n[Action required] {}", description);
        print!("Proceed? (y/N): ");
        let _ = std::io::stdout().flush();

        let mut input = String::new();
        if std::io::stdin().read_line(&mut input).is_err() {
            return false;
        }
        matches!(input.trim().to_lowercase().as_str(), "y" | "yes")
    }

    fn begin_assistant_response(&self) {
        eprint!("\r\x1b[2K");
        print!("{BRIGHT}assistant>{RESET} ");
        let _ = std::io::stdout().flush();
    }

    fn stream_assistant_token(&self, rendered: &str) {
        print!("{}", rendered);
        let _ = std::io::stdout().flush();
    }

    fn end_assistant_response(&self) {
        let _ = std::io::stdout().flush();
    }

    fn update_status(&self, status: &str, below_output: bool) {
        if below_output {
            eprint!("\x1b[s\n\r\x1b[2K{}{}{}\x1b[u", GREY, status, RESET);
        } else {
            eprint!("\r\x1b[2K{}{}{}", GREY, status, RESET);
        }
        let _ = std::io::stderr().flush();
    }

    fn clear_status(&self) {
        // Clear any status line that sits below the cursor.
        eprint!("\x1b[s\n\r\x1b[2K\x1b[u");
        let _ = std::io::stderr().flush();
    }

    fn show_tool_progress(&self, name: &str, elapsed: Duration) {
        eprint!(
            "\r{GREY}[tool: {} | {:.1}s]{RESET} ",
            name,
            elapsed.as_secs_f32()
        );
        let _ = std::io::stderr().flush();
    }

    fn clear_tool_progress(&self) {
        eprint!("\r\x1b[2K");
        let _ = std::io::stderr().flush();
    }

    fn show_tool_result(&self, name: &str, elapsed: Duration, summary: &str, is_error: bool) {
        if is_error {
            eprintln!(
                "{GREY}[Tool {} returned error after {:.1}s: {}]{RESET}",
                name,
                elapsed.as_secs_f32(),
                summary
            );
        } else {
            eprintln!(
                "{GREY}[Tool {} completed in {:.1}s | response: {}]{RESET}",
                name,
                elapsed.as_secs_f32(),
                summary
            );
        }
    }

    fn show_timing(&self, total: Duration, time_to_first_token: Option<Duration>) {
        if let Some(ttft) = time_to_first_token {
            eprintln!(
                "\n{GREY}[{:.1}s total, {:.1}s to first token]{RESET}",
                total.as_secs_f32(),
                ttft.as_secs_f32()
            );
        } else {
            eprintln!("\n{GREY}[{:.1}s total]{RESET}", total.as_secs_f32());
        }
        println!();
    }

    fn show_warning(&self, msg: &str) {
        eprintln!("\n{YELLOW}[Warning: {}]{RESET}", msg);
    }

    fn show_error(&self, msg: &str) {
        eprint!("\r\x1b[2K");
        eprintln!("{}", msg);
    }

    fn show_info(&self, msg: &str) {
        eprintln!("{GREY}{}{RESET}", msg);
    }

    fn show_command_output(&self, msg: &str) {
        println!("{}", msg);
    }

    fn goodbye(&self) {
        println!("Goodbye.");
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Mock implementation for testing
// ═══════════════════════════════════════════════════════════════════════

/// A recorded event from the UI for test assertions.
#[derive(Debug, Clone, PartialEq)]
pub enum UiEvent {
    Banner,
    BeginAssistantResponse,
    AssistantToken(String),
    EndAssistantResponse,
    UpdateStatus {
        status: String,
        below_output: bool,
    },
    ClearStatus,
    ToolProgress {
        name: String,
        elapsed: Duration,
    },
    ClearToolProgress,
    ToolResult {
        name: String,
        elapsed: Duration,
        summary: String,
        is_error: bool,
    },
    Timing {
        total: Duration,
        ttft: Option<Duration>,
    },
    Warning(String),
    Info(String),
    Error(String),
    CommandOutput(String),
    ConfirmInput(String),
    ConfirmAction {
        description: String,
        result: bool,
    },
    Goodbye,
}

/// Mock UI that records all output events and replays scripted user inputs.
///
/// # Usage
///
/// ```ignore
/// let ui = MockUi::new(vec!["hello".into(), "/quit".into()]);
/// // ... run the REPL with `ui` ...
/// let events = ui.events();
/// assert!(events.iter().any(|e| matches!(e, UiEvent::BeginAssistantResponse)));
/// ```
pub struct MockUi {
    events: Mutex<Vec<UiEvent>>,
    /// Scripted user inputs, consumed in FIFO order.
    inputs: Mutex<Vec<String>>,
    /// Scripted confirm responses, consumed in FIFO order.
    confirms: Mutex<Vec<bool>>,
}

impl MockUi {
    /// Create a new mock with pre-scripted user inputs.
    pub fn new(inputs: Vec<String>) -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            inputs: Mutex::new(inputs),
            confirms: Mutex::new(Vec::new()),
        }
    }

    /// Create a new mock with pre-scripted inputs and confirmation responses.
    pub fn with_confirms(inputs: Vec<String>, confirms: Vec<bool>) -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            inputs: Mutex::new(inputs),
            confirms: Mutex::new(confirms),
        }
    }

    /// Returns all recorded events.
    pub fn events(&self) -> Vec<UiEvent> {
        self.events.lock().unwrap().clone()
    }

    fn record(&self, event: UiEvent) {
        self.events.lock().unwrap().push(event);
    }
}

impl UserInterface for MockUi {
    fn display_banner(&self) {
        self.record(UiEvent::Banner);
    }

    fn read_user_input(&self) -> InputAction {
        let mut inputs = self.inputs.lock().unwrap();
        if inputs.is_empty() {
            return InputAction::Eof;
        }
        let input = inputs.remove(0);
        if input.is_empty() {
            InputAction::Empty
        } else {
            InputAction::Message(input)
        }
    }

    fn confirm_user_input(&self, input: &str) {
        self.record(UiEvent::ConfirmInput(input.to_string()));
    }

    fn confirm_action(&self, description: &str) -> bool {
        let mut confirms = self.confirms.lock().unwrap();
        let result = if confirms.is_empty() {
            false
        } else {
            confirms.remove(0)
        };
        self.record(UiEvent::ConfirmAction {
            description: description.to_string(),
            result,
        });
        result
    }

    fn begin_assistant_response(&self) {
        self.record(UiEvent::BeginAssistantResponse);
    }

    fn stream_assistant_token(&self, rendered: &str) {
        self.record(UiEvent::AssistantToken(rendered.to_string()));
    }

    fn end_assistant_response(&self) {
        self.record(UiEvent::EndAssistantResponse);
    }

    fn update_status(&self, status: &str, below_output: bool) {
        self.record(UiEvent::UpdateStatus {
            status: status.to_string(),
            below_output,
        });
    }

    fn clear_status(&self) {
        self.record(UiEvent::ClearStatus);
    }

    fn show_tool_progress(&self, name: &str, elapsed: Duration) {
        self.record(UiEvent::ToolProgress {
            name: name.to_string(),
            elapsed,
        });
    }

    fn clear_tool_progress(&self) {
        self.record(UiEvent::ClearToolProgress);
    }

    fn show_tool_result(&self, name: &str, elapsed: Duration, summary: &str, is_error: bool) {
        self.record(UiEvent::ToolResult {
            name: name.to_string(),
            elapsed,
            summary: summary.to_string(),
            is_error,
        });
    }

    fn show_timing(&self, total: Duration, time_to_first_token: Option<Duration>) {
        self.record(UiEvent::Timing {
            total,
            ttft: time_to_first_token,
        });
    }

    fn show_warning(&self, msg: &str) {
        self.record(UiEvent::Warning(msg.to_string()));
    }

    fn show_info(&self, msg: &str) {
        self.record(UiEvent::Info(msg.to_string()));
    }

    fn show_error(&self, msg: &str) {
        self.record(UiEvent::Error(msg.to_string()));
    }

    fn show_command_output(&self, msg: &str) {
        self.record(UiEvent::CommandOutput(msg.to_string()));
    }

    fn goodbye(&self) {
        self.record(UiEvent::Goodbye);
    }
}
