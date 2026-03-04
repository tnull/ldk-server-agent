/// Lightweight markdown-to-ANSI renderer for terminal output.
///
/// Supports:
/// - Headings (`#`, `##`, `###`, etc.) → bold (+ underline for `#`)
/// - Bold (`**text**`) → bold
/// - Emphasis/italic (`*text*`) → italic (ANSI dim as fallback)

const BOLD: &str = "\x1b[1m";
const ITALIC: &str = "\x1b[3m";
const UNDERLINE: &str = "\x1b[4m";
const RESET: &str = "\x1b[0m";

/// A line-buffering markdown renderer for streaming token output.
///
/// Feed it tokens via [`push`]; it calls the provided callback with
/// fully rendered lines as soon as a newline is encountered.
pub struct StreamRenderer {
    buf: String,
}

impl StreamRenderer {
    pub fn new() -> Self {
        Self { buf: String::new() }
    }

    /// Accepts a token fragment. Whenever a complete line is found in the
    /// buffer, it is rendered and forwarded to `emit`.
    pub fn push(&mut self, token: &str, emit: &mut dyn FnMut(&str)) {
        self.buf.push_str(token);

        // Process all complete lines in the buffer.
        while let Some(nl) = self.buf.find('\n') {
            let line: String = self.buf.drain(..=nl).collect();
            // line includes the trailing '\n'
            let rendered = render_line(line.trim_end_matches('\n'));
            emit(&rendered);
            emit("\n");
        }
    }

    /// Flushes any remaining partial line (call at end of generation).
    pub fn flush(&mut self, emit: &mut dyn FnMut(&str)) {
        if !self.buf.is_empty() {
            let rendered = render_line(&self.buf);
            emit(&rendered);
            self.buf.clear();
        }
    }
}

/// Renders a single line of markdown to ANSI-formatted text.
fn render_line(line: &str) -> String {
    // Check for heading: lines starting with one or more '#' followed by space.
    if let Some(rest) = try_strip_heading(line) {
        return rest;
    }

    render_inline(line)
}

/// If the line is a markdown heading, returns the rendered version.
fn try_strip_heading(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    if !trimmed.starts_with('#') {
        return None;
    }

    let hashes = trimmed.bytes().take_while(|&b| b == b'#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }

    let rest = &trimmed[hashes..];
    // Heading must be followed by a space (or be just hashes at end of line).
    if !rest.is_empty() && !rest.starts_with(' ') {
        return None;
    }

    let text = rest.trim();
    let inner = render_inline(text);

    if hashes == 1 {
        // H1: bold + underline
        Some(format!("{UNDERLINE}{BOLD}{inner}{RESET}"))
    } else {
        // H2+: bold
        Some(format!("{BOLD}{inner}{RESET}"))
    }
}

/// Renders inline markdown: `**bold**` and `*italic*`.
fn render_inline(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 32);
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        if bytes[i] == b'*' {
            // Check for ** (bold)
            if i + 1 < len && bytes[i + 1] == b'*' {
                if let Some((content, end)) = find_closing(text, i + 2, "**") {
                    out.push_str(BOLD);
                    out.push_str(content);
                    out.push_str(RESET);
                    i = end;
                    continue;
                }
            }
            // Single * (italic) — but not ** which was already handled
            if let Some((content, end)) = find_closing_single_star(text, i + 1) {
                out.push_str(ITALIC);
                out.push_str(content);
                out.push_str(RESET);
                i = end;
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }

    out
}

/// Finds the closing `marker` starting from position `start`.
/// Returns the content between markers and the position after the closing marker.
fn find_closing<'a>(text: &'a str, start: usize, marker: &str) -> Option<(&'a str, usize)> {
    let rest = &text[start..];
    // Don't match empty content (e.g., `****`)
    if rest.starts_with(marker) {
        return None;
    }
    let close = rest.find(marker)?;
    if close == 0 {
        return None;
    }
    Some((&rest[..close], start + close + marker.len()))
}

/// Finds closing single `*` that is not part of `**`.
fn find_closing_single_star(text: &str, start: usize) -> Option<(&str, usize)> {
    let rest = &text[start..];
    // Don't match if content starts with `*` (that's a `**` sequence).
    if rest.starts_with('*') {
        return None;
    }
    let bytes = rest.as_bytes();
    for j in 0..bytes.len() {
        if bytes[j] == b'*' {
            // Make sure this isn't a `**`
            if j + 1 < bytes.len() && bytes[j + 1] == b'*' {
                continue;
            }
            if j == 0 {
                return None; // empty content
            }
            return Some((&rest[..j], start + j + 1));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heading_h1() {
        let result = render_line("# Hello World");
        assert!(result.contains(UNDERLINE));
        assert!(result.contains(BOLD));
        assert!(result.contains("Hello World"));
        assert!(result.contains(RESET));
    }

    #[test]
    fn test_heading_h2() {
        let result = render_line("## Subheading");
        assert!(result.contains(BOLD));
        assert!(result.contains("Subheading"));
        assert!(!result.contains(UNDERLINE));
    }

    #[test]
    fn test_heading_h3() {
        let result = render_line("### Details");
        assert!(result.contains(BOLD));
        assert!(result.contains("Details"));
    }

    #[test]
    fn test_not_a_heading() {
        // No space after #
        let result = render_line("#hashtag");
        assert_eq!(result, "#hashtag");
    }

    #[test]
    fn test_bold() {
        let result = render_line("This is **bold** text");
        assert_eq!(result, format!("This is {BOLD}bold{RESET} text"));
    }

    #[test]
    fn test_italic() {
        let result = render_line("This is *italic* text");
        assert_eq!(result, format!("This is {ITALIC}italic{RESET} text"));
    }

    #[test]
    fn test_bold_and_italic() {
        let result = render_line("**bold** and *italic*");
        assert_eq!(
            result,
            format!("{BOLD}bold{RESET} and {ITALIC}italic{RESET}")
        );
    }

    #[test]
    fn test_no_formatting() {
        let result = render_line("Just plain text");
        assert_eq!(result, "Just plain text");
    }

    #[test]
    fn test_single_star_no_close() {
        let result = render_line("price is 5 * 3");
        assert_eq!(result, "price is 5 * 3");
    }

    #[test]
    fn test_empty_bold_not_matched() {
        let result = render_line("nothing **** here");
        assert_eq!(result, "nothing **** here");
    }

    #[test]
    fn test_heading_with_bold() {
        let result = render_line("## The **important** part");
        assert!(result.contains(BOLD));
        assert!(result.contains("important"));
    }

    #[test]
    fn test_stream_renderer() {
        let mut renderer = StreamRenderer::new();
        let mut output = String::new();

        renderer.push("# Hel", &mut |s| output.push_str(s));
        assert!(output.is_empty(), "no newline yet");

        renderer.push("lo\nsome **bold**\n", &mut |s| output.push_str(s));
        assert!(output.contains("Hello"));
        assert!(output.contains(BOLD));
        assert!(output.contains("bold"));

        renderer.push("trailing", &mut |s| output.push_str(s));
        let before_flush = output.clone();
        renderer.flush(&mut |s| output.push_str(s));
        assert!(output.len() > before_flush.len());
        assert!(output.contains("trailing"));
    }
}
