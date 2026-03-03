use anyhow::{Context, bail};
use serde_json::Value;

/// A parsed tool call extracted from the model's output.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

/// Extracts tool calls from the model's output text.
///
/// Qwen3 uses the format:
/// ```text
/// <tool_call>
/// {"name": "tool_name", "arguments": {"key": "value"}}
/// </tool_call>
/// ```
///
/// Returns a list of parsed tool calls and any remaining text that is not
/// part of a tool call.
pub fn parse_tool_calls(text: &str) -> (Vec<ToolCall>, String) {
    let mut tool_calls = Vec::new();
    let mut remaining = String::new();
    let mut rest = text;

    loop {
        if let Some(start_idx) = rest.find("<tool_call>") {
            // Text before the tool call tag is regular assistant text
            remaining.push_str(&rest[..start_idx]);

            let after_tag = &rest[start_idx + "<tool_call>".len()..];

            if let Some(end_idx) = after_tag.find("</tool_call>") {
                let json_str = after_tag[..end_idx].trim();

                if let Ok(call) = parse_single_tool_call(json_str) {
                    tool_calls.push(call);
                } else {
                    // If parsing fails, treat it as regular text
                    remaining.push_str(&rest[start_idx..start_idx + "<tool_call>".len()]);
                    remaining.push_str(&after_tag[..end_idx]);
                    remaining.push_str("</tool_call>");
                }

                rest = &after_tag[end_idx + "</tool_call>".len()..];
            } else {
                // No closing tag found — treat the rest as regular text
                remaining.push_str(&rest[start_idx..]);
                break;
            }
        } else {
            remaining.push_str(rest);
            break;
        }
    }

    (tool_calls, remaining.trim().to_string())
}

fn parse_single_tool_call(json_str: &str) -> anyhow::Result<ToolCall> {
    let value: Value = serde_json::from_str(json_str).context("Invalid JSON in tool call")?;

    let name = value
        .get("name")
        .and_then(|v| v.as_str())
        .context("Missing 'name' in tool call")?
        .to_string();

    let arguments = value
        .get("arguments")
        .cloned()
        .unwrap_or(Value::Object(Default::default()));

    if !arguments.is_object() {
        bail!("Tool call 'arguments' must be an object");
    }

    Ok(ToolCall { name, arguments })
}

/// Checks if the text appears to contain an incomplete/in-progress tool call.
/// This is useful during streaming to know when to buffer output.
pub fn has_pending_tool_call(text: &str) -> bool {
    let open_count = text.matches("<tool_call>").count();
    let close_count = text.matches("</tool_call>").count();
    open_count > close_count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_single_tool_call() {
        let text = r#"Let me check your balance.
<tool_call>
{"name": "get_balances", "arguments": {}}
</tool_call>"#;

        let (calls, remaining) = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_balances");
        assert_eq!(calls[0].arguments, serde_json::json!({}));
        assert_eq!(remaining, "Let me check your balance.");
    }

    #[test]
    fn test_parse_tool_call_with_arguments() {
        let text = r#"<tool_call>
{"name": "bolt11_send", "arguments": {"invoice": "lnbc1234..."}}
</tool_call>"#;

        let (calls, remaining) = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bolt11_send");
        assert_eq!(calls[0].arguments["invoice"], "lnbc1234...");
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_parse_multiple_tool_calls() {
        let text = r#"<tool_call>
{"name": "get_node_info", "arguments": {}}
</tool_call>
<tool_call>
{"name": "get_balances", "arguments": {}}
</tool_call>"#;

        let (calls, _) = parse_tool_calls(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_node_info");
        assert_eq!(calls[1].name, "get_balances");
    }

    #[test]
    fn test_no_tool_calls() {
        let text = "Just a regular message with no tool calls.";
        let (calls, remaining) = parse_tool_calls(text);
        assert!(calls.is_empty());
        assert_eq!(remaining, text);
    }

    #[test]
    fn test_has_pending_tool_call() {
        assert!(has_pending_tool_call("some text <tool_call> partial"));
        assert!(!has_pending_tool_call(
            "some text <tool_call>...</tool_call>"
        ));
        assert!(!has_pending_tool_call("no tags here"));
    }
}
