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
/// Supports two formats:
///
/// **Qwen3 (JSON):**
/// ```text
/// <tool_call>
/// {"name": "tool_name", "arguments": {"key": "value"}}
/// </tool_call>
/// ```
///
/// **Qwen3.5 (function tag):**
/// ```text
/// <tool_call>
/// <function=tool_name>{"key": "value"}</function>
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
                let inner = after_tag[..end_idx].trim();

                if let Ok(call) = parse_inner(inner) {
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

/// Parses the content between `<tool_call>` and `</tool_call>` tags.
/// Tries the Qwen3.5 function-tag format first, then falls back to Qwen3 JSON.
fn parse_inner(inner: &str) -> anyhow::Result<ToolCall> {
    // Try Qwen3.5 format: <function=tool_name>{"key": "value"}</function>
    // Also handles: <function=tool_name></function> (no arguments)
    if let Some(call) = try_parse_function_tag(inner) {
        return Ok(call);
    }

    // Fall back to Qwen3 JSON format: {"name": "...", "arguments": {...}}
    parse_json_tool_call(inner)
}

/// Tries to parse the Qwen3.5 `<function=name>args</function>` format.
fn try_parse_function_tag(inner: &str) -> Option<ToolCall> {
    let inner = inner.trim();

    // Match <function=NAME>
    let func_start = inner.find("<function=")?;
    let after_eq = &inner[func_start + "<function=".len()..];

    // Find the closing > of the opening tag
    let gt_idx = after_eq.find('>')?;
    let name = after_eq[..gt_idx].trim().to_string();

    if name.is_empty() {
        return None;
    }

    let after_gt = &after_eq[gt_idx + 1..];

    // Find </function>
    let end_tag = after_gt.find("</function>")?;
    let args_str = after_gt[..end_tag].trim();

    let arguments = if args_str.is_empty() {
        Value::Object(Default::default())
    } else {
        serde_json::from_str(args_str).ok()?
    };

    if !arguments.is_object() {
        return None;
    }

    Some(ToolCall { name, arguments })
}

/// Parses the Qwen3 JSON format: `{"name": "...", "arguments": {...}}`.
fn parse_json_tool_call(json_str: &str) -> anyhow::Result<ToolCall> {
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

#[cfg(test)]
mod tests {
    use super::*;

    // --- Qwen3 JSON format tests ---

    #[test]
    fn test_parse_single_tool_call_json() {
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
    fn test_parse_tool_call_with_arguments_json() {
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
    fn test_parse_multiple_tool_calls_json() {
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

    // --- Qwen3.5 function-tag format tests ---

    #[test]
    fn test_parse_function_tag_no_args() {
        let text = r#"I'll check your node status.
<tool_call>
<function=get_node_info>
</function>
</tool_call>"#;

        let (calls, remaining) = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_node_info");
        assert_eq!(calls[0].arguments, serde_json::json!({}));
        assert_eq!(remaining, "I'll check your node status.");
    }

    #[test]
    fn test_parse_function_tag_with_args() {
        let text = r#"<tool_call>
<function=bolt11_send>{"invoice": "lnbc1234...", "amount_msat": 50000}
</function>
</tool_call>"#;

        let (calls, remaining) = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bolt11_send");
        assert_eq!(calls[0].arguments["invoice"], "lnbc1234...");
        assert_eq!(calls[0].arguments["amount_msat"], 50000);
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_parse_function_tag_compact() {
        // Some models output this on a single line
        let text = "<tool_call><function=get_balances></function></tool_call>";

        let (calls, remaining) = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_balances");
        assert_eq!(calls[0].arguments, serde_json::json!({}));
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_parse_multiple_function_tags() {
        let text = r#"<tool_call>
<function=get_node_info></function>
</tool_call>
<tool_call>
<function=get_balances></function>
</tool_call>"#;

        let (calls, _) = parse_tool_calls(text);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_node_info");
        assert_eq!(calls[1].name, "get_balances");
    }

    // --- General tests ---

    #[test]
    fn test_no_tool_calls() {
        let text = "Just a regular message with no tool calls.";
        let (calls, remaining) = parse_tool_calls(text);
        assert!(calls.is_empty());
        assert_eq!(remaining, text);
    }
}
