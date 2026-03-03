use crate::mcp::protocol::ToolDefinition;

/// Builds the system prompt for the LLM, including tool descriptions
/// from the MCP server.
pub fn build_system_prompt(tools: &[ToolDefinition]) -> String {
    let mut prompt = String::new();

    prompt.push_str(SYSTEM_PREAMBLE);
    prompt.push_str("\n\n# Available Tools\n\n");
    prompt.push_str("You have access to the following tools to query and manage the user's Lightning Network node. ");
    prompt.push_str("To use a tool, output a tool call in this exact format:\n\n");
    prompt.push_str("<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {\"key\": \"value\"}}\n</tool_call>\n\n");
    prompt.push_str(
		"You may call multiple tools in sequence. After each tool call, you will receive the result ",
	);
    prompt.push_str(
        "in a tool response message. Use the results to formulate your answer to the user.\n\n",
    );
    prompt.push_str("IMPORTANT: Always use tool calls to get live data. Never guess or make up node information.\n\n");

    for tool in tools {
        prompt.push_str(&format!("## {}\n", tool.name));
        prompt.push_str(&format!("{}\n", tool.description));

        if let Some(properties) = tool.input_schema.get("properties") {
            if let Some(obj) = properties.as_object() {
                if !obj.is_empty() {
                    prompt.push_str("Parameters:\n");
                    let required: Vec<String> = tool
                        .input_schema
                        .get("required")
                        .and_then(|r| r.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        })
                        .unwrap_or_default();

                    for (param_name, param_schema) in obj {
                        let desc = param_schema
                            .get("description")
                            .and_then(|d| d.as_str())
                            .unwrap_or("");
                        let param_type = param_schema
                            .get("type")
                            .and_then(|t| t.as_str())
                            .unwrap_or("string");
                        let req = if required.contains(param_name) {
                            " (required)"
                        } else {
                            " (optional)"
                        };
                        prompt.push_str(&format!(
                            "  - `{}` ({}{}): {}\n",
                            param_name, param_type, req, desc
                        ));
                    }
                }
            }
        }

        prompt.push('\n');
    }

    prompt.push_str(DOMAIN_KNOWLEDGE);

    prompt
}

const SYSTEM_PREAMBLE: &str = r#"You are a Lightning Network node assistant. You help users understand and manage their Lightning Network node by querying live data and providing expert guidance.

You have deep knowledge of:
- The Lightning Network protocol (BOLT specifications)
- Channel management (opening, closing, rebalancing, liquidity)
- On-chain and off-chain Bitcoin transactions
- Fee policies and routing optimization
- LDK (Lightning Development Kit) specifics

When answering questions:
1. Always query live node data using the available tools rather than guessing
2. Explain technical concepts clearly and concisely
3. Warn about risks before suggesting mutating operations (sending funds, closing channels, etc.)
4. Format large numbers readably (e.g., "1,000,000 sats" rather than "1000000")
5. When displaying channel or payment information, organize it clearly"#;

const DOMAIN_KNOWLEDGE: &str = r#"# Lightning Network Domain Knowledge

## Key Concepts
- **Satoshi (sat)**: Smallest unit of Bitcoin (1 BTC = 100,000,000 sats)
- **Millisatoshi (msat)**: 1/1000 of a satoshi, used for Lightning precision
- **Channel capacity**: Total funds committed to a channel by both parties
- **Local balance**: Funds on your side of the channel (you can send)
- **Remote balance**: Funds on the counterparty's side (you can receive)
- **HTLC**: Hash Time-Locked Contract — the mechanism for routing payments
- **CLTV delta**: Time lock parameter for routing security

## Channel Lifecycle
1. **Opening**: Funds an on-chain transaction, requires confirmations
2. **Active**: Channel is usable for sending/receiving payments
3. **Cooperative close**: Both parties agree, fastest settlement
4. **Force close**: Unilateral close, funds locked for a time-lock period — use only as last resort

## Common Operations
- Check node health: get_node_info + get_balances
- Review channels: list_channels
- Review payment history: list_payments
- Send payments: bolt11_send (invoice) or bolt12_send (offer) or spontaneous_send (keysend)
- Receive payments: bolt11_receive (create invoice) or bolt12_receive (create offer)
- Manage liquidity: splice_in (add funds to channel) or splice_out (remove funds from channel)
"#;
