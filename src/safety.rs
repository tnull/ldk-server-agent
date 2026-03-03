/// Classifies MCP tools by their safety level for automatic execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolSafety {
    /// Read-only tools that can be executed without user confirmation.
    ReadOnly,
    /// Mutating tools that require explicit user confirmation before execution.
    Mutating,
}

/// Returns the safety classification for a given tool name.
pub fn classify_tool(name: &str) -> ToolSafety {
    match name {
        "get_node_info"
        | "get_balances"
        | "list_channels"
        | "list_payments"
        | "get_payment_details"
        | "list_forwarded_payments"
        | "verify_signature"
        | "export_pathfinding_scores" => ToolSafety::ReadOnly,
        _ => ToolSafety::Mutating,
    }
}

/// Returns a human-readable description of what a mutating tool does,
/// for use in confirmation prompts.
pub fn describe_action(name: &str, args: &serde_json::Value) -> String {
    match name {
        "onchain_send" => {
            let addr = args
                .get("address")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let amount = args
                .get("amount_sats")
                .and_then(|v| v.as_u64())
                .map(|a| format!("{} sats", a))
                .unwrap_or_else(|| "full balance".to_string());
            format!("Send {} on-chain to {}", amount, addr)
        }
        "bolt11_send" => {
            let invoice = args
                .get("invoice")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let short = if invoice.len() > 20 {
                &invoice[..20]
            } else {
                invoice
            };
            format!("Pay BOLT11 invoice {}...", short)
        }
        "bolt12_send" => {
            let offer = args
                .get("offer")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let short = if offer.len() > 20 {
                &offer[..20]
            } else {
                offer
            };
            format!("Pay BOLT12 offer {}...", short)
        }
        "spontaneous_send" => {
            let node_id = args
                .get("node_id")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let amount = args
                .get("amount_msat")
                .and_then(|v| v.as_u64())
                .unwrap_or(0)
                / 1000;
            format!("Send {} sats (keysend) to node {}", amount, node_id)
        }
        "open_channel" => {
            let pubkey = args
                .get("node_pubkey")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let amount = args
                .get("channel_amount_sats")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            format!("Open {} sat channel with {}", amount, pubkey)
        }
        "close_channel" => {
            let channel_id = args
                .get("user_channel_id")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            format!("Cooperatively close channel {}", channel_id)
        }
        "force_close_channel" => {
            let channel_id = args
                .get("user_channel_id")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            format!("FORCE close channel {}", channel_id)
        }
        "splice_in" => {
            let amount = args
                .get("splice_amount_sats")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let channel_id = args
                .get("user_channel_id")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            format!("Splice {} sats into channel {}", amount, channel_id)
        }
        "splice_out" => {
            let amount = args
                .get("splice_amount_sats")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let channel_id = args
                .get("user_channel_id")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            format!("Splice {} sats out of channel {}", amount, channel_id)
        }
        "update_channel_config" => {
            let channel_id = args
                .get("user_channel_id")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            format!("Update config for channel {}", channel_id)
        }
        "connect_peer" => {
            let pubkey = args
                .get("node_pubkey")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            format!("Connect to peer {}", pubkey)
        }
        "disconnect_peer" => {
            let pubkey = args
                .get("node_pubkey")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            format!("Disconnect from peer {}", pubkey)
        }
        "onchain_receive" => "Generate a new on-chain receive address".to_string(),
        "bolt11_receive" => {
            let amount = args
                .get("amount_msat")
                .and_then(|v| v.as_u64())
                .map(|a| format!("{} msat", a))
                .unwrap_or_else(|| "variable amount".to_string());
            format!("Create BOLT11 invoice for {}", amount)
        }
        "bolt12_receive" => {
            let desc = args
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("no description");
            format!("Create BOLT12 offer: {}", desc)
        }
        "sign_message" => {
            let msg = args
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let short = if msg.len() > 40 { &msg[..40] } else { msg };
            format!("Sign message: \"{}\"", short)
        }
        _ => format!("Execute tool: {} with args: {}", name, args),
    }
}
