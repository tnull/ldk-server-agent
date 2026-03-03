# ldk-server-agent

A local AI assistant for managing your Lightning Network node. Runs a local LLM
(no cloud APIs) that queries live node data via [ldk-server-mcp](https://github.com/lightningdevkit/ldk-server)
and provides expert guidance on channels, payments, balances, and more.

## How It Works

```
┌──────────────────────────────────────────┐
│            ldk-server-agent              │
│                                          │
│  CLI/REPL ──> Conversation Manager       │
│  (rustyline)   (tool-use orchestration)  │
│                    │                     │
│          ┌────────┼────────┐             │
│     LLM Engine  Tool-Call  Safety        │
│     (llama.cpp) Parser     Policy        │
│          └────────┼────────┘             │
│               MCP Client                 │
│          (JSON-RPC 2.0 stdio)            │
└──────────────┬───────────────────────────┘
               │ stdin/stdout
       ┌───────▼────────┐
       │ ldk-server-mcp │ ──> LDK Server (HTTPS)
       └────────────────┘
```

The agent spawns `ldk-server-mcp` as a child process, discovers its 24 tools
(balances, channels, payments, peers, etc.), and gives the LLM access to call
them. Read-only tools execute automatically; mutating operations (sending funds,
opening/closing channels) require explicit user confirmation.

## Prerequisites

- **Rust toolchain** (edition 2024)
- **C++ compiler + CMake + libclang-dev** (for `llama-cpp-2` / llama.cpp bindings)
- **A GGUF model file** (see [Model Selection](#model-selection))
- **ldk-server-mcp** binary on your `PATH` (or provide full path in config)
- **A running LDK Server** with config at `~/.ldk-server/`

### Installing build dependencies (Debian/Ubuntu)

```bash
sudo apt install build-essential cmake libclang-dev
```

## Quick Start

1. **Download a model:**

```bash
mkdir -p ~/.local/share/models
wget -O ~/.local/share/models/qwen3.5-4b-q4_k_m.gguf \
  https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf
```

2. **Create a config file:**

```bash
cp config.example.toml config.toml
# Edit config.toml — set model_path and binary_path
```

Minimal `config.toml`:

```toml
[model]
model_path = "/home/you/.local/share/models/qwen3.5-4b-q4_k_m.gguf"

[mcp]
binary_path = "ldk-server-mcp"
```

The MCP server auto-discovers LDK Server settings from `~/.ldk-server/`
(REST address from `config.toml`, API key from `signet/api_key`, TLS cert
from `tls.crt`).

3. **Build and run:**

```bash
cargo build --release
./target/release/ldk-server-agent
```

4. **Ask questions:**

```
you> What's my node's balance?
assistant> Let me check that for you...
[Calling tool: get_balances | args: {}]
[Tool get_balances completed in 0.3s]
Your node currently has:
- On-chain: 150,000 sats
- Lightning: 500,000 sats across 3 channels
...
```

## Configuration

See [`config.example.toml`](config.example.toml) for all options:

| Section | Key | Description | Default |
|---------|-----|-------------|---------|
| `model` | `model_path` | Path to GGUF model file | (required) |
| `model` | `lora_path` | Path to LoRA adapter file | none |
| `model` | `context_size` | Context window in tokens | 8192 |
| `model` | `gpu_layers` | Layers to offload to GPU | 0 (CPU-only) |
| `model` | `threads` | Inference threads | physical cores |
| `mcp` | `binary_path` | Path to `ldk-server-mcp` binary | (required) |
| `mcp.env` | `key`/`value` | Environment variables for MCP server | none |

## Model Selection

Recommended models (GGUF format, Q4_K_M quantization):

| Model | Size | RAM | Speed | Quality |
|-------|------|-----|-------|---------|
| **Qwen3.5-4B** | 2.7 GB | ~4 GB | Fast | Good tool calling |
| Qwen3-8B | 4.7 GB | ~7 GB | Moderate | Strong tool calling |
| Qwen3-4B | 2.7 GB | ~4 GB | Fast | Decent |

All models are available on [HuggingFace](https://huggingface.co/models?search=gguf).

## REPL Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/clear`, `/reset` | Clear conversation history |
| `/quit`, `/exit`, `/q` | Exit the assistant |

## Safety Policy

Tools are classified by their side effects:

- **Auto-execute (read-only):** `get_node_info`, `get_balances`, `list_channels`,
  `list_payments`, `get_payment_details`, `list_forwarded_payments`,
  `verify_signature`, `export_pathfinding_scores`
- **Require confirmation (mutating):** All operations that send funds, open/close
  channels, change configuration, or connect/disconnect peers

## Building Without LLM Support

If you don't have the C++ toolchain, you can build without the LLM feature
(useful for testing the MCP client):

```bash
cargo build --no-default-features
```

## License

MIT
