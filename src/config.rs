use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub model: ModelConfig,
    pub mcp: McpConfig,
}

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    /// Path to the GGUF model file.
    pub model_path: PathBuf,
    /// Optional path to a LoRA adapter file.
    pub lora_path: Option<PathBuf>,
    /// Number of context tokens (default: 8192).
    pub context_size: Option<u32>,
    /// Maximum number of tokens to generate per LLM turn (default: 4096).
    pub max_generation_tokens: Option<u32>,
    /// Number of GPU layers to offload (default: 0 for CPU-only).
    pub gpu_layers: Option<u32>,
    /// Number of threads for inference (default: number of physical cores).
    pub threads: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct McpConfig {
    /// Path to the ldk-server-mcp binary.
    pub binary_path: PathBuf,
    /// Environment variables to pass to the MCP server process.
    #[serde(default)]
    pub env: Vec<EnvVar>,
}

#[derive(Debug, Deserialize)]
pub struct EnvVar {
    pub key: String,
    pub value: String,
}

impl Config {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path).context("Failed to read configuration file")?;
        let config: Config = toml::from_str(&content).context("Failed to parse configuration")?;
        Ok(config)
    }
}

impl McpConfig {
    pub fn env_pairs(&self) -> Vec<(String, String)> {
        self.env
            .iter()
            .map(|e| (e.key.clone(), e.value.clone()))
            .collect()
    }
}
