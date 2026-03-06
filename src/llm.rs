use std::sync::atomic::AtomicU32;
use std::sync::Arc;

/// A message in the conversation history.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

/// The role of a chat message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

impl std::fmt::Display for ChatRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatRole::System => write!(f, "system"),
            ChatRole::User => write!(f, "user"),
            ChatRole::Assistant => write!(f, "assistant"),
            ChatRole::Tool => write!(f, "tool"),
        }
    }
}

/// Live generation statistics shared between the LLM engine and the UI.
///
/// The engine populates the fields as generation progresses; the CLI timer
/// thread reads them to display a consolidated status line.
pub struct GenerationStats {
    /// Number of prompt tokens (set before generation begins).
    pub prompt_tokens: AtomicU32,
    /// Context window size.
    pub context_size: AtomicU32,
    /// Number of messages in the conversation.
    pub message_count: AtomicU32,
    /// Number of tokens generated so far (updated during generation).
    pub generated_tokens: AtomicU32,
}

impl GenerationStats {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            prompt_tokens: AtomicU32::new(0),
            context_size: AtomicU32::new(0),
            message_count: AtomicU32::new(0),
            generated_tokens: AtomicU32::new(0),
        })
    }
}

mod engine {
    use std::io::Write;
    use std::num::NonZeroU32;
    use std::path::Path;
    use std::sync::atomic::Ordering;

    use anyhow::{bail, Context};
    use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
    use llama_cpp_2::context::LlamaContext;
    use llama_cpp_2::llama_backend::LlamaBackend;
    use llama_cpp_2::llama_batch::LlamaBatch;
    use llama_cpp_2::model::params::LlamaModelParams;
    use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
    use llama_cpp_2::sampling::LlamaSampler;
    use llama_cpp_2::LogOptions;

    use super::ChatMessage;
    use crate::mcp::protocol::ToolDefinition;

    const DEFAULT_CONTEXT_SIZE: u32 = 8192;
    const DEFAULT_MAX_GENERATION_TOKENS: u32 = 8192;

    pub struct LlmEngine {
        backend: LlamaBackend,
        model: LlamaModel,
        template: LlamaChatTemplate,
        context_size: u32,
        max_generation_tokens: u32,
        threads: i32,
    }

    impl LlmEngine {
        /// Loads a GGUF model from disk, optionally applying a LoRA adapter.
        pub fn load(
            model_path: &Path,
            lora_path: Option<&Path>,
            context_size: Option<u32>,
            max_generation_tokens: Option<u32>,
            gpu_layers: Option<u32>,
            threads: Option<u32>,
        ) -> anyhow::Result<Self> {
            // Suppress llama.cpp's verbose debug output
            llama_cpp_2::send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));

            let backend = LlamaBackend::init().context("Failed to initialize llama backend")?;

            let n_gpu_layers = gpu_layers.unwrap_or(0);
            let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);

            eprintln!("Loading model from: {}", model_path.display());
            let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
                .map_err(|e| anyhow::anyhow!("Failed to load model: {:?}", e))?;
            eprintln!("Model loaded successfully.");

            let template = model.chat_template(None).unwrap_or_else(|_| {
                LlamaChatTemplate::new("chatml").expect("valid chatml template")
            });

            let ctx_size = context_size.unwrap_or(DEFAULT_CONTEXT_SIZE);
            let max_gen = max_generation_tokens.unwrap_or(DEFAULT_MAX_GENERATION_TOKENS);
            let n_threads = threads.unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|p| p.get() as u32)
                    .unwrap_or(4)
            }) as i32;

            let mut engine = Self {
                backend,
                model,
                template,
                context_size: ctx_size,
                max_generation_tokens: max_gen,
                threads: n_threads,
            };

            if let Some(lora) = lora_path {
                engine.load_lora(lora)?;
            }

            Ok(engine)
        }

        /// Returns the configured context window size in tokens.
        pub fn context_size(&self) -> u32 {
            self.context_size
        }

        fn load_lora(&mut self, lora_path: &Path) -> anyhow::Result<()> {
            eprintln!("Loading LoRA adapter from: {}", lora_path.display());

            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(self.context_size))
                .with_n_threads(self.threads)
                .with_n_threads_batch(self.threads);

            let ctx = self
                .model
                .new_context(&self.backend, ctx_params)
                .map_err(|e| anyhow::anyhow!("Failed to create context for LoRA: {:?}", e))?;

            let mut adapter = self
                .model
                .lora_adapter_init(lora_path)
                .map_err(|e| anyhow::anyhow!("Failed to init LoRA adapter: {:?}", e))?;

            ctx.lora_adapter_set(&mut adapter, 1.0)
                .map_err(|e| anyhow::anyhow!("Failed to apply LoRA adapter: {:?}", e))?;

            eprintln!("LoRA adapter loaded successfully.");
            Ok(())
        }

        /// Generates a response given a conversation history and available tools.
        ///
        /// Calls `on_token` for each generated token fragment (for streaming output).
        /// Populates `stats` with live generation metrics for the UI to display.
        /// Returns the complete generated text.
        pub fn generate(
            &self,
            messages: &[ChatMessage],
            tools: &[ToolDefinition],
            stats: &super::GenerationStats,
            on_token: &mut dyn FnMut(&str),
        ) -> anyhow::Result<String> {
            let llama_messages = messages
                .iter()
                .map(|m| {
                    LlamaChatMessage::new(m.role.to_string(), m.content.clone())
                        .map_err(|e| anyhow::anyhow!("Failed to create chat message: {:?}", e))
                })
                .collect::<anyhow::Result<Vec<_>>>()?;

            let tools_json = Self::build_tools_json(tools);

            let result = self
                .model
                .apply_chat_template_with_tools_oaicompat(
                    &self.template,
                    &llama_messages,
                    tools_json.as_deref(),
                    None,
                    true,
                )
                .map_err(|e| anyhow::anyhow!("Failed to apply chat template: {:?}", e))?;

            let tokens = self
                .model
                .str_to_token(&result.prompt, AddBos::Always)
                .map_err(|e| anyhow::anyhow!("Failed to tokenize prompt: {:?}", e))?;

            // Publish stats for the live status line in the CLI timer thread.
            stats
                .prompt_tokens
                .store(tokens.len() as u32, Ordering::Relaxed);
            stats
                .context_size
                .store(self.context_size, Ordering::Relaxed);
            stats
                .message_count
                .store(messages.len() as u32, Ordering::Relaxed);

            if tokens.len() as u32 >= self.context_size {
                bail!(
                    "Prompt ({} tokens) exceeds context size ({}). \
					 Try shortening the conversation or using /clear.",
                    tokens.len(),
                    self.context_size
                );
            }

            let n_ctx = self.context_size;
            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(NonZeroU32::new(n_ctx))
                .with_n_batch(n_ctx)
                .with_n_threads(self.threads)
                .with_n_threads_batch(self.threads)
                .with_flash_attention_policy(1 /* LLAMA_FLASH_ATTN_TYPE_ENABLED */)
                .with_type_k(KvCacheType::Q8_0)
                .with_type_v(KvCacheType::Q8_0);

            let mut ctx: LlamaContext = self
                .model
                .new_context(&self.backend, ctx_params)
                .map_err(|e| anyhow::anyhow!("Failed to create inference context: {:?}", e))?;

            // Fill batch with prompt tokens
            let mut batch = LlamaBatch::new(n_ctx as usize, 1);
            let last_index = tokens.len() as i32 - 1;
            for (i, token) in (0_i32..).zip(tokens.into_iter()) {
                batch
                    .add(token, i, &[0], i == last_index)
                    .map_err(|e| anyhow::anyhow!("Failed to add token to batch: {:?}", e))?;
            }

            // Process prompt
            ctx.decode(&mut batch)
                .map_err(|e| anyhow::anyhow!("Failed to decode prompt: {:?}", e))?;

            // Set up sampler (no grammar constraint — Qwen3 handles
            // tool-call formatting reliably without it, and the grammar
            // from apply_chat_template_with_tools_oaicompat can trigger
            // assertion failures in llama.cpp's grammar engine)
            let mut sampler = LlamaSampler::chain_simple([
                LlamaSampler::temp(0.7),
                LlamaSampler::top_p(0.9, 1),
                LlamaSampler::dist(42),
            ]);

            // Generate tokens.
            // Cap at the configured max *and* the remaining context window,
            // whichever is smaller, so we never exceed the KV cache.
            let prompt_len = batch.n_tokens();
            let mut n_cur = prompt_len;
            let remaining_ctx = (n_ctx as i32 - n_cur).max(0) as u32;
            let gen_budget = self.max_generation_tokens.min(remaining_ctx);
            let max_tokens = n_cur + gen_budget as i32;
            let mut decoder = encoding_rs::UTF_8.new_decoder();
            let mut generated = String::new();
            let mut stopped_naturally = false;

            while n_cur <= max_tokens {
                let token = sampler.sample(&ctx, batch.n_tokens() - 1);
                sampler.accept(token);

                if self.model.is_eog_token(token) {
                    stopped_naturally = true;
                    break;
                }

                let piece = self
                    .model
                    .token_to_piece(token, &mut decoder, true, None)
                    .map_err(|e| anyhow::anyhow!("Failed to decode token: {:?}", e))?;

                generated.push_str(&piece);
                on_token(&piece);
                stats.generated_tokens.fetch_add(1, Ordering::Relaxed);

                // Check additional stop sequences from the template
                if result
                    .additional_stops
                    .iter()
                    .any(|s| !s.is_empty() && generated.ends_with(s))
                {
                    // Remove the stop sequence from the generated text
                    for stop in &result.additional_stops {
                        if !stop.is_empty() && generated.ends_with(stop) {
                            generated.truncate(generated.len() - stop.len());
                            break;
                        }
                    }
                    stopped_naturally = true;
                    break;
                }

                batch.clear();
                batch
                    .add(token, n_cur, &[0], true)
                    .map_err(|e| anyhow::anyhow!("Failed to add generated token: {:?}", e))?;
                n_cur += 1;

                ctx.decode(&mut batch)
                    .map_err(|e| anyhow::anyhow!("Failed to decode generated token: {:?}", e))?;
            }

            let generated_count = (n_cur - prompt_len) as u32;
            if !stopped_naturally {
                eprintln!(
                    "\n\x1b[33m[Warning: generation truncated after {} tokens \
                     (max_generation_tokens: {}, remaining context: {})]\x1b[0m",
                    generated_count, self.max_generation_tokens, remaining_ctx,
                );
            }

            // Flush stdout so streamed output is visible
            let _ = std::io::stdout().flush();

            Ok(generated)
        }

        fn build_tools_json(tools: &[ToolDefinition]) -> Option<String> {
            if tools.is_empty() {
                return None;
            }

            let tools_array: Vec<serde_json::Value> = tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.input_schema,
                        }
                    })
                })
                .collect();

            Some(serde_json::Value::Array(tools_array).to_string())
        }
    }
}

pub use engine::LlmEngine;
