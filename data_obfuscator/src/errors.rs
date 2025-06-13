use thiserror::Error;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("config error: {0}")]
    Config(#[from] crate::config::ConfigError),
    #[error("obfuscation error: {0}")]
    Obfuscation(#[from] crate::obfuscator::ObfuscationError),
    #[error("llm error: {0}")]
    Llm(#[from] crate::llm_client::LlmError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("other error: {0}")]
    Other(String),
}

