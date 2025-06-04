use thiserror::Error;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("config error: {0}")]
    Config(#[from] crate::config::ConfigError),
    #[error("obfuscation error: {0}")]
    Obfuscation(#[from] crate::obfuscator::ObfuscationError),
    #[error("llm error: {0}")]
    Llm(#[from] crate::llm_client::LlmError),
    #[error("other error: {0}")]
    Other(String),
}

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("HTTP request error: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("Invalid response format")]
    InvalidResponse,
}
