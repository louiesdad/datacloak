use thiserror::Error;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("config error: {0}")]
    Config(#[from] crate::config::ConfigError),
    #[error("obfuscation error: {0}")]
    Obfuscation(#[from] crate::obfuscator::ObfuscationError),
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("other error: {0}")]
    Other(String),
}
