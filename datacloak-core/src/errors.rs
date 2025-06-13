//! Error types for DataCloak

use thiserror::Error;

pub type Result<T> = std::result::Result<T, DataCloakError>;

#[derive(Error, Debug)]
pub enum DataCloakError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Database error: {0}")]
    Database(String),

    #[error("Invalid pattern: {0}")]
    InvalidPattern(String),

    #[error("LLM API error: {0}")]
    LlmApi(String),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Obfuscation error: {0}")]
    Obfuscation(String),

    #[error("Cache error: {0}")]
    Cache(String),

    #[error("Pattern not found: {0}")]
    PatternNotFound(String),

    #[error("Invalid data format: {0}")]
    InvalidDataFormat(String),

    #[error("Processing timeout")]
    Timeout,

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Other error: {0}")]
    Other(String),
}

impl From<tokio::time::error::Elapsed> for DataCloakError {
    fn from(_: tokio::time::error::Elapsed) -> Self {
        DataCloakError::Timeout
    }
}

impl From<tokio_postgres::Error> for DataCloakError {
    fn from(err: tokio_postgres::Error) -> Self {
        DataCloakError::Database(err.to_string())
    }
}
