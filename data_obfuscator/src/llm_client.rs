use reqwest::Client;
use thiserror::Error;

pub struct LlmClient {
    endpoint: String,
    api_key: String,
    client: Client,
}

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("invalid response: {0}")]
    InvalidResponse(String),
}

impl LlmClient {
    pub fn new(endpoint: String, api_key: String) -> Self {
        Self { endpoint, api_key, client: Client::new() }
    }

    pub async fn chat(&self, input: &str) -> Result<String, LlmError> {
        // For tests we just echo the input back.
        let _ = (&self.endpoint, &self.api_key); // suppress unused
        Ok(format!("echo: {}", input))
    }
}
