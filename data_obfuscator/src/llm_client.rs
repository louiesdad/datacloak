use anyhow::Result;
use reqwest::Client;

pub struct LlmClient {
    endpoint: String,
    api_key: String,
    client: Client,
}

impl LlmClient {
    pub fn new(endpoint: String, api_key: String) -> Self {
        Self { endpoint, api_key, client: Client::new() }
    }

    pub async fn chat(&self, input: &str) -> Result<String> {
        // For tests we just echo the input back.
        let _ = (&self.endpoint, &self.api_key); // suppress unused
        Ok(format!("echo: {}", input))
    }
}
