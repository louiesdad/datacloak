use reqwest::Client;
use serde_json::Value;
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
    #[error("invalid response")]
    InvalidResponse,
}

impl LlmClient {
    pub fn new(endpoint: String, api_key: String) -> Self {
        Self { endpoint, api_key, client: Client::new() }
    }

    pub async fn chat(&self, input: &str) -> Result<String, LlmError> {
        let request_body = serde_json::json!({
            "model": "gpt-4",
            "messages": [
                { "role": "system", "content": "You are a secure data processor." },
                { "role": "user", "content": input }
            ]
        });

        let resp: Value = self
            .client
            .post(&self.endpoint)
            .bearer_auth(&self.api_key)
            .json(&request_body)
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;

        let content = resp["choices"][0]["message"]["content"]
            .as_str()
            .ok_or(LlmError::InvalidResponse)?
            .to_string();
        Ok(content)
    }
}