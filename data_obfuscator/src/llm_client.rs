use reqwest::Client;
use serde_json::Value;
use thiserror::Error;
use governor::{Quota, RateLimiter};
use governor::state::{InMemoryState, NotKeyed};
use governor::clock::QuantaClock;
use std::time::Duration;
use std::num::NonZeroU32;

pub struct LlmClient {
    pub endpoint: String,
    api_key: String,
    client: Client,
    rate_limiter: RateLimiter<NotKeyed, InMemoryState, QuantaClock>,
}

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("invalid response")]
    InvalidResponse,
    #[error("rate limit exceeded")]
    RateLimitExceeded,
}

impl LlmClient {
    pub fn new(endpoint: String, api_key: String) -> Self {
        Self::with_rate_limit(endpoint, api_key, 3)
    }
    
    pub fn with_rate_limit(endpoint: String, api_key: String, requests_per_second: u32) -> Self {
        // Configure Governor with specified rate limit (default 3 req/s)
        let quota = Quota::per_second(NonZeroU32::new(requests_per_second).unwrap());
        let rate_limiter = RateLimiter::direct(quota);
        
        Self { 
            endpoint, 
            api_key, 
            client: Client::new(),
            rate_limiter,
        }
    }

    pub async fn chat(&self, input: &str) -> Result<String, LlmError> {
        // Wait for rate limiter permission
        self.rate_limiter.until_ready().await;
        
        let request_body = serde_json::json!({
            "model": "gpt-4",
            "messages": [
                { "role": "system", "content": "You are a secure data processor." },
                { "role": "user", "content": input }
            ]
        });

        let response = self
            .client
            .post(&self.endpoint)
            .bearer_auth(&self.api_key)
            .json(&request_body)
            .send()
            .await?;

        // Check for rate limiting response (429) and honor Retry-After header
        if response.status().as_u16() == 429 {
            if let Some(retry_after) = response.headers().get("retry-after") {
                if let Ok(retry_after_str) = retry_after.to_str() {
                    // Parse Retry-After header (can be seconds or HTTP date)
                    let sleep_duration = if let Ok(seconds) = retry_after_str.parse::<u64>() {
                        Duration::from_secs(seconds)
                    } else {
                        // If it's not a number, assume it's a date and default to 60 seconds
                        Duration::from_secs(60)
                    };
                    
                    // Sleep for the specified duration to respect the server's rate limit
                    tokio::time::sleep(sleep_duration).await;
                    return Err(LlmError::RateLimitExceeded);
                }
            }
            return Err(LlmError::RateLimitExceeded);
        }
        
        let response = response.error_for_status()?;
        let resp: Value = response.json().await?;

        let content = resp["choices"][0]["message"]["content"]
            .as_str()
            .ok_or(LlmError::InvalidResponse)?
            .to_string();
        Ok(content)
    }
}