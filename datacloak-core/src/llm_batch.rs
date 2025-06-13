//! Batch LLM client for efficient API calls

use crate::obfuscator::ObfuscatedChurnPrediction;
use crate::{DataCloakError, ObfuscatedBatch, ObfuscatedRecord, Result};
use backoff::{future::retry, ExponentialBackoff};
use futures::stream::{self, Stream, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;
use tokio::time::sleep;
use tracing::{debug, warn};

/// Configuration for batch LLM processing
#[derive(Debug, Clone)]
pub struct LlmBatchConfig {
    /// API endpoint URL
    pub endpoint: String,
    /// API key
    pub api_key: String,
    /// Model to use
    pub model: String,
    /// Maximum records per batch
    pub batch_size: usize,
    /// Maximum concurrent API calls
    pub max_concurrent_calls: usize,
    /// Request timeout
    pub timeout: Duration,
    /// Rate limit (requests per second)
    pub rate_limit: Option<f32>,
    /// System prompt for churn analysis
    pub system_prompt: String,
}

impl Default for LlmBatchConfig {
    fn default() -> Self {
        Self {
            endpoint: "https://api.openai.com/v1/chat/completions".to_string(),
            api_key: String::new(),
            model: "gpt-4".to_string(),
            batch_size: 10,
            max_concurrent_calls: 5,
            timeout: Duration::from_secs(60),
            rate_limit: Some(10.0),
            system_prompt: "You are a customer churn prediction expert. Analyze the provided customer data and predict the likelihood of churn. Return a JSON object with 'churn_probability' (0.0-1.0), 'confidence' (0.0-1.0), and 'reasoning' fields.".to_string(),
        }
    }
}

/// Batch LLM client for processing obfuscated data
pub struct BatchLlmClient {
    client: Client,
    config: LlmBatchConfig,
    semaphore: Arc<Semaphore>,
    rate_limiter: Option<Arc<tokio::sync::Mutex<RateLimiter>>>,
}

/// Simple rate limiter
struct RateLimiter {
    rate: f32,
    last_request: tokio::time::Instant,
}

impl RateLimiter {
    fn new(rate: f32) -> Self {
        Self {
            rate,
            last_request: tokio::time::Instant::now(),
        }
    }

    async fn wait_if_needed(&mut self) {
        let min_interval = Duration::from_secs_f32(1.0 / self.rate);
        let elapsed = self.last_request.elapsed();

        if elapsed < min_interval {
            sleep(min_interval - elapsed).await;
        }

        self.last_request = tokio::time::Instant::now();
    }
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    temperature: f32,
    max_tokens: u32,
}

#[derive(Serialize, Deserialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: Message,
}

#[derive(Deserialize)]
struct ChurnAnalysisResponse {
    churn_probability: f32,
    confidence: f32,
    reasoning: String,
}

impl BatchLlmClient {
    /// Create a new batch LLM client
    pub fn new(config: LlmBatchConfig) -> Self {
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .expect("Failed to create HTTP client");

        let semaphore = Arc::new(Semaphore::new(config.max_concurrent_calls));

        let rate_limiter = config
            .rate_limit
            .map(|rate| Arc::new(tokio::sync::Mutex::new(RateLimiter::new(rate))));

        Self {
            client,
            config,
            semaphore,
            rate_limiter,
        }
    }

    /// Process a batch of obfuscated records
    pub async fn process_batch(
        self: Arc<Self>,
        batch: ObfuscatedBatch,
    ) -> Result<Vec<ObfuscatedChurnPrediction>> {
        // Split into smaller batches if needed
        let chunks: Vec<_> = batch
            .chunks(self.config.batch_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        // Process chunks concurrently
        let client = self.clone();
        let results = stream::iter(chunks)
            .map(move |chunk| {
                let client = client.clone();
                async move { client.process_chunk(chunk).await }
            })
            .buffer_unordered(self.config.max_concurrent_calls)
            .collect::<Vec<_>>()
            .await;

        // Flatten results
        let mut predictions = Vec::new();
        for result in results {
            match result {
                Ok(chunk_predictions) => predictions.extend(chunk_predictions),
                Err(e) => {
                    warn!("Failed to process chunk: {}", e);
                    if self.config.rate_limit.is_none() {
                        return Err(e);
                    }
                }
            }
        }

        Ok(predictions)
    }

    /// Process a single chunk
    async fn process_chunk(
        self: Arc<Self>,
        chunk: Vec<ObfuscatedRecord>,
    ) -> Result<Vec<ObfuscatedChurnPrediction>> {
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|e| DataCloakError::Other(format!("Semaphore error: {}", e)))?;

        // Apply rate limiting
        if let Some(ref rate_limiter) = self.rate_limiter {
            rate_limiter.lock().await.wait_if_needed().await;
        }

        // Prepare batch prompt
        let batch_data =
            serde_json::to_string_pretty(&chunk).map_err(DataCloakError::Serialization)?;

        let user_prompt = format!(
            "Analyze the following customer records and predict churn probability for each:\n\n{}",
            batch_data
        );

        // Create request with retry logic
        let request = ChatRequest {
            model: self.config.model.clone(),
            messages: vec![
                Message {
                    role: "system".to_string(),
                    content: self.config.system_prompt.clone(),
                },
                Message {
                    role: "user".to_string(),
                    content: user_prompt,
                },
            ],
            temperature: 0.3,
            max_tokens: 1000,
        };

        let backoff = ExponentialBackoff {
            max_elapsed_time: Some(Duration::from_secs(300)),
            ..Default::default()
        };

        let response = retry(backoff, || async {
            self.make_api_call(&request).await.map_err(|e| {
                if e.to_string().contains("rate limit") {
                    backoff::Error::transient(e)
                } else {
                    backoff::Error::permanent(e)
                }
            })
        })
        .await?;

        // Parse response
        self.parse_response(response, chunk).await
    }

    /// Make API call
    async fn make_api_call(&self, request: &ChatRequest) -> Result<ChatResponse> {
        debug!("Making LLM API call");

        let response = self
            .client
            .post(&self.config.endpoint)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(DataCloakError::LlmApi(format!(
                "API error {}: {}",
                status, body
            )));
        }

        let chat_response: ChatResponse = response.json().await?;
        Ok(chat_response)
    }

    /// Parse LLM response into predictions
    async fn parse_response(
        &self,
        response: ChatResponse,
        original_records: Vec<ObfuscatedRecord>,
    ) -> Result<Vec<ObfuscatedChurnPrediction>> {
        let content = response
            .choices
            .first()
            .ok_or_else(|| DataCloakError::LlmApi("No response from LLM".to_string()))?
            .message
            .content
            .clone();

        // Try to parse as JSON array
        let analyses: Vec<ChurnAnalysisResponse> = serde_json::from_str(&content)
            .map_err(|e| DataCloakError::LlmApi(format!("Failed to parse LLM response: {}", e)))?;

        if analyses.len() != original_records.len() {
            return Err(DataCloakError::LlmApi(format!(
                "Response count mismatch: expected {}, got {}",
                original_records.len(),
                analyses.len()
            )));
        }

        let predictions: Vec<ObfuscatedChurnPrediction> = original_records
            .into_iter()
            .zip(analyses)
            .map(|(record, analysis)| ObfuscatedChurnPrediction {
                customer_id: record.id,
                churn_probability: analysis.churn_probability,
                confidence: analysis.confidence,
                reasoning: analysis.reasoning,
                data: record.data,
            })
            .collect();

        Ok(predictions)
    }

    /// Process a stream of obfuscated batches
    pub fn stream<S>(
        self: Arc<Self>,
        stream: S,
    ) -> impl Stream<Item = Result<Vec<ObfuscatedChurnPrediction>>>
    where
        S: Stream<Item = Result<ObfuscatedBatch>> + Send + 'static,
    {
        let client = self.clone();

        stream.then(move |batch_result| {
            let client = client.clone();
            async move {
                match batch_result {
                    Ok(batch) => client.process_batch(batch).await,
                    Err(e) => Err(e),
                }
            }
        })
    }
}

impl Clone for BatchLlmClient {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            config: self.config.clone(),
            semaphore: self.semaphore.clone(),
            rate_limiter: self.rate_limiter.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = LlmBatchConfig {
            api_key: "test-key".to_string(),
            batch_size: 20,
            ..Default::default()
        };

        assert_eq!(config.batch_size, 20);
        assert_eq!(config.model, "gpt-4");
    }
}
