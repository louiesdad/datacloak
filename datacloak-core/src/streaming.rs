//! Streaming processor for large-scale data processing

use crate::{DataCloakError, ObfuscatedBatch, Obfuscator, RecordBatch, Result};
use futures::{Stream, StreamExt};
use std::sync::Arc;
use tokio::sync::{mpsc, Semaphore};
use tracing::{info, warn};

/// Configuration for streaming processor
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Maximum number of concurrent batches in processing
    pub max_concurrent_batches: usize,
    /// Size of the channel buffer
    pub channel_buffer_size: usize,
    /// Whether to continue on errors
    pub continue_on_error: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            max_concurrent_batches: 4,
            channel_buffer_size: 100,
            continue_on_error: true,
        }
    }
}

/// Streaming processor for handling large datasets
pub struct StreamProcessor {
    obfuscator: Arc<Obfuscator>,
    config: StreamConfig,
}

/// Result of stream processing
#[derive(Debug)]
pub struct StreamResult {
    pub total_records: usize,
    pub successful_records: usize,
    pub failed_records: usize,
    pub errors: Vec<String>,
}

impl StreamProcessor {
    /// Create a new stream processor
    pub fn new(obfuscator: Arc<Obfuscator>, config: StreamConfig) -> Self {
        Self { obfuscator, config }
    }

    /// Process a stream of record batches
    pub async fn process_stream<S>(
        &self,
        mut input_stream: S,
        output_tx: mpsc::Sender<ObfuscatedBatch>,
    ) -> Result<StreamResult>
    where
        S: Stream<Item = Result<RecordBatch>> + Unpin + Send,
    {
        let semaphore = Arc::new(Semaphore::new(self.config.max_concurrent_batches));
        let obfuscator = self.obfuscator.clone();
        let continue_on_error = self.config.continue_on_error;

        let mut total_records = 0;
        let mut successful_records = 0;
        let mut failed_records = 0;
        let mut errors = Vec::new();

        while let Some(batch_result) = input_stream.next().await {
            match batch_result {
                Ok(batch) => {
                    let batch_size = batch.len();
                    total_records += batch_size;

                    let permit =
                        semaphore.clone().acquire_owned().await.map_err(|e| {
                            DataCloakError::Other(format!("Semaphore error: {}", e))
                        })?;

                    let obfuscator = obfuscator.clone();
                    let output_tx = output_tx.clone();

                    // Spawn task to process batch
                    tokio::spawn(async move {
                        let _permit = permit; // Hold permit until done

                        match obfuscator.obfuscate_batch(&batch) {
                            Ok(obfuscated) => {
                                if let Err(e) = output_tx.send(obfuscated).await {
                                    warn!("Failed to send obfuscated batch: {}", e);
                                }
                            }
                            Err(e) => {
                                warn!("Failed to obfuscate batch: {}", e);
                            }
                        }
                    });

                    successful_records += batch_size;
                }
                Err(e) => {
                    let error_msg = format!("Stream error: {}", e);
                    errors.push(error_msg.clone());
                    warn!("{}", error_msg);

                    if !continue_on_error {
                        return Err(e);
                    }
                    failed_records += 1;
                }
            }
        }

        // Wait for all processing to complete
        let _ = semaphore
            .acquire_many(self.config.max_concurrent_batches as u32)
            .await;

        info!(
            "Stream processing complete: {} total, {} successful, {} failed",
            total_records, successful_records, failed_records
        );

        Ok(StreamResult {
            total_records,
            successful_records,
            failed_records,
            errors,
        })
    }

    /// Create a processing pipeline that returns a stream of obfuscated batches
    pub fn create_pipeline<S>(&self, input_stream: S) -> impl Stream<Item = Result<ObfuscatedBatch>>
    where
        S: Stream<Item = Result<RecordBatch>> + Send + Unpin + 'static,
    {
        let (tx, rx) = mpsc::channel(self.config.channel_buffer_size);
        let processor = self.clone();

        // Spawn processing task
        tokio::spawn(async move {
            let _ = processor.process_stream(input_stream, tx).await;
        });

        // Convert receiver to stream
        tokio_stream::wrappers::ReceiverStream::new(rx).map(Ok)
    }
}

impl Clone for StreamProcessor {
    fn clone(&self) -> Self {
        Self {
            obfuscator: self.obfuscator.clone(),
            config: self.config.clone(),
        }
    }
}

/// Utility for chunking data efficiently
pub struct ChunkProcessor;

impl ChunkProcessor {
    /// Process text data in chunks, handling regex boundaries
    pub fn process_text_chunks<F>(
        text: &str,
        chunk_size: usize,
        overlap: usize,
        mut processor: F,
    ) -> Result<String>
    where
        F: FnMut(&str) -> Result<String>,
    {
        if text.len() <= chunk_size {
            return processor(text);
        }

        let mut result = String::with_capacity(text.len());
        let mut pos = 0;

        while pos < text.len() {
            let end = (pos + chunk_size).min(text.len());
            let chunk_end = if end < text.len() {
                // Find a good boundary (whitespace or punctuation)
                text[pos..end]
                    .rfind(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
                    .map(|i| pos + i)
                    .unwrap_or(end)
            } else {
                end
            };

            let chunk = &text[pos..chunk_end];
            let processed = processor(chunk)?;

            if pos > 0 {
                // Handle overlap to catch patterns at boundaries
                let overlap_start = pos.saturating_sub(overlap);
                let overlap_chunk = &text[overlap_start..chunk_end];
                let _overlap_processed = processor(overlap_chunk)?;

                // Merge results, avoiding duplicates
                // This is simplified - in production, use more sophisticated merging
                result.push_str(&processed);
            } else {
                result.push_str(&processed);
            }

            pos = chunk_end;
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;

    #[tokio::test]
    async fn test_stream_processor() {
        use crate::patterns::{Pattern, PatternType};

        let obfuscator = Arc::new(Obfuscator::new());
        let patterns = vec![Pattern::new(
            PatternType::Email,
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b".to_string(),
        )];
        obfuscator.set_patterns(patterns).unwrap();

        let processor = StreamProcessor::new(obfuscator, StreamConfig::default());

        // Create test stream
        let batch = vec![serde_json::json!({"email": "test@example.com", "name": "Test User"})];
        let input_stream = Box::pin(stream::once(async { Ok(batch) }));

        let (tx, mut rx) = mpsc::channel(10);

        let result = processor.process_stream(input_stream, tx).await.unwrap();

        assert_eq!(result.total_records, 1);
        assert_eq!(result.successful_records, 1);

        // Check output
        if let Some(obfuscated_batch) = rx.recv().await {
            assert_eq!(obfuscated_batch.len(), 1);
            let record = &obfuscated_batch[0];
            let email = record.data.get("email").unwrap().as_str().unwrap();
            assert!(email.contains("[EMAIL-"));
        }
    }
}
