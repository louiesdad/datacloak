use crate::{
    AdaptiveSampler, DataCloakError, DetectionResult, PatternDetector, RecordBatch, Result,
    SamplingStrategy,
};
use futures::{Stream, StreamExt};
use std::sync::Arc;
use tokio::sync::{mpsc, Semaphore};
use tracing::{debug, info, warn};

/// Configuration for streaming detection
#[derive(Debug, Clone)]
pub struct StreamDetectionConfig {
    /// Maximum number of concurrent batches in processing
    pub max_concurrent_batches: usize,
    /// Size of the channel buffer
    pub channel_buffer_size: usize,
    /// Batch size for detection
    pub batch_size: usize,
    /// Enable adaptive sampling
    pub use_adaptive_sampling: bool,
    /// Confidence threshold for early stopping
    pub confidence_threshold: f64,
    /// Maximum rows to scan (0 = unlimited)
    pub max_rows_to_scan: usize,
}

impl Default for StreamDetectionConfig {
    fn default() -> Self {
        Self {
            max_concurrent_batches: 4,
            channel_buffer_size: 100,
            batch_size: 1000,
            use_adaptive_sampling: true,
            confidence_threshold: 0.95,
            max_rows_to_scan: 0,
        }
    }
}

/// Streaming detection processor
pub struct StreamDetectionProcessor {
    detector: Arc<PatternDetector>,
    adaptive_sampler: Arc<AdaptiveSampler>,
    config: StreamDetectionConfig,
}

/// Intermediate detection result for streaming
#[derive(Debug, Clone)]
pub struct StreamDetectionUpdate {
    pub rows_processed: usize,
    pub patterns_found: usize,
    pub confidence: f64,
    pub should_stop: bool,
}

impl StreamDetectionProcessor {
    pub fn new(
        detector: Arc<PatternDetector>,
        adaptive_sampler: Arc<AdaptiveSampler>,
        config: StreamDetectionConfig,
    ) -> Self {
        Self {
            detector,
            adaptive_sampler,
            config,
        }
    }

    /// Process a stream of record batches for PII detection
    pub async fn detect_stream<S>(
        &self,
        mut input_stream: S,
        progress_tx: Option<mpsc::Sender<StreamDetectionUpdate>>,
    ) -> Result<DetectionResult>
    where
        S: Stream<Item = Result<RecordBatch>> + Unpin + Send,
    {
        let semaphore = Arc::new(Semaphore::new(self.config.max_concurrent_batches));
        let detector = self.detector.clone();

        let mut cumulative_result = DetectionResult::default();
        let mut total_rows_processed = 0;
        let mut confidence_history = Vec::new();
        let mut should_stop = false;

        info!(
            "Starting streaming detection with config: {:?}",
            self.config
        );

        while let Some(batch_result) = input_stream.next().await {
            if should_stop {
                info!("Early stopping triggered, ending stream processing");
                break;
            }

            // Check max rows limit
            if self.config.max_rows_to_scan > 0
                && total_rows_processed >= self.config.max_rows_to_scan
            {
                info!("Reached max rows limit: {}", self.config.max_rows_to_scan);
                break;
            }

            match batch_result {
                Ok(batch) => {
                    let batch_size = batch.len();

                    // Skip if we would exceed max rows
                    let batch = if self.config.max_rows_to_scan > 0 {
                        let remaining = self.config.max_rows_to_scan - total_rows_processed;
                        if remaining == 0 {
                            break;
                        }
                        // Process only what's needed
                        if batch_size > remaining {
                            batch.into_iter().take(remaining).collect()
                        } else {
                            batch
                        }
                    } else {
                        batch
                    };

                    // Acquire permit for concurrent processing
                    let permit =
                        semaphore.clone().acquire_owned().await.map_err(|e| {
                            DataCloakError::Other(format!("Semaphore error: {}", e))
                        })?;

                    let detector = detector.clone();

                    // Process batch
                    let batch_result = tokio::spawn(async move {
                        let _permit = permit; // Hold permit until done
                        detector.analyze_batch(batch).await
                    })
                    .await
                    .map_err(|e| DataCloakError::Other(format!("Task join error: {}", e)))?;

                    match batch_result {
                        Ok(batch_detection) => {
                            total_rows_processed += batch_size;

                            // Merge results
                            merge_detection_results(&mut cumulative_result, &batch_detection);

                            // Calculate confidence
                            let confidence = calculate_streaming_confidence(
                                &cumulative_result,
                                total_rows_processed,
                            );
                            confidence_history.push(confidence);

                            debug!(
                                "Processed batch: {} rows, confidence: {:.3}",
                                batch_size, confidence
                            );

                            // Check for early stopping
                            if self.config.use_adaptive_sampling {
                                should_stop = should_stop_streaming(
                                    &confidence_history,
                                    self.config.confidence_threshold,
                                );
                            }

                            // Send progress update
                            if let Some(ref tx) = progress_tx {
                                let update = StreamDetectionUpdate {
                                    rows_processed: total_rows_processed,
                                    patterns_found: cumulative_result.total_patterns_detected,
                                    confidence,
                                    should_stop,
                                };
                                let _ = tx.send(update).await;
                            }
                        }
                        Err(e) => {
                            warn!("Failed to process batch: {}", e);
                            // Continue processing other batches
                        }
                    }
                }
                Err(e) => {
                    warn!("Error reading batch from stream: {}", e);
                    // Continue with next batch
                }
            }
        }

        // Finalize result
        cumulative_result.sampling_strategy = Some(if should_stop {
            SamplingStrategy::EarlyStop
        } else if self.config.use_adaptive_sampling {
            SamplingStrategy::Adaptive
        } else {
            SamplingStrategy::Fixed(self.config.batch_size)
        });

        cumulative_result.rows_scanned = Some(total_rows_processed);
        if let Some(last_confidence) = confidence_history.last() {
            cumulative_result.confidence_score = Some(*last_confidence);
        }

        info!(
            "Streaming detection complete: {} rows processed, {} patterns found",
            total_rows_processed, cumulative_result.total_patterns_detected
        );

        Ok(cumulative_result)
    }

    /// Create a detection pipeline that returns a stream of results
    pub fn create_pipeline<S>(&self, input_stream: S) -> impl Stream<Item = Result<DetectionResult>>
    where
        S: Stream<Item = Result<RecordBatch>> + Unpin + Send + 'static,
    {
        let (tx, mut rx) = mpsc::channel(self.config.channel_buffer_size);
        let processor = self.clone();

        tokio::spawn(async move {
            let result = processor.detect_stream(input_stream, None).await;
            let _ = tx.send(result).await;
        });

        async_stream::stream! {
            while let Some(result) = rx.recv().await {
                yield result;
            }
        }
    }
}

impl Clone for StreamDetectionProcessor {
    fn clone(&self) -> Self {
        Self {
            detector: self.detector.clone(),
            adaptive_sampler: self.adaptive_sampler.clone(),
            config: self.config.clone(),
        }
    }
}

/// Merge detection results from a batch into cumulative result
fn merge_detection_results(target: &mut DetectionResult, source: &DetectionResult) {
    // Merge pattern counts
    for (pattern, count) in &source.pattern_counts {
        *target.pattern_counts.entry(*pattern).or_insert(0) += count;
    }

    // Merge column patterns
    for (column, patterns) in &source.column_patterns {
        let target_patterns = target
            .column_patterns
            .entry(column.clone())
            .or_insert_with(Vec::new);
        for pattern in patterns {
            if !target_patterns.contains(pattern) {
                target_patterns.push(*pattern);
            }
        }
    }

    // Merge sample matches (keep first 10 of each type)
    for (pattern, matches) in &source.sample_matches {
        let target_matches = target
            .sample_matches
            .entry(*pattern)
            .or_insert_with(Vec::new);
        for match_str in matches {
            if target_matches.len() < 10 && !target_matches.contains(match_str) {
                target_matches.push(match_str.clone());
            }
        }
    }

    // Update totals
    target.total_patterns_detected += source.total_patterns_detected;

    // Merge detected patterns
    for pattern in &source.detected_patterns {
        if !target
            .detected_patterns
            .iter()
            .any(|p| p.pattern_type == pattern.pattern_type)
        {
            target.detected_patterns.push(pattern.clone());
        }
    }

    // Update confidence scores
    for (pattern_type, confidence) in &source.confidence_scores {
        target.confidence_scores.insert(*pattern_type, *confidence);
    }
}

/// Calculate confidence for streaming detection
fn calculate_streaming_confidence(result: &DetectionResult, total_rows: usize) -> f64 {
    if total_rows == 0 {
        return 0.0;
    }

    let pattern_diversity = result.pattern_counts.len() as f64;
    let pattern_density = result.total_patterns_detected as f64 / total_rows as f64;
    let column_coverage = result.column_patterns.len() as f64;

    // Calculate base confidence from pattern detection
    let base_confidence = if pattern_diversity > 0.0 {
        (pattern_density * 100.0).min(1.0)
    } else {
        0.0
    };

    // Adjust for pattern diversity and column coverage
    let diversity_factor = (pattern_diversity / 12.0).min(1.0); // 12 standard pattern types
    let coverage_factor = (column_coverage / 10.0).min(1.0); // Assume 10 columns is good

    // Weighted confidence score
    let confidence = base_confidence * 0.5 + diversity_factor * 0.3 + coverage_factor * 0.2;

    confidence.min(1.0)
}

/// Check if streaming detection should stop early
fn should_stop_streaming(confidence_history: &[f64], threshold: f64) -> bool {
    const WINDOW_SIZE: usize = 5;

    if confidence_history.len() < WINDOW_SIZE {
        return false;
    }

    let last_values = &confidence_history[confidence_history.len() - WINDOW_SIZE..];

    // Check if confidence has plateaued
    let avg_confidence = last_values.iter().sum::<f64>() / last_values.len() as f64;
    let variance = last_values
        .iter()
        .map(|&x| (x - avg_confidence).powi(2))
        .sum::<f64>()
        / last_values.len() as f64;

    // Stop if confidence is high enough and stable
    avg_confidence >= threshold && variance < 0.001
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_confidence_calculation() {
        let mut result = DetectionResult::default();
        assert_eq!(calculate_streaming_confidence(&result, 100), 0.0);

        result.pattern_counts.insert(crate::PatternType::Email, 5);
        result.total_patterns_detected = 5;

        let confidence = calculate_streaming_confidence(&result, 100);
        assert!(confidence > 0.0 && confidence <= 1.0);
    }

    #[test]
    fn test_should_stop_streaming() {
        // Not enough history
        assert!(!should_stop_streaming(&[0.9, 0.91], 0.95));

        // Stable high confidence
        assert!(should_stop_streaming(&[0.95, 0.95, 0.95, 0.95, 0.95], 0.95));

        // Unstable confidence
        assert!(!should_stop_streaming(&[0.5, 0.9, 0.3, 0.8, 0.7], 0.95));
    }
}
