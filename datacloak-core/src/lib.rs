//! DataCloak: High-performance data obfuscation library

pub mod cache;
pub mod crypto;
pub mod data_source;
pub mod detector;
pub mod errors;
pub mod llm_batch;
pub mod obfuscator;
pub mod patterns;
pub mod streaming;

#[cfg(test)]
mod test_coverage_improvements;

// Re-exports
pub use cache::ObfuscationCache;
pub use data_source::{DataSource, DataSourceConfig, RecordBatch};
pub use detector::{DetectionResult, PatternDetector};
pub use errors::{DataCloakError, Result};
pub use llm_batch::{BatchLlmClient, LlmBatchConfig};
pub use obfuscator::{
    ChurnPredictionExport as ChurnPrediction, ObfuscatedBatch, ObfuscatedRecord, Obfuscator,
};
pub use patterns::{Pattern, PatternSet, PatternType};
pub use streaming::{StreamConfig, StreamProcessor};

use futures::StreamExt;
use std::sync::Arc;
use tracing::info;

/// Main configuration for DataCloak
#[derive(Debug, Clone)]
pub struct DataCloakConfig {
    /// Batch size for processing
    pub batch_size: usize,
    /// Maximum concurrent operations
    pub max_concurrency: usize,
    /// LLM configuration
    pub llm_config: LlmBatchConfig,
    /// Stream processing configuration
    pub stream_config: StreamConfig,
}

impl Default for DataCloakConfig {
    fn default() -> Self {
        Self {
            batch_size: 1000,
            max_concurrency: 4,
            llm_config: LlmBatchConfig::default(),
            stream_config: StreamConfig::default(),
        }
    }
}

/// Main DataCloak library interface
pub struct DataCloak {
    detector: Arc<PatternDetector>,
    obfuscator: Arc<Obfuscator>,
    llm_client: Arc<BatchLlmClient>,
    stream_processor: Arc<StreamProcessor>,
    cache: Arc<ObfuscationCache>,
    config: DataCloakConfig,
}

impl DataCloak {
    /// Create a new DataCloak instance
    pub fn new(config: DataCloakConfig) -> Self {
        let detector = Arc::new(PatternDetector::new(0.1)); // 10% confidence threshold
        let obfuscator = Arc::new(Obfuscator::new());
        let llm_client = Arc::new(BatchLlmClient::new(config.llm_config.clone()));
        let stream_processor = Arc::new(StreamProcessor::new(
            obfuscator.clone(),
            config.stream_config.clone(),
        ));
        let cache = Arc::new(ObfuscationCache::new());

        Self {
            detector,
            obfuscator,
            llm_client,
            stream_processor,
            cache,
            config,
        }
    }

    /// Detect PII patterns in a data source
    pub async fn detect_patterns(&self, source: DataSource) -> Result<DetectionResult> {
        self.detector.detect_patterns(source).await
    }

    /// Set patterns for obfuscation
    pub fn set_patterns(&self, patterns: Vec<Pattern>) -> Result<()> {
        self.obfuscator.set_patterns(patterns)
    }

    /// Get obfuscator stats
    pub fn obfuscator_stats(&self) -> crate::obfuscator::ObfuscatorStats {
        self.obfuscator.stats()
    }

    /// Obfuscate a batch of records
    pub async fn obfuscate_batch(&self, batch: RecordBatch) -> Result<ObfuscatedBatch> {
        self.obfuscator.obfuscate_batch(&batch)
    }

    /// Analyze churn probability for customers in a data source
    pub async fn analyze_churn(
        &self,
        source: DataSource,
        patterns: Vec<Pattern>,
        batch_size: Option<usize>,
    ) -> Result<ChurnAnalysisResult> {
        let batch_size = batch_size.unwrap_or(self.config.batch_size);

        // Set patterns in obfuscator
        self.obfuscator.set_patterns(patterns)?;

        // Create data stream
        let data_stream = source.stream(batch_size).await?;

        // Create processing pipeline
        let obfuscated_stream = self.stream_processor.create_pipeline(data_stream);

        // Process through LLM for churn predictions
        let prediction_stream = self.llm_client.clone().stream(obfuscated_stream);

        // Collect results
        let mut all_predictions = Vec::new();
        let mut total_records = 0;
        let mut errors = Vec::new();

        tokio::pin!(prediction_stream);

        while let Some(result) = prediction_stream.next().await {
            match result {
                Ok(predictions) => {
                    total_records += predictions.len();
                    all_predictions.extend(predictions);
                }
                Err(e) => {
                    errors.push(e.to_string());
                    if !self.config.stream_config.continue_on_error {
                        return Err(e);
                    }
                }
            }
        }

        // De-obfuscate predictions
        let final_predictions = self.obfuscator.deobfuscate_predictions(all_predictions)?;

        // Calculate statistics
        let avg_churn_probability = if !final_predictions.is_empty() {
            final_predictions
                .iter()
                .map(|p| p.churn_probability)
                .sum::<f32>()
                / final_predictions.len() as f32
        } else {
            0.0
        };

        let high_risk_count = final_predictions
            .iter()
            .filter(|p| p.churn_probability > 0.7)
            .count();

        info!(
            "Churn analysis complete: {} records processed, avg churn: {:.2}%, {} high risk",
            total_records,
            avg_churn_probability * 100.0,
            high_risk_count
        );

        Ok(ChurnAnalysisResult {
            predictions: final_predictions,
            total_records,
            average_churn_probability: avg_churn_probability,
            high_risk_customers: high_risk_count,
            errors,
        })
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> cache::CacheStats {
        self.cache.stats()
    }

    /// Save cache to persistent storage
    pub async fn save_cache(&self) -> Result<()> {
        self.cache.save().await
    }

    /// Load cache from persistent storage
    pub async fn load_cache(&self) -> Result<()> {
        self.cache.load().await
    }
}

/// Result of churn analysis
#[derive(Debug, Clone)]
pub struct ChurnAnalysisResult {
    /// Individual predictions
    pub predictions: Vec<ChurnPrediction>,
    /// Total records processed
    pub total_records: usize,
    /// Average churn probability across all customers
    pub average_churn_probability: f32,
    /// Number of high-risk customers (>70% churn probability)
    pub high_risk_customers: usize,
    /// Any errors encountered
    pub errors: Vec<String>,
}
