use crate::{
    DataCloakError, DataSource, DetectionResult, PatternDetector as PIIDetector, PatternType,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    pub min_sample: usize,
    pub max_sample: usize,
    pub confidence_threshold: f64,
    pub progressive_factor: f64,
    pub early_stop_enabled: bool,
    pub confidence_window: usize,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            min_sample: 1000,
            max_sample: 100000,
            confidence_threshold: 0.95,
            progressive_factor: 1.5,
            early_stop_enabled: true,
            confidence_window: 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SamplingStrategy {
    Full,
    Fixed(usize),
    Progressive,
    Adaptive,
    EarlyStop,
}

pub struct AdaptiveSampler {
    config: SamplingConfig,
}

impl AdaptiveSampler {
    pub fn new(config: SamplingConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(SamplingConfig::default())
    }

    pub async fn sample_with_confidence(
        &self,
        source: &mut DataSource,
        detector: &PIIDetector,
    ) -> Result<DetectionResult, DataCloakError> {
        let mut sample_size = self.config.min_sample;
        let mut cumulative_result = DetectionResult::default();
        let mut confidence_history = Vec::new();
        let mut total_rows_scanned = 0;

        tracing::info!(
            "Starting adaptive sampling with min_sample: {}, max_sample: {}",
            self.config.min_sample,
            self.config.max_sample
        );

        while sample_size <= self.config.max_sample {
            // Sample the next batch
            let batch = source.sample(sample_size - total_rows_scanned).await?;
            let batch_size = batch.len();

            if batch_size == 0 {
                tracing::info!("No more data to sample");
                break;
            }

            // Detect PII in the batch
            let batch_result = detector.analyze_batch(batch).await?;
            total_rows_scanned += batch_size;

            // Merge results
            self.merge_results(&mut cumulative_result, &batch_result);

            // Calculate confidence for this iteration
            let confidence = self.calculate_confidence(&cumulative_result, total_rows_scanned);
            confidence_history.push(confidence);

            tracing::debug!(
                "Iteration - Sample size: {}, Confidence: {:.3}, Total rows: {}",
                sample_size,
                confidence,
                total_rows_scanned
            );

            // Check for early termination
            if self.config.early_stop_enabled && self.should_stop(&confidence_history) {
                tracing::info!("Early stopping triggered at confidence: {:.3}", confidence);
                cumulative_result.sampling_strategy = Some(SamplingStrategy::EarlyStop);
                cumulative_result.rows_scanned = Some(total_rows_scanned);
                cumulative_result.confidence_score = Some(confidence);
                break;
            }

            // Calculate next sample size
            sample_size = (sample_size as f64 * self.config.progressive_factor) as usize;
            sample_size = sample_size.min(self.config.max_sample);
        }

        if cumulative_result.sampling_strategy.is_none() {
            cumulative_result.sampling_strategy = Some(SamplingStrategy::Adaptive);
        }

        cumulative_result.rows_scanned = Some(total_rows_scanned);
        if let Some(last_confidence) = confidence_history.last() {
            cumulative_result.confidence_score = Some(*last_confidence);
        }

        Ok(cumulative_result)
    }

    fn merge_results(&self, target: &mut DetectionResult, source: &DetectionResult) {
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
    }

    fn calculate_confidence(&self, result: &DetectionResult, total_rows: usize) -> f64 {
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
        let diversity_factor = (pattern_diversity / PatternType::all().len() as f64).min(1.0);
        let coverage_factor = (column_coverage / 10.0).min(1.0); // Assume 10 columns is good coverage

        // Weighted confidence score
        let confidence = base_confidence * 0.5 + diversity_factor * 0.3 + coverage_factor * 0.2;

        confidence.min(1.0)
    }

    fn should_stop(&self, confidence_history: &[f64]) -> bool {
        if confidence_history.len() < self.config.confidence_window {
            return false;
        }

        let last_values =
            &confidence_history[confidence_history.len() - self.config.confidence_window..];

        // Check if confidence has plateaued
        let avg_confidence = last_values.iter().sum::<f64>() / last_values.len() as f64;
        let variance = last_values
            .iter()
            .map(|&x| (x - avg_confidence).powi(2))
            .sum::<f64>()
            / last_values.len() as f64;

        // Stop if confidence is high enough and stable
        let meets_threshold = avg_confidence >= self.config.confidence_threshold;
        let is_stable = variance < 0.001;
        meets_threshold && is_stable
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_calculation() {
        let sampler = AdaptiveSampler::with_defaults();
        let mut result = DetectionResult::default();

        // Empty result should have 0 confidence
        assert_eq!(sampler.calculate_confidence(&result, 100), 0.0);

        // Add some patterns
        result.pattern_counts.insert(PatternType::Email, 5);
        result.pattern_counts.insert(PatternType::SSN, 3);
        result.total_patterns_detected = 8;

        let confidence = sampler.calculate_confidence(&result, 100);
        assert!(confidence > 0.0 && confidence <= 1.0);
    }

    #[test]
    fn test_should_stop() {
        let config = SamplingConfig {
            min_sample: 1000,
            max_sample: 100000,
            confidence_threshold: 0.90, // Lower threshold for test
            progressive_factor: 1.5,
            early_stop_enabled: true,
            confidence_window: 3,
        };
        let sampler = AdaptiveSampler::new(config);

        // Not enough history
        assert!(!sampler.should_stop(&[0.9, 0.91]));

        // Stable high confidence
        assert!(sampler.should_stop(&[0.95, 0.95, 0.95]));

        // Unstable confidence
        assert!(!sampler.should_stop(&[0.5, 0.9, 0.3]));
    }
}
