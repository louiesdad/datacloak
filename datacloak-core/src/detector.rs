//! Automatic PII pattern detection module

use crate::{DataCloakError, DataSource, PatternType, Result};
use dashmap::DashMap;
use once_cell::sync::Lazy;
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Pre-defined regex patterns for common PII types
static PATTERN_LIBRARY: Lazy<HashMap<PatternType, Vec<&'static str>>> = Lazy::new(|| {
    let mut m = HashMap::new();

    // Email patterns
    m.insert(
        PatternType::Email,
        vec![r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"],
    );

    // SSN patterns
    m.insert(
        PatternType::SSN,
        vec![r"\b\d{3}-\d{2}-\d{4}\b", r"\b\d{9}\b"],
    );

    // Phone patterns
    m.insert(
        PatternType::Phone,
        vec![
            r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            r"\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b",
            r"\b\+?1?\s*\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})\b",
        ],
    );

    // Credit card patterns
    m.insert(
        PatternType::CreditCard,
        vec![
            r"\b4[0-9]{12}(?:[0-9]{3})?\b",     // Visa
            r"\b5[1-5][0-9]{14}\b",             // Mastercard
            r"\b3[47][0-9]{13}\b",              // Amex
            r"\b6(?:011|5[0-9]{2})[0-9]{12}\b", // Discover
        ],
    );

    // IP Address patterns
    m.insert(PatternType::IPAddress, vec![
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
    ]);

    // Date of birth patterns
    m.insert(
        PatternType::DateOfBirth,
        vec![
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
        ],
    );

    // Medical Record Number patterns
    m.insert(
        PatternType::MedicalRecordNumber,
        vec![
            r"\bMRN\s*:?\s*\d{6,10}\b",
            r"\b(?:medical\s*record|patient\s*id)\s*:?\s*\d{6,10}\b",
        ],
    );

    // Driver's License patterns (generic)
    m.insert(
        PatternType::DriversLicense,
        vec![r"\b[A-Z]{1,2}\d{6,8}\b", r"\bDL\s*:?\s*[A-Z0-9]{6,12}\b"],
    );

    // Bank Account patterns
    m.insert(
        PatternType::BankAccount,
        vec![
            r"\b\d{8,17}\b", // Generic account number
            r"\bIBAN\s*:?\s*[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b",
        ],
    );

    // Passport patterns
    m.insert(
        PatternType::Passport,
        vec![
            r"\b[A-Z]{1,2}\d{6,9}\b",
            r"\bpassport\s*:?\s*[A-Z0-9]{6,9}\b",
        ],
    );

    m
});

/// Pattern detector for automatic PII identification
pub struct PatternDetector {
    confidence_threshold: f32,
    compiled_patterns: DashMap<PatternType, Vec<Regex>>,
}

/// Detection result for a data source
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DetectionResult {
    pub detected_patterns: Vec<DetectedPattern>,
    pub sample_size: usize,
    pub confidence_scores: HashMap<PatternType, f32>,
    pub recommendations: Vec<PatternRecommendation>,
    pub pattern_counts: HashMap<PatternType, usize>,
    pub column_patterns: HashMap<String, Vec<PatternType>>,
    pub sample_matches: HashMap<PatternType, Vec<String>>,
    pub total_patterns_detected: usize,
    pub sampling_strategy: Option<crate::adaptive_sampling::SamplingStrategy>,
    pub rows_scanned: Option<usize>,
    pub confidence_score: Option<f64>,
}

/// A detected pattern with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedPattern {
    pub pattern_type: PatternType,
    pub regex: String,
    pub match_count: usize,
    pub sample_matches: Vec<String>,
    pub confidence: f32,
    pub column_name: Option<String>,
}

/// Pattern recommendation for user
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternRecommendation {
    pub pattern_type: PatternType,
    pub confidence: f32,
    pub reason: String,
    pub suggested_regex: String,
    pub sample_matches: Vec<String>,
}

impl PatternDetector {
    /// Create a new pattern detector
    pub fn new(confidence_threshold: f32) -> Self {
        let compiled_patterns = DashMap::new();

        // Pre-compile all patterns
        for (pattern_type, patterns) in PATTERN_LIBRARY.iter() {
            let compiled: Vec<Regex> = patterns.iter().filter_map(|p| Regex::new(p).ok()).collect();
            compiled_patterns.insert(*pattern_type, compiled);
        }

        Self {
            confidence_threshold,
            compiled_patterns,
        }
    }

    /// Analyze a batch of records for adaptive sampling
    pub async fn analyze_batch(&self, batch: Vec<serde_json::Value>) -> Result<DetectionResult> {
        let sample_size = batch.len();
        let mut result = DetectionResult::default();
        result.sample_size = sample_size;

        // Parallel pattern detection
        let detection_results: Vec<_> = self
            .compiled_patterns
            .iter()
            .par_bridge()
            .map(|entry| {
                let pattern_type = *entry.key();
                let patterns = entry.value();
                let mut total_matches = 0;
                let mut sample_matches = Vec::new();
                let mut column_matches: HashMap<String, usize> = HashMap::new();

                for record in &batch {
                    // Check if record is an object with fields
                    if let Some(obj) = record.as_object() {
                        for (field_name, field_value) in obj {
                            let text = field_value.to_string();
                            for pattern in patterns.iter() {
                                let matches: Vec<_> = pattern.find_iter(&text).collect();
                                if !matches.is_empty() {
                                    *column_matches.entry(field_name.clone()).or_insert(0) +=
                                        matches.len();
                                    total_matches += matches.len();

                                    // Collect sample matches
                                    for m in matches.iter() {
                                        if sample_matches.len() < 10 {
                                            sample_matches.push(m.as_str().to_string());
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        // Fallback for non-object records
                        let text = record.to_string();
                        for pattern in patterns.iter() {
                            let matches: Vec<_> = pattern.find_iter(&text).collect();
                            total_matches += matches.len();

                            for m in matches.iter().take(10 - sample_matches.len()) {
                                if sample_matches.len() < 10 {
                                    sample_matches.push(m.as_str().to_string());
                                }
                            }
                        }
                    }
                }

                (pattern_type, total_matches, sample_matches, column_matches)
            })
            .collect();

        // Process results
        for (pattern_type, match_count, sample_matches, column_matches) in detection_results {
            if match_count > 0 {
                result.pattern_counts.insert(pattern_type, match_count);
                result
                    .sample_matches
                    .insert(pattern_type, sample_matches.clone());
                result.total_patterns_detected += match_count;

                // Update column patterns
                for (column, _) in column_matches {
                    result
                        .column_patterns
                        .entry(column)
                        .or_insert_with(Vec::new)
                        .push(pattern_type);
                }

                let confidence = (match_count as f32) / (sample_size as f32);
                result.confidence_scores.insert(pattern_type, confidence);

                if confidence >= self.confidence_threshold {
                    result.detected_patterns.push(DetectedPattern {
                        pattern_type,
                        regex: self
                            .compiled_patterns
                            .get(&pattern_type)
                            .and_then(|patterns| {
                                patterns.value().first().map(|r| r.as_str().to_string())
                            })
                            .unwrap_or_default(),
                        match_count,
                        sample_matches: sample_matches.clone(),
                        confidence,
                        column_name: None,
                    });
                }
            }
        }

        Ok(result)
    }

    /// Analyze a data source to detect PII patterns
    pub async fn analyze_source(&self, source: DataSource) -> Result<DetectionResult> {
        let sample_data = source.sample(10000).await?; // Sample first 10k records
        let sample_size = sample_data.len();

        // Parallel pattern detection
        let detection_results: Vec<_> = self
            .compiled_patterns
            .iter()
            .par_bridge()
            .map(|entry| {
                let pattern_type = *entry.key();
                let patterns = entry.value();
                let mut total_matches = 0;
                let mut sample_matches = Vec::new();

                for record in &sample_data {
                    for pattern in patterns {
                        let text = record.to_string();
                        let matches: Vec<_> = pattern.find_iter(&text).collect();
                        total_matches += matches.len();

                        // Collect sample matches (up to 10)
                        for m in matches.iter().take(10 - sample_matches.len()) {
                            if sample_matches.len() < 10 {
                                sample_matches.push(m.as_str().to_string());
                            }
                        }
                    }
                }

                let confidence = (total_matches as f32) / (sample_size as f32);

                DetectedPattern {
                    pattern_type,
                    regex: patterns
                        .first()
                        .map(|r| r.as_str().to_string())
                        .unwrap_or_default(),
                    match_count: total_matches,
                    sample_matches,
                    confidence,
                    column_name: None, // TODO: Implement column detection
                }
            })
            .collect();

        // Filter by confidence threshold and create recommendations
        let mut detected_patterns = Vec::new();
        let mut recommendations = Vec::new();
        let mut confidence_scores = HashMap::new();

        for pattern in detection_results {
            confidence_scores.insert(pattern.pattern_type, pattern.confidence);

            if pattern.confidence >= self.confidence_threshold {
                recommendations.push(PatternRecommendation {
                    pattern_type: pattern.pattern_type,
                    confidence: pattern.confidence,
                    reason: format!(
                        "Found {} matches in sample of {} records",
                        pattern.match_count, sample_size
                    ),
                    suggested_regex: pattern.regex.clone(),
                    sample_matches: pattern.sample_matches.clone(),
                });

                detected_patterns.push(pattern);
            }
        }

        // Sort recommendations by confidence
        recommendations.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        Ok(DetectionResult {
            detected_patterns,
            sample_size,
            confidence_scores,
            recommendations,
            pattern_counts: HashMap::new(),
            column_patterns: HashMap::new(),
            sample_matches: HashMap::new(),
            total_patterns_detected: 0,
            sampling_strategy: None,
            rows_scanned: Some(sample_size),
            confidence_score: None,
        })
    }

    /// Detect patterns in a data source (alias for analyze_source)
    pub async fn detect_patterns(&self, source: DataSource) -> Result<DetectionResult> {
        self.analyze_source(source).await
    }

    /// Add custom pattern for detection
    pub fn add_custom_pattern(&self, pattern_type: PatternType, regex: &str) -> Result<()> {
        let compiled =
            Regex::new(regex).map_err(|e| DataCloakError::InvalidPattern(e.to_string()))?;

        self.compiled_patterns
            .entry(pattern_type)
            .or_default()
            .push(compiled);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_library_compilation() {
        let detector = PatternDetector::new(0.8);
        assert!(!detector.compiled_patterns.is_empty());
    }

    #[test]
    fn test_email_detection() {
        let detector = PatternDetector::new(0.8);
        let email_patterns = detector.compiled_patterns.get(&PatternType::Email).unwrap();

        let test_text = "Contact me at john.doe@example.com";
        let matches: Vec<_> = email_patterns[0].find_iter(test_text).collect();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].as_str(), "john.doe@example.com");
    }
}
