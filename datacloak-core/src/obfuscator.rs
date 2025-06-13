//! High-performance obfuscation engine

use crate::{DataCloakError, Pattern, PatternType, RecordBatch, Result};
use dashmap::DashMap;
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Obfuscated batch of records
pub type ObfuscatedBatch = Vec<ObfuscatedRecord>;

/// An obfuscated record with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObfuscatedRecord {
    pub id: Option<String>,
    pub data: Value,
    pub tokens_used: Vec<String>,
}

/// High-performance obfuscator with parallel processing
pub struct Obfuscator {
    patterns: Arc<DashMap<PatternType, CompiledPattern>>,
    token_counter: Arc<AtomicUsize>,
    token_map: Arc<DashMap<String, String>>, // token -> original
    reverse_map: Arc<DashMap<String, String>>, // original -> token
}

/// Compiled pattern with regex
#[allow(dead_code)]
struct CompiledPattern {
    pattern_type: PatternType,
    regex: Regex,
    priority: u32,
}

impl Default for Obfuscator {
    fn default() -> Self {
        Self::new()
    }
}

impl Obfuscator {
    /// Create a new obfuscator
    pub fn new() -> Self {
        Self {
            patterns: Arc::new(DashMap::new()),
            token_counter: Arc::new(AtomicUsize::new(0)),
            token_map: Arc::new(DashMap::new()),
            reverse_map: Arc::new(DashMap::new()),
        }
    }

    /// Set patterns for obfuscation
    pub fn set_patterns(&self, patterns: Vec<Pattern>) -> Result<()> {
        self.patterns.clear();

        for pattern in patterns {
            let regex = Regex::new(&pattern.regex).map_err(|e| {
                DataCloakError::InvalidPattern(format!("{}: {}", pattern.pattern_type, e))
            })?;

            let compiled = CompiledPattern {
                pattern_type: pattern.pattern_type,
                regex,
                priority: pattern.priority,
            };

            self.patterns.insert(pattern.pattern_type, compiled);
        }

        Ok(())
    }

    /// Obfuscate a batch of records in parallel
    pub fn obfuscate_batch(&self, batch: &RecordBatch) -> Result<ObfuscatedBatch> {
        // Sort patterns by priority for consistent application order
        let mut sorted_patterns: Vec<_> = self
            .patterns
            .iter()
            .map(|entry| (*entry.key(), entry.value().priority))
            .collect();
        sorted_patterns.sort_by(|a, b| b.1.cmp(&a.1));

        // Process records in parallel
        let obfuscated: Vec<_> = batch
            .par_iter()
            .map(|record| self.obfuscate_record(record, &sorted_patterns))
            .collect::<Result<Vec<_>>>()?;

        Ok(obfuscated)
    }

    /// Obfuscate a single record
    fn obfuscate_record(
        &self,
        record: &Value,
        sorted_patterns: &[(PatternType, u32)],
    ) -> Result<ObfuscatedRecord> {
        let mut tokens_used = Vec::new();
        let obfuscated_data = self.obfuscate_value(record, sorted_patterns, &mut tokens_used)?;

        // Extract ID if present
        let id = record
            .get("id")
            .or_else(|| record.get("customer_id"))
            .or_else(|| record.get("_id"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        Ok(ObfuscatedRecord {
            id,
            data: obfuscated_data,
            tokens_used,
        })
    }

    /// Recursively obfuscate a JSON value
    fn obfuscate_value(
        &self,
        value: &Value,
        sorted_patterns: &[(PatternType, u32)],
        tokens_used: &mut Vec<String>,
    ) -> Result<Value> {
        match value {
            Value::String(s) => {
                let obfuscated = self.obfuscate_string(s, sorted_patterns, tokens_used);
                Ok(Value::String(obfuscated))
            }
            Value::Object(map) => {
                let mut obfuscated_map = serde_json::Map::new();
                for (key, val) in map {
                    let obfuscated_val = self.obfuscate_value(val, sorted_patterns, tokens_used)?;
                    obfuscated_map.insert(key.clone(), obfuscated_val);
                }
                Ok(Value::Object(obfuscated_map))
            }
            Value::Array(arr) => {
                let obfuscated_arr: Vec<_> = arr
                    .iter()
                    .map(|v| self.obfuscate_value(v, sorted_patterns, tokens_used))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Value::Array(obfuscated_arr))
            }
            // Numbers, booleans, and nulls pass through unchanged
            _ => Ok(value.clone()),
        }
    }

    /// Obfuscate a string using all patterns
    fn obfuscate_string(
        &self,
        input: &str,
        sorted_patterns: &[(PatternType, u32)],
        tokens_used: &mut Vec<String>,
    ) -> String {
        let mut result = input.to_string();

        for (pattern_type, _) in sorted_patterns {
            if let Some(pattern) = self.patterns.get(pattern_type) {
                let temp_result = pattern
                    .regex
                    .replace_all(&result, |caps: &regex::Captures| {
                        let matched = caps.get(0).unwrap().as_str();

                        // Check if already tokenized
                        if let Some(existing_token) = self.reverse_map.get(matched) {
                            return existing_token.value().clone();
                        }

                        // Generate new token
                        let counter = self.token_counter.fetch_add(1, Ordering::SeqCst);
                        let token = format!("[{}-{}]", pattern_type, counter);

                        // Store mappings
                        self.token_map.insert(token.clone(), matched.to_string());
                        self.reverse_map.insert(matched.to_string(), token.clone());
                        tokens_used.push(token.clone());

                        token
                    });

                result = temp_result.into_owned();
            }
        }

        result
    }

    /// De-obfuscate predictions from LLM
    pub fn deobfuscate_predictions(
        &self,
        predictions: Vec<ObfuscatedChurnPrediction>,
    ) -> Result<Vec<ChurnPrediction>> {
        predictions
            .into_par_iter()
            .map(|pred| {
                let deobfuscated_data = self.deobfuscate_value(&pred.data)?;
                Ok(ChurnPrediction {
                    customer_id: pred.customer_id,
                    churn_probability: pred.churn_probability,
                    confidence: pred.confidence,
                    reasoning: self.deobfuscate_string(&pred.reasoning),
                    data: deobfuscated_data,
                })
            })
            .collect()
    }

    /// De-obfuscate a JSON value
    fn deobfuscate_value(&self, value: &Value) -> Result<Value> {
        match value {
            Value::String(s) => Ok(Value::String(self.deobfuscate_string(s))),
            Value::Object(map) => {
                let mut deobfuscated_map = serde_json::Map::new();
                for (key, val) in map {
                    let deobfuscated_val = self.deobfuscate_value(val)?;
                    deobfuscated_map.insert(key.clone(), deobfuscated_val);
                }
                Ok(Value::Object(deobfuscated_map))
            }
            Value::Array(arr) => {
                let deobfuscated_arr: Vec<_> = arr
                    .iter()
                    .map(|v| self.deobfuscate_value(v))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Value::Array(deobfuscated_arr))
            }
            _ => Ok(value.clone()),
        }
    }

    /// De-obfuscate a string
    fn deobfuscate_string(&self, input: &str) -> String {
        let mut result = input.to_string();

        // Find all tokens in the string
        let token_pattern = Regex::new(r"\[[A-Z_]+-\d+\]").unwrap();

        for cap in token_pattern.captures_iter(input) {
            if let Some(token) = cap.get(0) {
                if let Some(original) = self.token_map.get(token.as_str()) {
                    result = result.replace(token.as_str(), original.value());
                }
            }
        }

        result
    }

    /// Get statistics about the obfuscation cache
    pub fn stats(&self) -> ObfuscatorStats {
        ObfuscatorStats {
            total_tokens: self.token_map.len(),
            patterns_loaded: self.patterns.len(),
            next_token_id: self.token_counter.load(Ordering::SeqCst),
        }
    }

    /// Clear all mappings (use with caution!)
    pub fn clear_mappings(&self) {
        self.token_map.clear();
        self.reverse_map.clear();
        self.token_counter.store(0, Ordering::SeqCst);
    }
}

/// Statistics about the obfuscator
#[derive(Debug, Clone)]
pub struct ObfuscatorStats {
    pub total_tokens: usize,
    pub patterns_loaded: usize,
    pub next_token_id: usize,
}

/// Obfuscated churn prediction from LLM
#[derive(Debug, Clone)]
pub struct ObfuscatedChurnPrediction {
    pub customer_id: Option<String>,
    pub churn_probability: f32,
    pub confidence: f32,
    pub reasoning: String,
    pub data: Value,
}

/// De-obfuscated churn prediction
#[derive(Debug, Clone)]
pub struct ChurnPrediction {
    pub customer_id: Option<String>,
    pub churn_probability: f32,
    pub confidence: f32,
    pub reasoning: String,
    pub data: Value,
}

// Re-export for use in llm_batch module
pub use self::ChurnPrediction as ChurnPredictionExport;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patterns::Pattern;

    #[test]
    fn test_obfuscation() {
        let obfuscator = Obfuscator::new();

        let patterns = vec![
            Pattern::new(
                PatternType::Email,
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b".to_string(),
            ),
            Pattern::new(
                PatternType::Phone,
                r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b".to_string(),
            ),
        ];

        obfuscator.set_patterns(patterns).unwrap();

        let input = "Contact john.doe@example.com or call 555-123-4567";
        let mut tokens = Vec::new();
        let obfuscated = obfuscator.obfuscate_string(
            input,
            &[(PatternType::Email, 100), (PatternType::Phone, 90)],
            &mut tokens,
        );

        assert!(obfuscated.contains("[EMAIL-0]"));
        assert!(obfuscated.contains("[PHONE-1]"));
        assert_eq!(tokens.len(), 2);

        // Test de-obfuscation
        let deobfuscated = obfuscator.deobfuscate_string(&obfuscated);
        assert_eq!(deobfuscated, input);
    }
}
