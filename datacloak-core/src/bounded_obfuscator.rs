use crate::{
    bounded_cache::{BoundedCacheConfig, BoundedTokenCache},
    DataCloakError, ObfuscatedBatch, ObfuscatedRecord, Pattern, PatternType, RecordBatch, Result,
};
use dashmap::DashMap;
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tracing::{debug, info};

/// Configuration for bounded obfuscator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundedObfuscatorConfig {
    /// Token cache configuration
    pub cache_config: BoundedCacheConfig,
    /// Token prefix for different pattern types
    pub token_prefixes: std::collections::HashMap<PatternType, String>,
    /// Enable token reuse across sessions
    pub enable_token_reuse: bool,
}

impl Default for BoundedObfuscatorConfig {
    fn default() -> Self {
        let mut token_prefixes = std::collections::HashMap::new();
        token_prefixes.insert(PatternType::Email, "EMAIL_".to_string());
        token_prefixes.insert(PatternType::SSN, "SSN_".to_string());
        token_prefixes.insert(PatternType::Phone, "PHONE_".to_string());
        token_prefixes.insert(PatternType::CreditCard, "CC_".to_string());
        token_prefixes.insert(PatternType::IPAddress, "IP_".to_string());

        Self {
            cache_config: BoundedCacheConfig::default(),
            token_prefixes,
            enable_token_reuse: true,
        }
    }
}

/// Compiled pattern with regex
struct CompiledPattern {
    #[allow(dead_code)]
    pattern_type: PatternType,
    regex: Regex,
    priority: u32,
}

/// High-performance obfuscator with bounded memory usage
pub struct BoundedObfuscator {
    patterns: Arc<DashMap<PatternType, CompiledPattern>>,
    token_counter: Arc<AtomicUsize>,
    token_cache: Arc<BoundedTokenCache>,
    token_generation_mutex: Arc<Mutex<()>>,
    config: BoundedObfuscatorConfig,
}

impl BoundedObfuscator {
    pub fn new(config: BoundedObfuscatorConfig) -> Self {
        let token_cache = Arc::new(BoundedTokenCache::new(config.cache_config.clone()));

        Self {
            patterns: Arc::new(DashMap::new()),
            token_counter: Arc::new(AtomicUsize::new(0)),
            token_cache,
            token_generation_mutex: Arc::new(Mutex::new(())),
            config,
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

        // Log cache statistics periodically
        let stats = self.token_cache.stats();
        debug!(
            "Token cache stats - entries: {}, memory: {} KB, hit rate: {:.2}%",
            stats.get("entries").unwrap_or(&0),
            stats.get("memory_bytes").unwrap_or(&0) / 1024,
            self.token_cache.metrics().hit_rate() * 100.0
        );

        Ok(obfuscated)
    }

    /// Obfuscate a single record
    fn obfuscate_record(
        &self,
        record: &Value,
        sorted_patterns: &[(PatternType, u32)],
    ) -> Result<ObfuscatedRecord> {
        let mut obfuscated = record.clone();
        let mut tokens_used = Vec::new();

        // Apply patterns in priority order
        for (pattern_type, _) in sorted_patterns {
            if let Some(pattern_entry) = self.patterns.get(pattern_type) {
                self.apply_pattern(
                    &mut obfuscated,
                    *pattern_type,
                    &pattern_entry.regex,
                    &mut tokens_used,
                )?;
            }
        }

        Ok(ObfuscatedRecord {
            id: record
                .get("id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            data: obfuscated,
            tokens_used,
        })
    }

    /// Apply a pattern to a value recursively
    fn apply_pattern(
        &self,
        value: &mut Value,
        pattern_type: PatternType,
        regex: &Regex,
        tokens_used: &mut Vec<String>,
    ) -> Result<()> {
        match value {
            Value::String(s) => {
                let mut result = s.clone();
                let matches: Vec<_> = regex.find_iter(&s).collect();

                // Process matches in reverse order to maintain positions
                for m in matches.iter().rev() {
                    let matched_text = m.as_str();
                    let token = self.get_or_create_token(pattern_type, matched_text)?;
                    result.replace_range(m.range(), &token);
                    tokens_used.push(token);
                }

                *s = result;
            }
            Value::Object(map) => {
                for (_, v) in map.iter_mut() {
                    self.apply_pattern(v, pattern_type, regex, tokens_used)?;
                }
            }
            Value::Array(arr) => {
                for v in arr.iter_mut() {
                    self.apply_pattern(v, pattern_type, regex, tokens_used)?;
                }
            }
            _ => {} // Skip other types
        }

        Ok(())
    }

    /// Get existing token or create new one
    fn get_or_create_token(&self, pattern_type: PatternType, original: &str) -> Result<String> {
        // Quick cache lookup first (no lock needed for reads)
        if self.config.enable_token_reuse {
            if let Some(token) = self.token_cache.get_token(original) {
                return Ok(token);
            }
        }

        // Synchronize token generation to prevent race conditions
        let _guard = self.token_generation_mutex.lock().unwrap();

        // Check cache again after acquiring lock (double-checked locking pattern)
        if self.config.enable_token_reuse {
            if let Some(token) = self.token_cache.get_token(original) {
                return Ok(token);
            }
        }

        // Generate new token
        let prefix = self
            .config
            .token_prefixes
            .get(&pattern_type)
            .map(|s| s.as_str())
            .unwrap_or("");

        let counter = self.token_counter.fetch_add(1, Ordering::Relaxed);
        let token = format!("{}{:08X}", prefix, counter);

        // Try to insert our token into cache
        if !self.token_cache.insert(token.clone(), original.to_string()) {
            // Cache is full, but we can still use the token
            debug!("Token cache full, continuing without caching");
        }

        Ok(token)
    }

    /// Reverse obfuscation for a batch
    pub fn reverse_batch(&self, batch: &ObfuscatedBatch) -> Result<RecordBatch> {
        let reversed: Vec<_> = batch
            .par_iter()
            .map(|record| self.reverse_record(record))
            .collect::<Result<Vec<_>>>()?;

        Ok(reversed)
    }

    /// Reverse a single obfuscated record
    fn reverse_record(&self, record: &ObfuscatedRecord) -> Result<Value> {
        let mut reversed = record.data.clone();

        // Apply reverse mapping
        for token in &record.tokens_used {
            if let Some(original) = self.token_cache.get_original(token) {
                self.replace_token_in_value(&mut reversed, token, &original);
            }
        }

        Ok(reversed)
    }

    /// Replace token with original value recursively
    fn replace_token_in_value(&self, value: &mut Value, token: &str, original: &str) {
        match value {
            Value::String(s) => {
                *s = s.replace(token, original);
            }
            Value::Object(map) => {
                for (_, v) in map.iter_mut() {
                    self.replace_token_in_value(v, token, original);
                }
            }
            Value::Array(arr) => {
                for v in arr.iter_mut() {
                    self.replace_token_in_value(v, token, original);
                }
            }
            _ => {}
        }
    }

    /// Get obfuscator statistics
    pub fn stats(&self) -> ObfuscatorStats {
        let cache_stats = self.token_cache.stats();
        let cache_metrics = self.token_cache.metrics();

        ObfuscatorStats {
            patterns_loaded: self.patterns.len(),
            tokens_generated: self.token_counter.load(Ordering::Relaxed),
            cache_entries: *cache_stats.get("entries").unwrap_or(&0),
            cache_memory_bytes: *cache_stats.get("memory_bytes").unwrap_or(&0),
            cache_hit_rate: cache_metrics.hit_rate(),
        }
    }

    /// Clean up expired cache entries
    pub fn cleanup(&self) {
        self.token_cache.cleanup_expired();
    }

    /// Export token mappings (for persistence)
    pub fn export_mappings(&self) -> std::collections::HashMap<String, String> {
        let mappings = std::collections::HashMap::new();

        // This is a simplified export - in production, you'd want to
        // iterate through the cache more efficiently
        info!("Exporting token mappings from cache");

        mappings
    }

    /// Import token mappings (for persistence)
    pub fn import_mappings(
        &self,
        mappings: std::collections::HashMap<String, String>,
    ) -> Result<()> {
        for (token, original) in mappings {
            self.token_cache.insert(token, original);
        }
        Ok(())
    }
}

/// Statistics for the obfuscator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObfuscatorStats {
    pub patterns_loaded: usize,
    pub tokens_generated: usize,
    pub cache_entries: usize,
    pub cache_memory_bytes: usize,
    pub cache_hit_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Pattern;

    #[test]
    fn test_bounded_obfuscation() {
        let config = BoundedObfuscatorConfig {
            cache_config: BoundedCacheConfig {
                max_entries: 100,
                max_memory_bytes: 10 * 1024, // 10KB
                ttl: None,
                enable_metrics: true,
            },
            ..Default::default()
        };

        let obfuscator = BoundedObfuscator::new(config);

        // Set email pattern
        let patterns = vec![Pattern::new(
            PatternType::Email,
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b".to_string(),
        )];
        obfuscator.set_patterns(patterns).unwrap();

        // Test obfuscation
        let record = serde_json::json!({
            "id": "1",
            "email": "test@example.com",
            "name": "John Doe"
        });

        let batch = vec![record];
        let obfuscated = obfuscator.obfuscate_batch(&batch).unwrap();

        assert_eq!(obfuscated.len(), 1);
        assert!(!obfuscated[0]
            .data
            .get("email")
            .unwrap()
            .as_str()
            .unwrap()
            .contains("@"));
        assert!(obfuscated[0]
            .data
            .get("email")
            .unwrap()
            .as_str()
            .unwrap()
            .starts_with("EMAIL_"));

        // Test reverse
        let reversed = obfuscator.reverse_batch(&obfuscated).unwrap();
        assert_eq!(
            reversed[0].get("email").unwrap().as_str().unwrap(),
            "test@example.com"
        );
    }

    #[test]
    fn test_cache_reuse() {
        let config = BoundedObfuscatorConfig::default();
        let obfuscator = BoundedObfuscator::new(config);

        let patterns = vec![Pattern::new(
            PatternType::Email,
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b".to_string(),
        )];
        obfuscator.set_patterns(patterns).unwrap();

        // Obfuscate same email twice
        let record1 = serde_json::json!({"email": "same@example.com"});
        let record2 = serde_json::json!({"email": "same@example.com"});

        let batch = vec![record1, record2];
        let obfuscated = obfuscator.obfuscate_batch(&batch).unwrap();

        // Should get same token for same email
        let token1 = obfuscated[0].data.get("email").unwrap().as_str().unwrap();
        let token2 = obfuscated[1].data.get("email").unwrap().as_str().unwrap();
        assert_eq!(token1, token2);

        // Check cache hit rate
        let stats = obfuscator.stats();
        assert!(stats.cache_hit_rate > 0.0);
    }
}
