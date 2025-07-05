use dashmap::DashMap;
use lru::LruCache;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, warn};

/// Configuration for the bounded token cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundedCacheConfig {
    /// Maximum number of entries in the cache
    pub max_entries: usize,
    /// Maximum memory usage in bytes (0 = unlimited)
    pub max_memory_bytes: usize,
    /// TTL for cache entries (None = no expiration)
    pub ttl: Option<Duration>,
    /// Enable metrics collection
    pub enable_metrics: bool,
}

impl Default for BoundedCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 100_000,
            max_memory_bytes: 512 * 1024 * 1024,  // 512MB
            ttl: Some(Duration::from_secs(3600)), // 1 hour
            enable_metrics: true,
        }
    }
}

/// Cache entry with metadata
#[derive(Debug, Clone)]
struct CacheEntry {
    value: String,
    size_bytes: usize,
    created_at: Instant,
    access_count: usize,
}

/// Metrics for cache performance
#[derive(Debug, Default)]
pub struct CacheMetrics {
    pub hits: AtomicUsize,
    pub misses: AtomicUsize,
    pub evictions: AtomicUsize,
    pub insertions: AtomicUsize,
    pub memory_usage: AtomicUsize,
}

impl Clone for CacheMetrics {
    fn clone(&self) -> Self {
        Self {
            hits: AtomicUsize::new(self.hits.load(Ordering::Relaxed)),
            misses: AtomicUsize::new(self.misses.load(Ordering::Relaxed)),
            evictions: AtomicUsize::new(self.evictions.load(Ordering::Relaxed)),
            insertions: AtomicUsize::new(self.insertions.load(Ordering::Relaxed)),
            memory_usage: AtomicUsize::new(self.memory_usage.load(Ordering::Relaxed)),
        }
    }
}

impl CacheMetrics {
    pub fn hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    pub fn eviction(&self) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
    }

    pub fn insertion(&self) {
        self.insertions.fetch_add(1, Ordering::Relaxed);
    }

    pub fn update_memory(&self, bytes: usize) {
        self.memory_usage.store(bytes, Ordering::Relaxed);
    }

    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed) as f64;
        let total = hits + self.misses.load(Ordering::Relaxed) as f64;
        if total > 0.0 {
            hits / total
        } else {
            0.0
        }
    }
}

/// Memory-bounded token cache with LRU eviction
pub struct BoundedTokenCache {
    /// Token to original value mapping (LRU)
    token_to_original: Arc<Mutex<LruCache<String, CacheEntry>>>,
    /// Original value to token mapping (concurrent)
    original_to_token: Arc<DashMap<String, String>>,
    /// Reverse index for fast lookups
    token_index: Arc<DashMap<String, String>>,
    /// Configuration
    config: BoundedCacheConfig,
    /// Metrics
    metrics: Arc<CacheMetrics>,
    /// Current memory usage estimate
    current_memory: Arc<AtomicUsize>,
}

impl BoundedTokenCache {
    pub fn new(config: BoundedCacheConfig) -> Self {
        let token_to_original = Arc::new(Mutex::new(LruCache::new(
            config
                .max_entries
                .try_into()
                .unwrap_or(std::num::NonZeroUsize::MAX),
        )));

        Self {
            token_to_original,
            original_to_token: Arc::new(DashMap::new()),
            token_index: Arc::new(DashMap::new()),
            config,
            metrics: Arc::new(CacheMetrics::default()),
            current_memory: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Insert a token mapping
    pub fn insert(&self, token: String, original: String) -> bool {
        let entry_size = Self::estimate_entry_size(&token, &original);

        // Check memory limit
        if self.config.max_memory_bytes > 0 {
            let current = self.current_memory.load(Ordering::Relaxed);
            if current + entry_size > self.config.max_memory_bytes {
                // Try to evict entries to make room
                self.evict_to_fit(entry_size);

                // Check again
                let current = self.current_memory.load(Ordering::Relaxed);
                if current + entry_size > self.config.max_memory_bytes {
                    warn!("Cannot insert token: memory limit exceeded");
                    return false;
                }
            }
        }

        let entry = CacheEntry {
            value: original.clone(),
            size_bytes: entry_size,
            created_at: Instant::now(),
            access_count: 0,
        };

        // Insert with LRU eviction if needed
        let mut cache = self.token_to_original.lock();

        // Check if we need to evict due to entry count
        if cache.len() >= self.config.max_entries {
            if let Some((evicted_token, evicted_entry)) = cache.pop_lru() {
                self.handle_eviction(&evicted_token, &evicted_entry);
            }
        }

        cache.put(token.clone(), entry);
        drop(cache);

        // Update other mappings
        self.original_to_token
            .insert(original.clone(), token.clone());
        self.token_index.insert(token, original);

        // Update metrics
        self.current_memory.fetch_add(entry_size, Ordering::Relaxed);
        if self.config.enable_metrics {
            self.metrics.insertion();
            self.metrics
                .update_memory(self.current_memory.load(Ordering::Relaxed));
        }

        true
    }

    /// Get original value for a token
    pub fn get_original(&self, token: &str) -> Option<String> {
        let mut cache = self.token_to_original.lock();

        if let Some(entry) = cache.get_mut(token) {
            // Check TTL
            if let Some(ttl) = self.config.ttl {
                if entry.created_at.elapsed() > ttl {
                    // Entry expired
                    cache.pop(token);
                    drop(cache);
                    self.handle_expiration(token);
                    if self.config.enable_metrics {
                        self.metrics.miss();
                    }
                    return None;
                }
            }

            entry.access_count += 1;
            let value = entry.value.clone();

            if self.config.enable_metrics {
                self.metrics.hit();
            }

            Some(value)
        } else {
            if self.config.enable_metrics {
                self.metrics.miss();
            }
            None
        }
    }

    /// Get token for an original value
    pub fn get_token(&self, original: &str) -> Option<String> {
        if let Some(token) = self.original_to_token.get(original) {
            // Verify token still exists in main cache
            if self.token_to_original.lock().contains(token.value()) {
                if self.config.enable_metrics {
                    self.metrics.hit();
                }
                Some(token.clone())
            } else {
                // Inconsistent state, clean up
                self.original_to_token.remove(original);
                if self.config.enable_metrics {
                    self.metrics.miss();
                }
                None
            }
        } else {
            if self.config.enable_metrics {
                self.metrics.miss();
            }
            None
        }
    }

    /// Check if a token exists
    pub fn contains_token(&self, token: &str) -> bool {
        let cache = self.token_to_original.lock();
        let exists = cache.contains(token);

        if exists && self.config.enable_metrics {
            self.metrics.hit();
        } else if self.config.enable_metrics {
            self.metrics.miss();
        }

        exists
    }

    /// Clear all entries
    pub fn clear(&self) {
        self.token_to_original.lock().clear();
        self.original_to_token.clear();
        self.token_index.clear();
        self.current_memory.store(0, Ordering::Relaxed);

        if self.config.enable_metrics {
            self.metrics.update_memory(0);
        }
    }

    /// Get current cache size
    pub fn len(&self) -> usize {
        self.token_to_original.lock().len()
    }

    /// Get current memory usage
    pub fn memory_usage(&self) -> usize {
        self.current_memory.load(Ordering::Relaxed)
    }

    /// Get cache metrics
    pub fn metrics(&self) -> CacheMetrics {
        (*self.metrics).clone()
    }

    /// Evict entries to fit new entry
    fn evict_to_fit(&self, required_size: usize) {
        let mut cache = self.token_to_original.lock();
        let mut freed_memory = 0;

        while freed_memory < required_size {
            if let Some((token, entry)) = cache.pop_lru() {
                freed_memory += entry.size_bytes;
                drop(cache); // Release lock for cleanup
                self.handle_eviction(&token, &entry);
                cache = self.token_to_original.lock(); // Re-acquire
            } else {
                break; // Cache is empty
            }
        }
    }

    /// Handle eviction of an entry
    fn handle_eviction(&self, token: &str, entry: &CacheEntry) {
        self.original_to_token.remove(&entry.value);
        self.token_index.remove(token);
        self.current_memory
            .fetch_sub(entry.size_bytes, Ordering::Relaxed);

        if self.config.enable_metrics {
            self.metrics.eviction();
            self.metrics
                .update_memory(self.current_memory.load(Ordering::Relaxed));
        }

        debug!("Evicted token {} (size: {} bytes)", token, entry.size_bytes);
    }

    /// Handle expiration of an entry
    fn handle_expiration(&self, token: &str) {
        if let Some((_, original)) = self.token_index.remove(token) {
            self.original_to_token.remove(&original);
        }

        debug!("Token {} expired", token);
    }

    /// Estimate memory size of an entry
    fn estimate_entry_size(token: &str, original: &str) -> usize {
        // Size of strings + overhead
        token.len() + original.len() + std::mem::size_of::<CacheEntry>() + 64 // HashMap overhead
    }

    /// Clean up expired entries (call periodically)
    pub fn cleanup_expired(&self) {
        if self.config.ttl.is_none() {
            return;
        }

        let ttl = self.config.ttl.unwrap();
        let mut cache = self.token_to_original.lock();
        let mut expired_tokens: Vec<String> = Vec::new();

        // Find expired entries
        for (token, entry) in cache.iter() {
            if entry.created_at.elapsed() > ttl {
                expired_tokens.push(token.clone());
            }
        }

        // Remove expired entries
        for token in expired_tokens {
            if let Some(entry) = cache.pop(&token) {
                drop(cache); // Release lock
                self.handle_eviction(&token, &entry);
                cache = self.token_to_original.lock(); // Re-acquire
            }
        }
    }

    /// Export cache statistics
    pub fn stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("entries".to_string(), self.len());
        stats.insert("memory_bytes".to_string(), self.memory_usage());
        stats.insert(
            "hits".to_string(),
            self.metrics.hits.load(Ordering::Relaxed),
        );
        stats.insert(
            "misses".to_string(),
            self.metrics.misses.load(Ordering::Relaxed),
        );
        stats.insert(
            "evictions".to_string(),
            self.metrics.evictions.load(Ordering::Relaxed),
        );
        stats.insert(
            "insertions".to_string(),
            self.metrics.insertions.load(Ordering::Relaxed),
        );
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let config = BoundedCacheConfig {
            max_entries: 3,
            max_memory_bytes: 0, // Unlimited for this test
            ttl: None,
            enable_metrics: true,
        };

        let cache = BoundedTokenCache::new(config);

        // Insert entries
        assert!(cache.insert("token1".to_string(), "original1".to_string()));
        assert!(cache.insert("token2".to_string(), "original2".to_string()));
        assert!(cache.insert("token3".to_string(), "original3".to_string()));

        // Verify lookups
        assert_eq!(cache.get_original("token1"), Some("original1".to_string()));
        assert_eq!(cache.get_token("original2"), Some("token2".to_string()));

        // Insert fourth entry should evict LRU
        assert!(cache.insert("token4".to_string(), "original4".to_string()));

        // token2 should be evicted (LRU)
        assert_eq!(cache.get_original("token2"), None);
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_memory_limit() {
        let config = BoundedCacheConfig {
            max_entries: 100,
            max_memory_bytes: 200, // Very small limit
            ttl: None,
            enable_metrics: true,
        };

        let cache = BoundedTokenCache::new(config);

        // Insert should succeed
        assert!(cache.insert("small".to_string(), "value".to_string()));

        // Large insert should fail
        let large_value = "x".repeat(1000);
        assert!(!cache.insert("large".to_string(), large_value));
    }

    #[test]
    fn test_ttl_expiration() {
        let config = BoundedCacheConfig {
            max_entries: 10,
            max_memory_bytes: 0,
            ttl: Some(Duration::from_millis(100)),
            enable_metrics: true,
        };

        let cache = BoundedTokenCache::new(config);

        cache.insert("token".to_string(), "original".to_string());
        assert!(cache.contains_token("token"));

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(150));

        // Should be expired
        assert_eq!(cache.get_original("token"), None);
    }
}
