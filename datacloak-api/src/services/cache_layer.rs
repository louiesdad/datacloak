use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc, Duration as ChronoDuration};
use anyhow::Result;
use async_trait::async_trait;

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub total_entries: usize,
    pub memory_usage_bytes: usize,
    pub hit_rate: f64,
    pub compression_ratio: f64,
    pub evictions: u64,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    value: Value,
    created_at: DateTime<Utc>,
    expires_at: DateTime<Utc>,
    last_accessed: DateTime<Utc>,
    access_count: u64,
    compressed_size: Option<usize>,
}

impl CacheEntry {
    fn new(value: Value, ttl_seconds: i64) -> Self {
        let now = Utc::now();
        Self {
            value,
            created_at: now,
            expires_at: now + ChronoDuration::seconds(ttl_seconds),
            last_accessed: now,
            access_count: 0,
            compressed_size: None,
        }
    }
    
    fn is_expired(&self) -> bool {
        Utc::now() > self.expires_at
    }
    
    fn touch(&mut self) {
        self.last_accessed = Utc::now();
        self.access_count += 1;
    }
    
    fn estimated_size(&self) -> usize {
        self.compressed_size.unwrap_or_else(|| {
            // Rough estimate of JSON size in memory
            serde_json::to_vec(&self.value).map(|v| v.len()).unwrap_or(0)
        })
    }
}

#[async_trait]
pub trait CacheLayer: Send + Sync {
    async fn get(&self, key: &str) -> Option<Value>;
    async fn put(&self, key: &str, value: Value, ttl_seconds: i64) -> Result<()>;
    async fn invalidate(&self, key: &str) -> Result<()>;
    async fn clear(&self) -> Result<()>;
    async fn stats(&self) -> CacheStats;
}

pub struct LRUCache {
    data: Arc<RwLock<HashMap<String, CacheEntry>>>,
    stats: Arc<RwLock<CacheStats>>,
    max_memory_bytes: usize,
    compression_enabled: bool,
}

impl LRUCache {
    pub fn new(max_memory_bytes: usize) -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(CacheStats {
                hits: 0,
                misses: 0,
                total_entries: 0,
                memory_usage_bytes: 0,
                hit_rate: 0.0,
                compression_ratio: 1.0,
                evictions: 0,
            })),
            max_memory_bytes,
            compression_enabled: false,
        }
    }
    
    pub fn with_compression(mut self, enabled: bool) -> Self {
        self.compression_enabled = enabled;
        self
    }
    
    pub async fn warm(&self, entries: Vec<(&str, Value)>) -> Result<()> {
        for (key, value) in entries {
            self.put(key, value, 3600).await?; // 1 hour default TTL for warm data
        }
        Ok(())
    }
    
    async fn evict_if_needed(&self) {
        let current_memory = {
            let data = self.data.read().await;
            data.values().map(|entry| entry.estimated_size()).sum::<usize>()
        };
        
        if current_memory > self.max_memory_bytes {
            let mut data = self.data.write().await;
            let mut stats = self.stats.write().await;
            
            // Sort by LRU (least recently used first)
            let mut entries: Vec<_> = data.iter().map(|(k, v)| (k.clone(), v.last_accessed)).collect();
            entries.sort_by(|a, b| a.1.cmp(&b.1));
            
            // Remove oldest entries until under memory limit
            let mut current_size = current_memory;
            for (key, _) in entries {
                if current_size <= self.max_memory_bytes {
                    break;
                }
                
                if let Some(entry) = data.remove(&key) {
                    current_size -= entry.estimated_size();
                    stats.evictions += 1;
                }
            }
            
            stats.total_entries = data.len();
            stats.memory_usage_bytes = current_size;
        }
    }
    
    async fn cleanup_expired(&self) {
        let mut data = self.data.write().await;
        let expired_keys: Vec<String> = data.iter()
            .filter(|(_, entry)| entry.is_expired())
            .map(|(key, _)| key.clone())
            .collect();
        
        for key in expired_keys {
            data.remove(&key);
        }
        
        let mut stats = self.stats.write().await;
        stats.total_entries = data.len();
        stats.memory_usage_bytes = data.values().map(|e| e.estimated_size()).sum();
    }
    
    fn compress_value(&self, value: &Value) -> Result<(Value, Option<usize>)> {
        if !self.compression_enabled {
            return Ok((value.clone(), None));
        }
        
        // Simple compression simulation - in real implementation would use zstd/gzip
        let serialized = serde_json::to_vec(value)?;
        let original_size = serialized.len();
        
        // Simulate compression (mock 50% compression ratio for demo)
        let compressed_size = original_size / 2;
        
        Ok((value.clone(), Some(compressed_size)))
    }
}

#[async_trait]
impl CacheLayer for LRUCache {
    async fn get(&self, key: &str) -> Option<Value> {
        // Cleanup expired entries periodically
        if rand::random::<f32>() < 0.01 { // 1% chance
            self.cleanup_expired().await;
        }
        
        let mut data = self.data.write().await;
        let mut stats = self.stats.write().await;
        
        if let Some(entry) = data.get_mut(key) {
            if entry.is_expired() {
                data.remove(key);
                stats.misses += 1;
                None
            } else {
                entry.touch();
                stats.hits += 1;
                stats.hit_rate = stats.hits as f64 / (stats.hits + stats.misses) as f64;
                Some(entry.value.clone())
            }
        } else {
            stats.misses += 1;
            stats.hit_rate = stats.hits as f64 / (stats.hits + stats.misses) as f64;
            None
        }
    }
    
    async fn put(&self, key: &str, value: Value, ttl_seconds: i64) -> Result<()> {
        let (compressed_value, compressed_size) = self.compress_value(&value)?;
        
        {
            let mut data = self.data.write().await;
            let mut entry = CacheEntry::new(compressed_value, ttl_seconds);
            entry.compressed_size = compressed_size;
            data.insert(key.to_string(), entry);
            
            let mut stats = self.stats.write().await;
            stats.total_entries = data.len();
            stats.memory_usage_bytes = data.values().map(|e| e.estimated_size()).sum();
            
            if self.compression_enabled && compressed_size.is_some() {
                let original_size = serde_json::to_vec(&value)?.len();
                stats.compression_ratio = original_size as f64 / compressed_size.unwrap() as f64;
            }
        }
        
        // Check if eviction is needed
        self.evict_if_needed().await;
        
        Ok(())
    }
    
    async fn invalidate(&self, key: &str) -> Result<()> {
        let mut data = self.data.write().await;
        data.remove(key);
        
        let mut stats = self.stats.write().await;
        stats.total_entries = data.len();
        stats.memory_usage_bytes = data.values().map(|e| e.estimated_size()).sum();
        
        Ok(())
    }
    
    async fn clear(&self) -> Result<()> {
        let mut data = self.data.write().await;
        data.clear();
        
        let mut stats = self.stats.write().await;
        stats.total_entries = 0;
        stats.memory_usage_bytes = 0;
        
        Ok(())
    }
    
    async fn stats(&self) -> CacheStats {
        self.stats.read().await.clone()
    }
}

// Redis-based cache implementation
pub struct RedisCache {
    client: redis::Client,
    stats: Arc<RwLock<CacheStats>>,
}

impl RedisCache {
    pub async fn new(redis_url: &str) -> Result<Self> {
        let client = redis::Client::open(redis_url)?;
        
        // Test connection
        let mut conn = client.get_async_connection().await?;
        redis::cmd("PING").query_async(&mut conn).await?;
        
        Ok(Self {
            client,
            stats: Arc::new(RwLock::new(CacheStats {
                hits: 0,
                misses: 0,
                total_entries: 0,
                memory_usage_bytes: 0,
                hit_rate: 0.0,
                compression_ratio: 1.0,
                evictions: 0,
            })),
        })
    }
}

#[async_trait]
impl CacheLayer for RedisCache {
    async fn get(&self, key: &str) -> Option<Value> {
        let mut conn = match self.client.get_async_connection().await {
            Ok(conn) => conn,
            Err(_) => {
                let mut stats = self.stats.write().await;
                stats.misses += 1;
                return None;
            }
        };
        
        let result: Option<String> = redis::cmd("GET")
            .arg(key)
            .query_async(&mut conn)
            .await
            .ok()
            .flatten();
        
        let mut stats = self.stats.write().await;
        if let Some(json_str) = result {
            if let Ok(value) = serde_json::from_str(&json_str) {
                stats.hits += 1;
                stats.hit_rate = stats.hits as f64 / (stats.hits + stats.misses) as f64;
                Some(value)
            } else {
                stats.misses += 1;
                None
            }
        } else {
            stats.misses += 1;
            stats.hit_rate = stats.hits as f64 / (stats.hits + stats.misses) as f64;
            None
        }
    }
    
    async fn put(&self, key: &str, value: Value, ttl_seconds: i64) -> Result<()> {
        let mut conn = self.client.get_async_connection().await?;
        let json_str = serde_json::to_string(&value)?;
        
        redis::cmd("SETEX")
            .arg(key)
            .arg(ttl_seconds)
            .arg(json_str)
            .query_async(&mut conn)
            .await?;
        
        Ok(())
    }
    
    async fn invalidate(&self, key: &str) -> Result<()> {
        let mut conn = self.client.get_async_connection().await?;
        redis::cmd("DEL").arg(key).query_async(&mut conn).await?;
        Ok(())
    }
    
    async fn clear(&self) -> Result<()> {
        let mut conn = self.client.get_async_connection().await?;
        redis::cmd("FLUSHDB").query_async(&mut conn).await?;
        Ok(())
    }
    
    async fn stats(&self) -> CacheStats {
        self.stats.read().await.clone()
    }
}

// Tiered cache: LRU + Redis
pub struct TieredCache {
    l1_cache: LRUCache,
    l2_cache: Option<RedisCache>,
}

impl TieredCache {
    pub fn new(l1_size: usize) -> Self {
        Self {
            l1_cache: LRUCache::new(l1_size),
            l2_cache: None,
        }
    }
    
    pub fn with_redis(mut self, redis_cache: RedisCache) -> Self {
        self.l2_cache = Some(redis_cache);
        self
    }
}

#[async_trait]
impl CacheLayer for TieredCache {
    async fn get(&self, key: &str) -> Option<Value> {
        // Try L1 cache first
        if let Some(value) = self.l1_cache.get(key).await {
            return Some(value);
        }
        
        // Try L2 cache (Redis)
        if let Some(ref l2) = self.l2_cache {
            if let Some(value) = l2.get(key).await {
                // Promote to L1 cache
                let _ = self.l1_cache.put(key, value.clone(), 3600).await;
                return Some(value);
            }
        }
        
        None
    }
    
    async fn put(&self, key: &str, value: Value, ttl_seconds: i64) -> Result<()> {
        // Store in both caches
        self.l1_cache.put(key, value.clone(), ttl_seconds).await?;
        
        if let Some(ref l2) = self.l2_cache {
            l2.put(key, value, ttl_seconds).await?;
        }
        
        Ok(())
    }
    
    async fn invalidate(&self, key: &str) -> Result<()> {
        self.l1_cache.invalidate(key).await?;
        
        if let Some(ref l2) = self.l2_cache {
            l2.invalidate(key).await?;
        }
        
        Ok(())
    }
    
    async fn clear(&self) -> Result<()> {
        self.l1_cache.clear().await?;
        
        if let Some(ref l2) = self.l2_cache {
            l2.clear().await?;
        }
        
        Ok(())
    }
    
    async fn stats(&self) -> CacheStats {
        // Combine stats from both caches
        let l1_stats = self.l1_cache.stats().await;
        
        if let Some(ref l2) = self.l2_cache {
            let l2_stats = l2.stats().await;
            CacheStats {
                hits: l1_stats.hits + l2_stats.hits,
                misses: l1_stats.misses + l2_stats.misses,
                total_entries: l1_stats.total_entries + l2_stats.total_entries,
                memory_usage_bytes: l1_stats.memory_usage_bytes,
                hit_rate: (l1_stats.hits + l2_stats.hits) as f64 / 
                         (l1_stats.hits + l1_stats.misses + l2_stats.hits + l2_stats.misses) as f64,
                compression_ratio: l1_stats.compression_ratio,
                evictions: l1_stats.evictions,
            }
        } else {
            l1_stats
        }
    }
}