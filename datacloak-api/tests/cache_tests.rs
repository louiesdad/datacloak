use datacloak_api::services::{CacheLayer, LRUCache, RedisCache, CacheStats};
use uuid::Uuid;
use serde_json::Value;

const MB: usize = 1024 * 1024;
const KB: usize = 1024;

#[tokio::test]
async fn test_profile_result_caching() {
    let cache = LRUCache::new(100 * MB);
    let file_id = Uuid::new_v4();
    
    let profile_result = serde_json::json!({
        "candidates": [
            {"name": "col1", "score": 0.9},
            {"name": "col2", "score": 0.8}
        ],
        "total_columns": 10
    });
    
    // First call - miss
    assert!(cache.get(&file_id.to_string()).await.is_none());
    cache.put(&file_id.to_string(), profile_result.clone(), 3600).await.unwrap();
    
    let stats = cache.stats().await;
    assert_eq!(stats.misses, 1);
    assert_eq!(stats.hits, 0);
    
    // Second call - hit
    let cached = cache.get(&file_id.to_string()).await.unwrap();
    assert_eq!(cached, profile_result);
    
    let stats = cache.stats().await;
    assert_eq!(stats.hits, 1);
}

#[tokio::test]
async fn test_cache_invalidation() {
    let cache = LRUCache::new(100 * MB);
    let file_id = Uuid::new_v4();
    
    let profile_result = serde_json::json!({"test": "data"});
    
    // Cache result
    cache.put(&file_id.to_string(), profile_result, 300).await.unwrap();
    assert!(cache.get(&file_id.to_string()).await.is_some());
    
    // Manual invalidation
    cache.invalidate(&file_id.to_string()).await.unwrap();
    assert!(cache.get(&file_id.to_string()).await.is_none());
}

#[tokio::test]
async fn test_cache_memory_limits() {
    let cache = LRUCache::new(1 * MB); // Small cache
    
    // Fill cache with large objects
    for i in 0..20 {
        let key = format!("large_key_{}", i);
        let large_result = create_large_value(100 * KB);
        cache.put(&key, large_result, 3600).await.unwrap();
    }
    
    // Verify memory limit respected
    let stats = cache.stats().await;
    assert!(stats.memory_usage_bytes <= 1 * MB);
    assert!(stats.total_entries < 20); // Some evicted due to memory pressure
}

#[tokio::test]
async fn test_cache_ttl_expiry() {
    let cache = LRUCache::new(10 * MB);
    let key = "test_key";
    let value = serde_json::json!({"expires": "soon"});
    
    // Cache with 1 second TTL
    cache.put(key, value.clone(), 1).await.unwrap();
    assert_eq!(cache.get(key).await.unwrap(), value);
    
    // Wait for expiry
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    assert!(cache.get(key).await.is_none());
}

#[tokio::test]
async fn test_lru_eviction_policy() {
    let cache = LRUCache::new(512 * KB); // Small cache to force eviction
    
    // Add items to fill cache
    for i in 0..10 {
        let key = format!("key_{}", i);
        let value = create_large_value(100 * KB);
        cache.put(&key, value, 3600).await.unwrap();
    }
    
    // Access some items to make them recently used
    let _ = cache.get("key_0").await;
    let _ = cache.get("key_1").await;
    let _ = cache.get("key_2").await;
    
    // Add new items to trigger eviction
    for i in 10..15 {
        let key = format!("key_{}", i);
        let value = create_large_value(100 * KB);
        cache.put(&key, value, 3600).await.unwrap();
    }
    
    // Recently accessed items should still be there
    assert!(cache.get("key_0").await.is_some());
    assert!(cache.get("key_1").await.is_some());
    
    // Older, unaccessed items should be evicted
    assert!(cache.get("key_3").await.is_none());
    assert!(cache.get("key_4").await.is_none());
}

#[tokio::test]
async fn test_cache_compression() {
    let cache = LRUCache::new(10 * MB).with_compression(true);
    let key = "compressible_data";
    
    // Create highly compressible data
    let repetitive_data = serde_json::json!({
        "data": "A".repeat(10000), // Highly compressible
        "metadata": {
            "repeated": vec!["same"; 1000]
        }
    });
    
    cache.put(key, repetitive_data.clone(), 3600).await.unwrap();
    let retrieved = cache.get(key).await.unwrap();
    assert_eq!(retrieved, repetitive_data);
    
    let stats = cache.stats().await;
    // Memory usage should be significantly less than uncompressed size
    assert!(stats.compression_ratio > 2.0);
}

#[tokio::test]
async fn test_redis_cache_integration() {
    // Skip if Redis not available
    if std::env::var("REDIS_URL").is_err() {
        return;
    }
    
    let redis_url = std::env::var("REDIS_URL").unwrap_or("redis://localhost:6379".to_string());
    let cache = RedisCache::new(&redis_url).await.unwrap();
    
    let key = "redis_test_key";
    let value = serde_json::json!({"redis": "test", "number": 42});
    
    // Test basic operations
    cache.put(key, value.clone(), 300).await.unwrap();
    let retrieved = cache.get(key).await.unwrap();
    assert_eq!(retrieved, value);
    
    // Test expiry
    cache.put(key, value, 1).await.unwrap();
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    assert!(cache.get(key).await.is_none());
}

#[tokio::test]
async fn test_cache_warming() {
    let cache = LRUCache::new(10 * MB);
    
    // Simulate cache warming with common queries
    let warm_data = vec![
        ("common_file_1", serde_json::json!({"type": "common_profile"})),
        ("common_file_2", serde_json::json!({"type": "common_profile"})),
        ("common_file_3", serde_json::json!({"type": "common_profile"})),
    ];
    
    // Warm the cache
    cache.warm(warm_data.clone()).await.unwrap();
    
    // All items should be cached
    for (key, expected_value) in warm_data {
        let cached_value = cache.get(key).await.unwrap();
        assert_eq!(cached_value, expected_value);
    }
    
    let stats = cache.stats().await;
    assert_eq!(stats.total_entries, 3);
    assert_eq!(stats.hits, 3); // From the get() calls above
}

#[tokio::test]
async fn test_concurrent_cache_access() {
    let cache = std::sync::Arc::new(LRUCache::new(10 * MB));
    let mut handles = vec![];
    
    // Spawn multiple tasks accessing cache concurrently
    for i in 0..10 {
        let cache_clone = cache.clone();
        let handle = tokio::spawn(async move {
            let key = format!("concurrent_key_{}", i);
            let value = serde_json::json!({"thread": i, "data": "test"});
            
            // Write
            cache_clone.put(&key, value.clone(), 3600).await.unwrap();
            
            // Read multiple times
            for _ in 0..5 {
                let retrieved = cache_clone.get(&key).await.unwrap();
                assert_eq!(retrieved, value);
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            }
        });
        handles.push(handle);
    }
    
    // Wait for all tasks
    for handle in handles {
        handle.await.unwrap();
    }
    
    let stats = cache.stats().await;
    assert_eq!(stats.total_entries, 10);
    assert!(stats.hits >= 50); // 10 tasks * 5 reads each
}

// Helper function to create large test values
fn create_large_value(size_bytes: usize) -> Value {
    let data = "x".repeat(size_bytes / 2); // Approximate size
    serde_json::json!({
        "large_data": data,
        "metadata": {
            "size": size_bytes,
            "created": chrono::Utc::now().to_rfc3339()
        }
    })
}