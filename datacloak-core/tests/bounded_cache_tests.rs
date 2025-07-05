use datacloak_core::{BoundedCacheConfig, BoundedTokenCache};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

#[test]
fn test_basic_cache_operations() {
    let config = BoundedCacheConfig {
        max_entries: 10,
        max_memory_bytes: 0, // Unlimited for this test
        ttl: None,
        enable_metrics: true,
    };

    let cache = BoundedTokenCache::new(config);

    // Test insertion and retrieval
    assert!(cache.insert("TOKEN_001".to_string(), "test@example.com".to_string()));
    assert_eq!(
        cache.get_original("TOKEN_001"),
        Some("test@example.com".to_string())
    );
    assert_eq!(
        cache.get_token("test@example.com"),
        Some("TOKEN_001".to_string())
    );

    // Test non-existent keys
    assert_eq!(cache.get_original("TOKEN_999"), None);
    assert_eq!(cache.get_token("nonexistent@example.com"), None);

    // Test contains
    assert!(cache.contains_token("TOKEN_001"));
    assert!(!cache.contains_token("TOKEN_999"));
}

#[test]
fn test_lru_eviction() {
    let config = BoundedCacheConfig {
        max_entries: 3,
        max_memory_bytes: 0,
        ttl: None,
        enable_metrics: true,
    };

    let cache = BoundedTokenCache::new(config);

    // Fill cache to capacity
    cache.insert("TOKEN_001".to_string(), "value1".to_string());
    cache.insert("TOKEN_002".to_string(), "value2".to_string());
    cache.insert("TOKEN_003".to_string(), "value3".to_string());

    assert_eq!(cache.len(), 3);

    // Access TOKEN_001 to make it recently used
    assert_eq!(cache.get_original("TOKEN_001"), Some("value1".to_string()));

    // Insert new item should evict TOKEN_002 (LRU)
    cache.insert("TOKEN_004".to_string(), "value4".to_string());

    assert_eq!(cache.len(), 3);
    assert!(cache.contains_token("TOKEN_001")); // Still there (recently used)
    assert!(!cache.contains_token("TOKEN_002")); // Evicted (LRU)
    assert!(cache.contains_token("TOKEN_003"));
    assert!(cache.contains_token("TOKEN_004"));

    // Verify metrics
    let metrics = cache.metrics();
    assert_eq!(
        metrics.evictions.load(std::sync::atomic::Ordering::Relaxed),
        1
    );
}

#[test]
fn test_memory_limit() {
    let config = BoundedCacheConfig {
        max_entries: 100,
        max_memory_bytes: 500, // Very small limit
        ttl: None,
        enable_metrics: true,
    };

    let cache = BoundedTokenCache::new(config);

    // Small entries should succeed
    assert!(cache.insert("T1".to_string(), "small".to_string()));
    assert!(cache.insert("T2".to_string(), "value".to_string()));

    // Large entry should fail due to memory limit
    let large_value = "x".repeat(1000);
    assert!(!cache.insert("T3".to_string(), large_value.clone()));

    // But small entry should still work
    assert!(cache.insert("T4".to_string(), "tiny".to_string()));

    // Check memory usage
    assert!(cache.memory_usage() < 500);
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

    // Insert entries
    cache.insert("TOKEN_001".to_string(), "value1".to_string());
    cache.insert("TOKEN_002".to_string(), "value2".to_string());

    // Verify they exist
    assert!(cache.contains_token("TOKEN_001"));
    assert!(cache.contains_token("TOKEN_002"));

    // Wait for TTL to expire
    thread::sleep(Duration::from_millis(150));

    // Should be expired now
    assert_eq!(cache.get_original("TOKEN_001"), None);
    assert_eq!(cache.get_original("TOKEN_002"), None);

    // Reverse lookups should also fail
    assert_eq!(cache.get_token("value1"), None);
}

#[test]
fn test_cleanup_expired() {
    let config = BoundedCacheConfig {
        max_entries: 10,
        max_memory_bytes: 0,
        ttl: Some(Duration::from_millis(50)),
        enable_metrics: true,
    };

    let cache = BoundedTokenCache::new(config);

    // Insert entries at different times
    cache.insert("TOKEN_001".to_string(), "value1".to_string());
    thread::sleep(Duration::from_millis(30));
    cache.insert("TOKEN_002".to_string(), "value2".to_string());

    // Wait for first to expire but not second
    thread::sleep(Duration::from_millis(30));

    // Run cleanup
    cache.cleanup_expired();

    // TOKEN_001 should be gone, TOKEN_002 should remain
    assert!(!cache.contains_token("TOKEN_001"));
    assert!(cache.contains_token("TOKEN_002"));
}

#[test]
fn test_concurrent_access() {
    let config = BoundedCacheConfig {
        max_entries: 1000,
        max_memory_bytes: 0,
        ttl: None,
        enable_metrics: true,
    };

    let cache = Arc::new(BoundedTokenCache::new(config));
    let mut handles = vec![];

    // Spawn multiple threads for concurrent writes
    for i in 0..10 {
        let cache_clone = cache.clone();
        let handle = thread::spawn(move || {
            for j in 0..100 {
                let token = format!("TOKEN_{:03}_{:03}", i, j);
                let value = format!("value_{}_{}", i, j);
                cache_clone.insert(token, value);
            }
        });
        handles.push(handle);
    }

    // Spawn threads for concurrent reads
    for i in 0..5 {
        let cache_clone = cache.clone();
        let handle = thread::spawn(move || {
            for j in 0..50 {
                let token = format!("TOKEN_{:03}_{:03}", i, j);
                cache_clone.get_original(&token);
            }
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    // Verify cache state
    assert!(cache.len() <= 1000);
    let metrics = cache.metrics();
    assert!(
        metrics
            .insertions
            .load(std::sync::atomic::Ordering::Relaxed)
            > 0
    );
}

#[test]
fn test_metrics_tracking() {
    let config = BoundedCacheConfig {
        max_entries: 5,
        max_memory_bytes: 0,
        ttl: None,
        enable_metrics: true,
    };

    let cache = BoundedTokenCache::new(config);

    // Generate some activity
    cache.insert("TOKEN_001".to_string(), "value1".to_string());
    cache.insert("TOKEN_002".to_string(), "value2".to_string());

    // Hits
    cache.get_original("TOKEN_001");
    cache.get_original("TOKEN_001");
    cache.get_token("value2");

    // Misses
    cache.get_original("TOKEN_999");
    cache.get_token("nonexistent");

    // Check metrics
    let metrics = cache.metrics();
    assert_eq!(metrics.hits.load(std::sync::atomic::Ordering::Relaxed), 3);
    assert_eq!(metrics.misses.load(std::sync::atomic::Ordering::Relaxed), 2);
    assert_eq!(
        metrics
            .insertions
            .load(std::sync::atomic::Ordering::Relaxed),
        2
    );

    // Test hit rate
    assert_eq!(metrics.hit_rate(), 0.6); // 3 hits / 5 total
}

#[test]
fn test_clear_operation() {
    let config = BoundedCacheConfig::default();
    let cache = BoundedTokenCache::new(config);

    // Add some data
    for i in 0..10 {
        cache.insert(format!("TOKEN_{:03}", i), format!("value_{}", i));
    }

    assert_eq!(cache.len(), 10);
    assert!(cache.memory_usage() > 0);

    // Clear cache
    cache.clear();

    // Verify everything is gone
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.memory_usage(), 0);
    assert!(!cache.contains_token("TOKEN_001"));
    assert_eq!(cache.get_token("value_1"), None);
}

#[test]
fn test_stats_export() {
    let config = BoundedCacheConfig {
        max_entries: 10,
        max_memory_bytes: 0,
        ttl: None,
        enable_metrics: true,
    };

    let cache = BoundedTokenCache::new(config);

    // Generate activity
    cache.insert("TOKEN_001".to_string(), "value1".to_string());
    cache.insert("TOKEN_002".to_string(), "value2".to_string());
    cache.get_original("TOKEN_001");
    cache.get_original("TOKEN_999");

    // Export stats
    let stats = cache.stats();

    assert_eq!(stats["entries"], 2);
    assert!(stats["memory_bytes"] > 0);
    assert_eq!(stats["hits"], 1);
    assert_eq!(stats["misses"], 1);
    assert_eq!(stats["insertions"], 2);
    assert_eq!(stats["evictions"], 0);
}

#[test]
fn test_token_value_consistency() {
    let config = BoundedCacheConfig::default();
    let cache = BoundedTokenCache::new(config);

    // Insert mapping
    cache.insert("TOKEN_001".to_string(), "original@example.com".to_string());

    // Both directions should work
    assert_eq!(
        cache.get_original("TOKEN_001"),
        Some("original@example.com".to_string())
    );
    assert_eq!(
        cache.get_token("original@example.com"),
        Some("TOKEN_001".to_string())
    );

    // After eviction, both should be gone
    let config2 = BoundedCacheConfig {
        max_entries: 1,
        max_memory_bytes: 0,
        ttl: None,
        enable_metrics: false,
    };
    let small_cache = BoundedTokenCache::new(config2);

    small_cache.insert("TOKEN_001".to_string(), "value1".to_string());
    small_cache.insert("TOKEN_002".to_string(), "value2".to_string()); // Evicts TOKEN_001

    // Both lookups should fail for evicted entry
    assert_eq!(small_cache.get_original("TOKEN_001"), None);
    assert_eq!(small_cache.get_token("value1"), None);
}

#[test]
fn test_memory_eviction_to_fit() {
    let config = BoundedCacheConfig {
        max_entries: 100,
        max_memory_bytes: 1000, // 1KB limit
        ttl: None,
        enable_metrics: true,
    };

    let cache = BoundedTokenCache::new(config);

    // Fill with small entries
    for i in 0..20 {
        cache.insert(format!("T{}", i), format!("val{}", i));
    }

    let initial_entries = cache.len();

    // Try to insert a large entry that requires eviction
    let large_value = "x".repeat(200);
    assert!(cache.insert("LARGE".to_string(), large_value));

    // Should have evicted some entries to make room
    assert!(cache.len() < initial_entries);
    assert!(cache.contains_token("LARGE"));
    assert!(cache.memory_usage() <= 1000);
}
