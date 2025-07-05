use datacloak_core::cache::ObfuscationCache;
use rand::Rng;
use std::env;
use std::sync::Mutex;
use tempfile::tempdir;

// Mutex to ensure cache crypto tests don't interfere with each other
static TEST_MUTEX: Mutex<()> = Mutex::new(());

#[tokio::test]
async fn test_cache_encryption_round_trip() {
    let _guard = TEST_MUTEX.lock().unwrap();

    // Generate a random key for testing
    let mut rng = rand::rng();
    let key: Vec<u8> = (0..32).map(|_| rng.random()).collect();
    let key_hex = hex::encode(&key);

    // Save original key and set test key
    let original_key = env::var("DATACLOAK_CACHE_KEY").ok();
    env::set_var("DATACLOAK_CACHE_KEY", &key_hex);

    let dir = tempdir().unwrap();
    let cache_file = dir.path().join("test_cache.bin");

    // Create cache and store 100 random strings
    let cache = ObfuscationCache::with_storage(cache_file.clone()).unwrap();
    let mut test_data = Vec::new();

    for i in 0..100 {
        let token = format!("TOKEN-{}", i);
        let original = format!("test_data_{}_random", i);
        let pattern_type = match i % 5 {
            0 => "EMAIL",
            1 => "PHONE",
            2 => "SSN",
            3 => "NAME",
            _ => "ADDRESS",
        }
        .to_string();

        cache.store(token.clone(), original.clone(), pattern_type.clone());
        test_data.push((token, original, pattern_type));
    }

    // Save to disk
    cache.save().await.expect("Failed to save cache");

    // Create new cache instance and load
    let cache2 = ObfuscationCache::with_storage(cache_file).unwrap();
    cache2.load().await.expect("Failed to load cache");

    // Verify all data was correctly encrypted and decrypted
    for (token, original, _) in test_data {
        assert_eq!(
            cache2.get_original(&token),
            Some(original.clone()),
            "Failed to retrieve original for token: {}",
            token
        );
        assert_eq!(
            cache2.get_token(&original),
            Some(token.clone()),
            "Failed to retrieve token for original: {}",
            original
        );
    }

    // Verify cache stats
    let stats = cache2.stats();
    assert_eq!(stats.total_entries, 100);

    // Restore original environment
    if let Some(key) = original_key {
        env::set_var("DATACLOAK_CACHE_KEY", key);
    } else {
        env::remove_var("DATACLOAK_CACHE_KEY");
    }
}

#[tokio::test]
async fn test_cache_encryption_with_invalid_key() {
    let _guard = TEST_MUTEX.lock().unwrap();

    // Test with missing environment variable
    let original_key = env::var("DATACLOAK_CACHE_KEY").ok();
    env::remove_var("DATACLOAK_CACHE_KEY");

    let dir = tempdir().unwrap();
    let cache_file = dir.path().join("test_cache.bin");

    let result = ObfuscationCache::with_storage(cache_file);

    // Restore original key if it existed
    if let Some(key) = original_key {
        env::set_var("DATACLOAK_CACHE_KEY", key);
    }

    assert!(result.is_err());
    if let Err(e) = result {
        assert!(e.to_string().contains("DATACLOAK_CACHE_KEY"));
    }
}

#[tokio::test]
async fn test_cache_encryption_with_wrong_key_size() {
    let _guard = TEST_MUTEX.lock().unwrap();

    // Test with invalid key size (16 bytes instead of 32)
    let original_key = env::var("DATACLOAK_CACHE_KEY").ok();
    env::set_var("DATACLOAK_CACHE_KEY", "0123456789abcdef"); // Wrong size: 16 chars instead of 64

    let dir = tempdir().unwrap();
    let cache_file = dir.path().join("test_cache.bin");

    let result = ObfuscationCache::with_storage(cache_file);

    // Restore original key if it existed
    if let Some(key) = original_key {
        env::set_var("DATACLOAK_CACHE_KEY", key);
    }

    assert!(result.is_err());
    if let Err(e) = result {
        assert!(e.to_string().contains("Invalid DATACLOAK_CACHE_KEY size"));
    }
}

#[tokio::test]
async fn test_cache_encryption_with_invalid_hex() {
    let _guard = TEST_MUTEX.lock().unwrap();

    // Test with invalid hex characters
    let original_key = env::var("DATACLOAK_CACHE_KEY").ok();
    env::set_var(
        "DATACLOAK_CACHE_KEY",
        "xyz1234567890abcdef1234567890abcdef1234567890abcdef1234567890ab",
    );

    let dir = tempdir().unwrap();
    let cache_file = dir.path().join("test_cache.bin");

    let result = ObfuscationCache::with_storage(cache_file);

    // Restore original key if it existed
    if let Some(key) = original_key {
        env::set_var("DATACLOAK_CACHE_KEY", key);
    }

    assert!(result.is_err());
    if let Err(e) = result {
        assert!(e.to_string().contains("Invalid DATACLOAK_CACHE_KEY format"));
    }
}

#[tokio::test]
async fn test_cache_decryption_with_wrong_key() {
    let _guard = TEST_MUTEX.lock().unwrap();

    // Create and save cache with one key
    let key1 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    env::set_var("DATACLOAK_CACHE_KEY", key1);

    let dir = tempdir().unwrap();
    let cache_file = dir.path().join("test_cache.bin");

    let cache = ObfuscationCache::with_storage(cache_file.clone()).unwrap();
    cache.store(
        "TOKEN-1".to_string(),
        "test@example.com".to_string(),
        "EMAIL".to_string(),
    );
    cache.save().await.unwrap();

    // Try to load with different key
    let key2 = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
    env::set_var("DATACLOAK_CACHE_KEY", key2);

    let cache2 = ObfuscationCache::with_storage(cache_file).unwrap();
    let result = cache2.load().await;

    // Should fail due to decryption error
    assert!(result.is_err());
}

#[tokio::test]
async fn test_large_cache_encryption_performance() {
    let _guard = TEST_MUTEX.lock().unwrap();

    use std::time::Instant;

    let mut rng = rand::rng();
    let key: Vec<u8> = (0..32).map(|_| rng.random()).collect();
    let original_key = env::var("DATACLOAK_CACHE_KEY").ok();
    env::set_var("DATACLOAK_CACHE_KEY", hex::encode(&key));

    let dir = tempdir().unwrap();
    let cache_file = dir.path().join("large_cache.bin");

    let cache = ObfuscationCache::with_storage(cache_file.clone()).unwrap();

    // Store 10,000 entries
    let start = Instant::now();
    for i in 0..10_000 {
        let token = format!("TOKEN-{}", i);
        let original = format!("user{}@example.com", i);
        cache.store(token, original, "EMAIL".to_string());
    }
    let store_duration = start.elapsed();

    // Save encrypted cache
    let start = Instant::now();
    cache.save().await.unwrap();
    let save_duration = start.elapsed();

    // Load encrypted cache
    let cache2 = ObfuscationCache::with_storage(cache_file).unwrap();
    let start = Instant::now();
    cache2.load().await.unwrap();
    let load_duration = start.elapsed();

    // Performance assertions (generous limits for CI environments)
    assert!(
        store_duration.as_secs() < 5,
        "Storing 10k entries took too long: {:?}",
        store_duration
    );
    assert!(
        save_duration.as_secs() < 10,
        "Saving encrypted cache took too long: {:?}",
        save_duration
    );
    assert!(
        load_duration.as_secs() < 10,
        "Loading encrypted cache took too long: {:?}",
        load_duration
    );

    // Verify data integrity
    assert_eq!(cache2.stats().total_entries, 10_000);
    assert_eq!(
        cache2.get_original("TOKEN-5000"),
        Some("user5000@example.com".to_string())
    );

    // Restore original environment
    if let Some(key) = original_key {
        env::set_var("DATACLOAK_CACHE_KEY", key);
    } else {
        env::remove_var("DATACLOAK_CACHE_KEY");
    }
}
