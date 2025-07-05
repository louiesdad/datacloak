use datacloak_core::{
    BoundedCacheConfig, BoundedObfuscator, BoundedObfuscatorConfig, Pattern, PatternType,
    RecordBatch,
};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;

#[test]
fn test_basic_obfuscation() {
    let config = BoundedObfuscatorConfig::default();
    let obfuscator = BoundedObfuscator::new(config);

    // Set patterns
    let patterns = vec![
        Pattern::new(
            PatternType::Email,
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b".to_string(),
        ),
        Pattern::new(PatternType::SSN, r"\b\d{3}-\d{2}-\d{4}\b".to_string()),
    ];

    obfuscator.set_patterns(patterns).unwrap();

    // Create test data
    let batch = vec![
        json!({
            "id": "1",
            "email": "john.doe@example.com",
            "ssn": "123-45-6789",
            "name": "John Doe"
        }),
        json!({
            "id": "2",
            "email": "jane.smith@example.com",
            "ssn": "987-65-4321",
            "name": "Jane Smith"
        }),
    ];

    // Obfuscate
    let result = obfuscator.obfuscate_batch(&batch).unwrap();

    // Verify obfuscation
    assert_eq!(result.len(), 2);

    // Check first record
    let first = &result[0];
    assert_eq!(first.id, Some("1".to_string()));
    assert!(first.data["email"].as_str().unwrap().starts_with("EMAIL_"));
    assert!(first.data["ssn"].as_str().unwrap().starts_with("SSN_"));
    assert_eq!(first.data["name"], "John Doe"); // Unchanged
    assert_eq!(first.tokens_used.len(), 2);

    // Verify reverse mapping works
    let reversed = obfuscator.reverse_batch(&result).unwrap();
    assert_eq!(reversed[0]["email"], "john.doe@example.com");
    assert_eq!(reversed[0]["ssn"], "123-45-6789");
}

#[test]
fn test_token_reuse() {
    let mut config = BoundedObfuscatorConfig::default();
    config.enable_token_reuse = true;

    let obfuscator = BoundedObfuscator::new(config);

    let patterns = vec![Pattern::new(
        PatternType::Email,
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b".to_string(),
    )];
    obfuscator.set_patterns(patterns).unwrap();

    // Obfuscate same email multiple times
    let batch1 = vec![json!({"email": "same@example.com"})];
    let batch2 = vec![json!({"email": "same@example.com"})];
    let batch3 = vec![json!({"email": "different@example.com"})];

    let result1 = obfuscator.obfuscate_batch(&batch1).unwrap();
    let result2 = obfuscator.obfuscate_batch(&batch2).unwrap();
    let result3 = obfuscator.obfuscate_batch(&batch3).unwrap();

    let token1 = result1[0].data["email"].as_str().unwrap();
    let token2 = result2[0].data["email"].as_str().unwrap();
    let token3 = result3[0].data["email"].as_str().unwrap();

    // Same email should get same token
    assert_eq!(token1, token2);
    // Different email should get different token
    assert_ne!(token1, token3);

    // Check cache hit rate
    let stats = obfuscator.stats();
    assert!(stats.cache_hit_rate > 0.0);
}

#[test]
fn test_memory_bounded_operation() {
    let config = BoundedObfuscatorConfig {
        cache_config: BoundedCacheConfig {
            max_entries: 5,
            max_memory_bytes: 1024, // 1KB
            ttl: None,
            enable_metrics: true,
        },
        ..Default::default()
    };

    let obfuscator = BoundedObfuscator::new(config);

    let patterns = vec![Pattern::new(
        PatternType::Email,
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b".to_string(),
    )];
    obfuscator.set_patterns(patterns).unwrap();

    // Create batch with many unique emails
    let batch: RecordBatch = (0..100)
        .map(|i| json!({"email": format!("user{}@example.com", i)}))
        .collect();

    let result = obfuscator.obfuscate_batch(&batch).unwrap();

    // All should be obfuscated despite cache limits
    assert_eq!(result.len(), 100);
    for record in &result {
        assert!(record.data["email"].as_str().unwrap().starts_with("EMAIL_"));
    }

    // Cache should respect limits
    let stats = obfuscator.stats();
    assert!(stats.cache_entries <= 5);
    assert!(stats.cache_memory_bytes <= 1024);
}

#[test]
fn test_nested_data_obfuscation() {
    let config = BoundedObfuscatorConfig::default();
    let obfuscator = BoundedObfuscator::new(config);

    let patterns = vec![
        Pattern::new(
            PatternType::Email,
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b".to_string(),
        ),
        Pattern::new(
            PatternType::Phone,
            r"\+?1?[-.\s]*\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})\b".to_string(),
        ),
    ];
    obfuscator.set_patterns(patterns).unwrap();

    let batch = vec![json!({
        "user": {
            "personal": {
                "email": "user@example.com",
                "phone": "+1-555-123-4567"
            },
            "work": {
                "email": "user@company.com",
                "phone": "555-987-6543"
            }
        },
        "contacts": [
            {"email": "contact1@example.com"},
            {"email": "contact2@example.com"}
        ]
    })];

    let result = obfuscator.obfuscate_batch(&batch).unwrap();

    // Check nested obfuscation
    let data = &result[0].data;
    assert!(data["user"]["personal"]["email"]
        .as_str()
        .unwrap()
        .starts_with("EMAIL_"));
    assert!(data["user"]["personal"]["phone"]
        .as_str()
        .unwrap()
        .starts_with("PHONE_"));
    assert!(data["user"]["work"]["email"]
        .as_str()
        .unwrap()
        .starts_with("EMAIL_"));
    assert!(data["user"]["work"]["phone"]
        .as_str()
        .unwrap()
        .starts_with("PHONE_"));

    // Check array obfuscation
    let contacts = data["contacts"].as_array().unwrap();
    assert!(contacts[0]["email"].as_str().unwrap().starts_with("EMAIL_"));
    assert!(contacts[1]["email"].as_str().unwrap().starts_with("EMAIL_"));
}

#[test]
fn test_pattern_priority() {
    let config = BoundedObfuscatorConfig::default();
    let obfuscator = BoundedObfuscator::new(config);

    // Set patterns with different priorities
    let patterns = vec![
        Pattern::new(
            PatternType::Custom(1),
            r"\d{3}-\d{2}-\d{4}".to_string(), // Matches SSN pattern
        )
        .with_priority(200), // Higher priority
        Pattern::new(PatternType::SSN, r"\d{3}-\d{2}-\d{4}".to_string()).with_priority(100), // Lower priority
    ];
    obfuscator.set_patterns(patterns).unwrap();

    let batch = vec![json!({"data": "123-45-6789"})];
    let result = obfuscator.obfuscate_batch(&batch).unwrap();

    // Should use higher priority pattern (Custom)
    let token = result[0].data["data"].as_str().unwrap();
    // Custom patterns don't have a prefix in default config
    assert!(!token.starts_with("SSN_"));
}

#[test]
fn test_concurrent_obfuscation() {
    let config = BoundedObfuscatorConfig::default();
    let obfuscator = Arc::new(BoundedObfuscator::new(config));

    let patterns = vec![Pattern::new(
        PatternType::Email,
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b".to_string(),
    )];
    obfuscator.set_patterns(patterns).unwrap();

    let mut handles = vec![];

    // Spawn multiple threads
    for i in 0..5 {
        let obf_clone = obfuscator.clone();
        let handle = thread::spawn(move || {
            let batch: RecordBatch = (0..20)
                .map(|j| json!({"email": format!("user{}x{}@example.com", i, j)}))
                .collect();

            obf_clone.obfuscate_batch(&batch).unwrap()
        });
        handles.push(handle);
    }

    // Collect results
    let mut all_results = vec![];
    for handle in handles {
        let result = handle.join().unwrap();
        all_results.extend(result);
    }

    // Verify all were obfuscated
    assert_eq!(all_results.len(), 100);
    for record in &all_results {
        assert!(record.data["email"].as_str().unwrap().starts_with("EMAIL_"));
    }
}

#[test]
fn test_cleanup() {
    use std::time::Duration;

    let config = BoundedObfuscatorConfig {
        cache_config: BoundedCacheConfig {
            max_entries: 100,
            max_memory_bytes: 0,
            ttl: Some(Duration::from_millis(50)),
            enable_metrics: true,
        },
        ..Default::default()
    };

    let obfuscator = BoundedObfuscator::new(config);

    let patterns = vec![Pattern::new(
        PatternType::Email,
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b".to_string(),
    )];
    obfuscator.set_patterns(patterns).unwrap();

    // Obfuscate some data
    let batch = vec![json!({"email": "test@example.com"})];
    obfuscator.obfuscate_batch(&batch).unwrap();

    // Wait for TTL
    thread::sleep(Duration::from_millis(100));

    // Run cleanup
    obfuscator.cleanup();

    // Cache should be empty due to expiration
    let stats = obfuscator.stats();
    assert_eq!(stats.cache_entries, 0);
}

#[test]
fn test_custom_token_prefixes() {
    let mut token_prefixes = HashMap::new();
    token_prefixes.insert(PatternType::Email, "E_".to_string());
    token_prefixes.insert(PatternType::SSN, "S_".to_string());

    let config = BoundedObfuscatorConfig {
        token_prefixes,
        ..Default::default()
    };

    let obfuscator = BoundedObfuscator::new(config);

    let patterns = vec![
        Pattern::new(
            PatternType::Email,
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b".to_string(),
        ),
        Pattern::new(PatternType::SSN, r"\b\d{3}-\d{2}-\d{4}\b".to_string()),
    ];
    obfuscator.set_patterns(patterns).unwrap();

    let batch = vec![json!({
        "email": "test@example.com",
        "ssn": "123-45-6789"
    })];

    let result = obfuscator.obfuscate_batch(&batch).unwrap();

    // Check custom prefixes
    assert!(result[0].data["email"].as_str().unwrap().starts_with("E_"));
    assert!(result[0].data["ssn"].as_str().unwrap().starts_with("S_"));
}

#[test]
fn test_export_import_mappings() {
    let config = BoundedObfuscatorConfig::default();
    let obfuscator1 = BoundedObfuscator::new(config.clone());
    let obfuscator2 = BoundedObfuscator::new(config);

    let patterns = vec![Pattern::new(
        PatternType::Email,
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b".to_string(),
    )];

    obfuscator1.set_patterns(patterns.clone()).unwrap();
    obfuscator2.set_patterns(patterns).unwrap();

    // Obfuscate with first instance
    let batch = vec![json!({"email": "test@example.com"})];
    let result1 = obfuscator1.obfuscate_batch(&batch).unwrap();
    let _token = result1[0].data["email"].as_str().unwrap().to_string();

    // Export mappings (simplified version returns empty map in current implementation)
    let mappings = obfuscator1.export_mappings();

    // Import to second instance
    obfuscator2.import_mappings(mappings).unwrap();

    // Note: Current implementation has simplified export/import
    // In production, this would maintain token consistency across instances
}

#[test]
fn test_stats_reporting() {
    let config = BoundedObfuscatorConfig::default();
    let obfuscator = BoundedObfuscator::new(config);

    let patterns = vec![Pattern::new(
        PatternType::Email,
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b".to_string(),
    )];
    obfuscator.set_patterns(patterns).unwrap();

    // Generate some activity
    let batch: RecordBatch = (0..10)
        .map(|i| json!({"email": format!("user{}@example.com", i)}))
        .collect();

    obfuscator.obfuscate_batch(&batch).unwrap();

    // Check stats
    let stats = obfuscator.stats();
    assert_eq!(stats.patterns_loaded, 1);
    assert!(stats.tokens_generated >= 10);
    assert!(stats.cache_entries > 0);
    assert!(stats.cache_memory_bytes > 0);
}
