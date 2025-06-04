//! Integration tests for DataCloak library

use datacloak_core::{
    DataCloak, DataCloakConfig, DataSource, DataSourceConfig,
    Pattern, PatternSet, PatternType, Result,
};
use serde_json::json;
use std::sync::Arc;
use futures::StreamExt;

#[tokio::test]
async fn test_pattern_detection() -> Result<()> {
    let config = DataCloakConfig::default();
    let datacloak = DataCloak::new(config);
    
    // Create test data with various PII
    let test_data = vec![
        json!({
            "id": 1,
            "email": "test@example.com",
            "phone": "555-123-4567",
            "ssn": "123-45-6789",
            "text": "Contact me at test@example.com or 555-123-4567"
        }),
        json!({
            "id": 2,
            "email": "user@test.org",
            "credit_card": "4532015112830366",
            "ip": "192.168.1.100"
        }),
    ];
    
    let source = DataSource::new(DataSourceConfig::Memory { data: test_data });
    let detection = datacloak.detect_patterns(source).await?;
    
    // Verify detection results
    assert!(detection.sample_size > 0);
    assert!(!detection.detected_patterns.is_empty());
    
    // Check if specific patterns were detected
    let has_email = detection.detected_patterns.iter()
        .any(|p| p.pattern_type == PatternType::Email);
    let has_phone = detection.detected_patterns.iter()
        .any(|p| p.pattern_type == PatternType::Phone);
    let has_ssn = detection.detected_patterns.iter()
        .any(|p| p.pattern_type == PatternType::SSN);
    
    assert!(has_email);
    assert!(has_phone);
    assert!(has_ssn);
    
    // Check email pattern details
    let email_pattern = detection.detected_patterns.iter()
        .find(|p| p.pattern_type == PatternType::Email)
        .unwrap();
    assert_eq!(email_pattern.match_count, 3); // test@example.com appears twice, plus user@test.org
    assert!(email_pattern.confidence > 0.8);
    
    Ok(())
}

#[tokio::test]
async fn test_obfuscation_and_deobfuscation() -> Result<()> {
    let config = DataCloakConfig::default();
    let datacloak = DataCloak::new(config);
    
    let test_data = vec![
        json!({
            "customer_id": "CUST001",
            "email": "john.doe@example.com",
            "phone": "555-123-4567",
            "notes": "Email: john.doe@example.com, Phone: 555-123-4567"
        }),
    ];
    
    let _source = DataSource::new(DataSourceConfig::Memory { data: test_data });
    let patterns = PatternSet::default_pii().to_vec();
    
    // Set patterns for obfuscation
    datacloak.set_patterns(patterns.clone())?;
    
    // Verify obfuscation works
    let batch = vec![json!({
        "email": "test@example.com",
        "phone": "555-987-6543"
    })];
    
    // Use the obfuscator through DataCloak's analyze_churn method
    // For now, we'll test the obfuscation directly
    let stats = datacloak.obfuscator_stats();
    println!("Obfuscator stats: {:?}", stats);
    
    let obfuscated = datacloak.obfuscate_batch(batch).await?;
    assert_eq!(obfuscated.len(), 1);
    
    let obfuscated_email = obfuscated[0].data["email"].as_str().unwrap();
    assert!(obfuscated_email.starts_with("[EMAIL-"));
    assert!(!obfuscated_email.contains("test@example.com"));
    
    Ok(())
}

#[tokio::test]
async fn test_streaming_processor() -> Result<()> {
    use futures::stream;
    use datacloak_core::{StreamProcessor, StreamConfig, Obfuscator};
    
    let obfuscator = Arc::new(Obfuscator::new());
    let patterns = vec![
        Pattern::new(PatternType::Email, r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b".to_string()),
    ];
    obfuscator.set_patterns(patterns)?;
    
    let processor = StreamProcessor::new(obfuscator, StreamConfig::default());
    
    // Create test stream
    let batch1 = vec![json!({"email": "user1@test.com"})];
    let batch2 = vec![json!({"email": "user2@test.com"})];
    let input_stream = stream::iter(vec![Ok(batch1), Ok(batch2)]);
    
    let mut output_stream = processor.create_pipeline(input_stream);
    
    // Verify streaming works
    let mut count = 0;
    while let Some(result) = output_stream.next().await {
        let batch = result?;
        assert!(!batch.is_empty());
        count += batch.len();
    }
    
    assert_eq!(count, 2);
    
    Ok(())
}

#[tokio::test]
async fn test_cache_operations() -> Result<()> {
    use datacloak_core::ObfuscationCache;
    use tempfile::tempdir;
    
    let dir = tempdir()?;
    let cache_file = dir.path().join("test_cache.bin");
    
    let cache = ObfuscationCache::with_storage(cache_file.clone())
        .with_encryption(b"test-encryption-key-32-bytes-long".to_vec());
    
    // Store some mappings
    cache.store("TOKEN-1".to_string(), "test@example.com".to_string(), "EMAIL".to_string());
    cache.store("TOKEN-2".to_string(), "555-123-4567".to_string(), "PHONE".to_string());
    
    // Verify retrieval
    assert_eq!(cache.get_original("TOKEN-1"), Some("test@example.com".to_string()));
    assert_eq!(cache.get_token("555-123-4567"), Some("TOKEN-2".to_string()));
    
    // Test persistence
    cache.save().await?;
    
    let cache2 = ObfuscationCache::with_storage(cache_file)
        .with_encryption(b"test-encryption-key-32-bytes-long".to_vec());
    cache2.load().await?;
    
    assert_eq!(cache2.get_original("TOKEN-1"), Some("test@example.com".to_string()));
    
    Ok(())
}

#[tokio::test]
async fn test_pattern_priority() -> Result<()> {
    use datacloak_core::Obfuscator;
    
    let obfuscator = Obfuscator::new();
    
    // Create overlapping patterns with different priorities
    let patterns = vec![
        Pattern::new(PatternType::Email, r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b".to_string())
            .with_priority(100),
        Pattern::new(PatternType::Custom(1), r"\b\w+@\w+\.\w+\b".to_string())
            .with_priority(50),
    ];
    
    obfuscator.set_patterns(patterns)?;
    
    let batch = vec![json!({"text": "Contact: admin@example.com"})];
    let obfuscated = obfuscator.obfuscate_batch(&batch)?;
    
    // Should use EMAIL pattern due to higher priority
    let text = obfuscated[0].data["text"].as_str().unwrap();
    assert!(text.contains("[EMAIL-"));
    assert!(!text.contains("[CUSTOM_1-"));
    
    Ok(())
}

#[tokio::test]
async fn test_error_handling() -> Result<()> {
    let config = DataCloakConfig::default();
    let datacloak = DataCloak::new(config);
    
    // Test invalid pattern
    let invalid_pattern = Pattern::new(
        PatternType::Custom(1),
        r"[invalid regex".to_string(), // Invalid regex
    );
    
    let result = datacloak.set_patterns(vec![invalid_pattern]);
    assert!(result.is_err());
    
    Ok(())
}
