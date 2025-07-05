use datacloak_core::*;
use futures::stream;
use serde_json::json;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

#[tokio::test]
async fn test_end_to_end_adaptive_detection_and_obfuscation() {
    // Initialize with optimized thread pools
    let thread_config = thread_config::get_optimized_config();
    let mut datacloak_config = DataCloakConfig::default();
    datacloak_config.thread_pool_config = thread_config;

    // Enable adaptive sampling
    datacloak_config.sampling_config = SamplingConfig {
        min_sample: 100,
        max_sample: 5000,
        confidence_threshold: 0.85,
        progressive_factor: 2.0,
        early_stop_enabled: true,
        confidence_window: 3,
    };

    let datacloak = DataCloak::new_with_thread_pools(datacloak_config).unwrap();

    // Create test data source
    let mut records = vec![];
    for i in 0..10000 {
        records.push(json!({
            "id": i,
            "email": format!("user{}@example.com", i),
            "ssn": if i % 10 == 0 {
                format!("{:03}-{:02}-{:04}", i % 1000, (i / 10) % 100, i % 10000)
            } else {
                "N/A".to_string()
            },
            "phone": if i % 5 == 0 {
                format!("+1-555-{:03}-{:04}", i % 1000, i % 10000)
            } else {
                "N/A".to_string()
            },
            "description": format!("User {} description with no PII", i),
        }));
    }

    let source = DataSource::new(DataSourceConfig::Memory {
        data: records.clone(),
    });

    // Phase 1: Adaptive Detection
    let start = Instant::now();
    let detection_result = datacloak.detect_patterns_adaptive(source).await.unwrap();
    let detection_time = start.elapsed();

    println!("Adaptive detection completed in {:?}", detection_time);
    println!("Rows scanned: {:?}", detection_result.rows_scanned);
    println!("Patterns found: {:?}", detection_result.pattern_counts);
    println!("Confidence: {:?}", detection_result.confidence_score);

    // Verify adaptive sampling worked
    assert!(matches!(
        detection_result.sampling_strategy,
        Some(SamplingStrategy::EarlyStop) | Some(SamplingStrategy::Adaptive)
    ));
    assert!(detection_result.rows_scanned.unwrap() < 10000);
    assert!(detection_result.confidence_score.unwrap() >= 0.6);

    // Phase 2: Set patterns for obfuscation based on detection
    let mut patterns = vec![];
    for (pattern_type, _) in &detection_result.pattern_counts {
        match pattern_type {
            PatternType::Email => patterns.push(Pattern::new(
                PatternType::Email,
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b".to_string(),
            )),
            PatternType::SSN => patterns.push(Pattern::new(
                PatternType::SSN,
                r"\b\d{3}-\d{2}-\d{4}\b".to_string(),
            )),
            PatternType::Phone => patterns.push(Pattern::new(
                PatternType::Phone,
                r"\+?1?\s*\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})\b".to_string(),
            )),
            _ => {}
        }
    }

    datacloak.set_patterns(patterns).unwrap();

    // Phase 3: Obfuscate data
    let start = Instant::now();
    let obfuscated = datacloak
        .obfuscate_batch(records[..100].to_vec())
        .await
        .unwrap();
    let obfuscation_time = start.elapsed();

    println!("Obfuscation completed in {:?}", obfuscation_time);

    // Verify obfuscation
    for (i, record) in obfuscated.iter().enumerate() {
        let email_token = record.data["email"].as_str().unwrap();
        assert!(
            email_token.contains("EMAIL") && !email_token.contains("@"),
            "Email should be obfuscated: {}",
            email_token
        );

        if i % 10 == 0 {
            let ssn_token = record.data["ssn"].as_str().unwrap();
            assert!(
                ssn_token.contains("SSN"),
                "SSN should be obfuscated: {}",
                ssn_token
            );
        }
        if i % 5 == 0 {
            let phone_token = record.data["phone"].as_str().unwrap();
            assert!(
                phone_token.contains("PHONE"),
                "Phone should be obfuscated: {}",
                phone_token
            );
        }
    }
}

#[tokio::test]
async fn test_streaming_detection_with_bounded_cache() {
    // Create custom configuration with bounded cache
    let mut config = DataCloakConfig::default();
    config.stream_detection_config = StreamDetectionConfig {
        max_concurrent_batches: 4,
        channel_buffer_size: 100,
        batch_size: 100,
        use_adaptive_sampling: true,
        confidence_threshold: 0.9,
        max_rows_to_scan: 5000,
    };

    let datacloak = DataCloak::new(config);

    // Create bounded obfuscator for comparison
    let obf_config = BoundedObfuscatorConfig {
        cache_config: BoundedCacheConfig {
            max_entries: 1000,
            max_memory_bytes: 10 * 1024 * 1024, // 10MB
            ttl: Some(Duration::from_secs(300)),
            enable_metrics: true,
        },
        ..Default::default()
    };

    let bounded_obfuscator = BoundedObfuscator::new(obf_config);

    // Create data stream
    let batches: Vec<RecordBatch> = (0..50)
        .map(|batch_idx| {
            (0..100)
                .map(|item_idx| {
                    let idx = batch_idx * 100 + item_idx;
                    json!({
                        "id": idx,
                        "email": format!("user{}@example.com", idx),
                        "credit_card": if idx % 20 == 0 {
                            "4111111111111111"
                        } else {
                            "N/A"
                        },
                        "ip_address": if idx % 15 == 0 {
                            format!("192.168.1.{}", idx % 256)
                        } else {
                            "N/A".to_string()
                        }
                    })
                })
                .collect()
        })
        .collect();

    let stream = stream::iter(batches.clone().into_iter().map(Ok));

    // Create progress channel
    let (progress_tx, mut progress_rx) = mpsc::channel(10);

    // Start streaming detection
    let detection_handle = tokio::spawn({
        let datacloak = datacloak.clone();
        async move {
            datacloak
                .detect_patterns_stream(stream, Some(progress_tx))
                .await
        }
    });

    // Monitor progress
    let mut updates = vec![];
    let progress_handle = tokio::spawn(async move {
        while let Some(update) = progress_rx.recv().await {
            println!(
                "Progress: {} rows, {} patterns, confidence: {:.3}",
                update.rows_processed, update.patterns_found, update.confidence
            );
            updates.push(update);
        }
        updates
    });

    // Wait for detection
    let detection_result = detection_handle.await.unwrap().unwrap();
    let progress_updates = progress_handle.await.unwrap();

    // Verify results
    assert!(detection_result.rows_scanned.unwrap() <= 5000);
    assert!(detection_result
        .pattern_counts
        .contains_key(&PatternType::Email));
    assert!(detection_result
        .pattern_counts
        .contains_key(&PatternType::CreditCard));
    assert!(detection_result
        .pattern_counts
        .contains_key(&PatternType::IPAddress));

    // Verify progress updates
    assert!(!progress_updates.is_empty());

    // Set patterns for bounded obfuscator
    let patterns = vec![
        Pattern::new(PatternType::Email, r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b".to_string()),
        Pattern::new(PatternType::CreditCard, r"\b4[0-9]{12}(?:[0-9]{3})?\b".to_string()),
        Pattern::new(PatternType::IPAddress, r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b".to_string()),
    ];
    bounded_obfuscator.set_patterns(patterns).unwrap();

    // Obfuscate first batch with bounded cache
    let _obfuscated = bounded_obfuscator.obfuscate_batch(&batches[0]).unwrap();

    // Check cache metrics
    let stats = bounded_obfuscator.stats();
    println!(
        "Cache stats - entries: {}, memory: {} KB, hit rate: {:.2}%",
        stats.cache_entries,
        stats.cache_memory_bytes / 1024,
        stats.cache_hit_rate * 100.0
    );

    assert!(stats.cache_entries > 0);
    assert!(stats.cache_memory_bytes > 0);
}

#[tokio::test]
async fn test_high_performance_pipeline() {
    // Configure for maximum performance
    let thread_config = ThreadPoolConfig {
        rayon_threads: num_cpus::get(),
        rayon_thread_prefix: "perf-cpu".to_string(),
        rayon_stack_size: 8 * 1024 * 1024,
        enable_cpu_affinity: false,
        tokio_worker_threads: num_cpus::get(),
        tokio_blocking_threads: num_cpus::get() * 2,
        tokio_thread_prefix: "perf-io".to_string(),
    };

    let mut config = DataCloakConfig::default();
    config.thread_pool_config = thread_config;
    config.batch_size = 1000;
    config.max_concurrency = 8;

    // Fast sampling for performance
    config.sampling_config = SamplingConfig {
        min_sample: 500,
        max_sample: 2000,
        confidence_threshold: 0.8,
        progressive_factor: 3.0,
        early_stop_enabled: true,
        confidence_window: 2,
    };

    // High-throughput streaming
    config.stream_detection_config = StreamDetectionConfig {
        max_concurrent_batches: 8,
        channel_buffer_size: 200,
        batch_size: 500,
        use_adaptive_sampling: true,
        confidence_threshold: 0.8,
        max_rows_to_scan: 10000,
    };

    let datacloak = DataCloak::new_with_thread_pools(config).unwrap();

    // Generate large dataset
    let start_gen = Instant::now();
    let batches: Vec<RecordBatch> = (0..100)
        .map(|batch_idx| {
            (0..1000)
                .map(|item_idx| {
                    let idx = batch_idx * 1000 + item_idx;
                    json!({
                        "id": idx,
                        "email": format!("user{}@example.com", idx),
                        "ssn": format!("{:09}", idx),
                        "phone": format!("+1-555-{:07}", idx % 10000000),
                        "data": format!("Random data {} with no PII information", idx),
                    })
                })
                .collect()
        })
        .collect();
    let gen_time = start_gen.elapsed();
    println!("Generated 100k records in {:?}", gen_time);

    // Stream detection
    let stream = stream::iter(batches.clone().into_iter().map(Ok));
    let start_detect = Instant::now();
    let detection_result = datacloak
        .detect_patterns_stream(stream, None)
        .await
        .unwrap();
    let detect_time = start_detect.elapsed();

    println!(
        "Stream detection of 100k records completed in {:?}",
        detect_time
    );
    println!(
        "Throughput: {:.0} records/second",
        detection_result.rows_scanned.unwrap() as f64 / detect_time.as_secs_f64()
    );

    // Set patterns
    let patterns = vec![
        Pattern::new(
            PatternType::Email,
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b".to_string(),
        ),
        Pattern::new(PatternType::SSN, r"\b\d{9}\b".to_string()),
        Pattern::new(PatternType::Phone, r"\+1-555-\d{7}\b".to_string()),
    ];
    datacloak.set_patterns(patterns).unwrap();

    // Parallel obfuscation
    let start_obf = Instant::now();
    let mut obfuscated_count = 0;

    use rayon::prelude::*;
    let obfuscated_batches: Vec<_> = batches[..10]
        .par_iter()
        .map(|batch| datacloak.obfuscator.obfuscate_batch(batch).unwrap())
        .collect();

    for batch in &obfuscated_batches {
        obfuscated_count += batch.len();
    }

    let obf_time = start_obf.elapsed();
    println!("Obfuscated {} records in {:?}", obfuscated_count, obf_time);
    println!(
        "Throughput: {:.0} records/second",
        obfuscated_count as f64 / obf_time.as_secs_f64()
    );

    // Verify performance targets
    assert!(detect_time < Duration::from_secs(10)); // Should process 100k in < 10s
    assert!(obf_time < Duration::from_secs(2)); // Should obfuscate 10k in < 2s
}

#[tokio::test]
async fn test_memory_pressure_handling() {
    // Configure with strict memory limits
    let _config = DataCloakConfig::default();

    // Use bounded obfuscator with memory limits
    let obf_config = BoundedObfuscatorConfig {
        cache_config: BoundedCacheConfig {
            max_entries: 100,
            max_memory_bytes: 1024 * 1024, // 1MB limit
            ttl: Some(Duration::from_secs(60)),
            enable_metrics: true,
        },
        ..Default::default()
    };

    let bounded_obfuscator = Arc::new(BoundedObfuscator::new(obf_config));

    // Set patterns
    let patterns = vec![
        Pattern::new(
            PatternType::Email,
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b".to_string(),
        ),
        Pattern::new(PatternType::SSN, r"\b\d{3}-\d{2}-\d{4}\b".to_string()),
    ];
    bounded_obfuscator.set_patterns(patterns).unwrap();

    // Generate data that will exceed cache limits
    let mut handles = vec![];

    for thread_id in 0..5 {
        let obf_clone = bounded_obfuscator.clone();
        let handle = tokio::spawn(async move {
            let batch: RecordBatch = (0..1000)
                .map(|i| {
                    json!({
                        "id": format!("{}_{}", thread_id, i),
                        "email": format!("user_{}_{}_unique@example.com", thread_id, i),
                        "ssn": format!("{:03}-{:02}-{:04}",
                               thread_id * 100 + (i % 100),
                               (i / 100) % 100,
                               i % 10000),
                    })
                })
                .collect();

            obf_clone.obfuscate_batch(&batch).unwrap()
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.await.unwrap();
    }

    // Check that memory limits were respected
    let stats = bounded_obfuscator.stats();
    println!(
        "Final cache stats - entries: {}, memory: {} KB",
        stats.cache_entries,
        stats.cache_memory_bytes / 1024
    );

    assert!(stats.cache_memory_bytes <= 1024 * 1024);
    assert!(stats.cache_entries <= 100);

    // Run cleanup
    bounded_obfuscator.cleanup();
}

#[tokio::test]
async fn test_error_recovery() {
    use datacloak_core::DataCloakError;

    let config = DataCloakConfig::default();
    let datacloak = DataCloak::new(config);

    // Create stream with some errors
    let batches: Vec<Result<RecordBatch>> = vec![
        Ok(vec![json!({"email": "test1@example.com"})]),
        Err(DataCloakError::Other("Network error".to_string())),
        Ok(vec![json!({"email": "test2@example.com"})]),
        Err(DataCloakError::Other("Timeout".to_string())),
        Ok(vec![json!({"email": "test3@example.com"})]),
    ];

    let stream = stream::iter(batches);

    // Detection should continue despite errors
    let result = datacloak
        .detect_patterns_stream(stream, None)
        .await
        .unwrap();

    assert_eq!(result.rows_scanned, Some(3));
    assert_eq!(result.pattern_counts[&PatternType::Email], 3);
}
