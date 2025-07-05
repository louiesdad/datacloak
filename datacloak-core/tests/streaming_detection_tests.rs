use datacloak_core::{
    AdaptiveSampler, PatternDetector, PatternType, RecordBatch, Result, SamplingConfig,
    StreamDetectionConfig, StreamDetectionProcessor,
};
use futures::stream::{self, StreamExt};
use serde_json::json;
use std::sync::Arc;
use tokio::sync::mpsc;

#[tokio::test]
async fn test_basic_streaming_detection() {
    let detector = Arc::new(PatternDetector::new(0.1));
    let sampler = Arc::new(AdaptiveSampler::new(SamplingConfig::default()));
    let config = StreamDetectionConfig::default();

    let processor = StreamDetectionProcessor::new(detector, sampler, config);

    // Create test stream
    let batches = vec![
        vec![
            json!({"email": "test1@example.com", "data": "some data"}),
            json!({"email": "test2@example.com", "ssn": "123-45-6789"}),
        ],
        vec![
            json!({"phone": "+1-555-123-4567", "data": "more data"}),
            json!({"email": "test3@example.com", "credit_card": "4111111111111111"}),
        ],
    ];

    let stream = stream::iter(batches.into_iter().map(Ok::<RecordBatch, DataCloakError>));

    let result = processor.detect_stream(stream, None).await.unwrap();

    assert_eq!(result.rows_scanned, Some(4));
    assert!(result.pattern_counts.contains_key(&PatternType::Email));
    assert_eq!(result.pattern_counts[&PatternType::Email], 3);
    assert!(result.pattern_counts.contains_key(&PatternType::SSN));
    assert!(result.pattern_counts.contains_key(&PatternType::Phone));
}

#[tokio::test]
async fn test_streaming_with_progress_updates() {
    let detector = Arc::new(PatternDetector::new(0.1));
    let sampler = Arc::new(AdaptiveSampler::new(SamplingConfig::default()));
    let config = StreamDetectionConfig {
        max_concurrent_batches: 2,
        channel_buffer_size: 10,
        batch_size: 2,
        use_adaptive_sampling: true,
        confidence_threshold: 0.9,
        max_rows_to_scan: 0,
    };

    let processor = StreamDetectionProcessor::new(detector, sampler, config);

    // Create progress channel
    let (progress_tx, mut progress_rx) = mpsc::channel(10);

    // Create test stream with multiple batches
    let batches: Vec<RecordBatch> = (0..5)
        .map(|i| {
            vec![
                json!({"id": i * 2, "email": format!("test{}@example.com", i * 2)}),
                json!({"id": i * 2 + 1, "email": format!("test{}@example.com", i * 2 + 1)}),
            ]
        })
        .collect();

    let stream = stream::iter(batches.into_iter().map(Ok));

    // Start detection with progress updates
    let detection_handle =
        tokio::spawn(async move { processor.detect_stream(stream, Some(progress_tx)).await });

    // Collect progress updates
    let mut updates = vec![];
    while let Some(update) = progress_rx.recv().await {
        updates.push(update);
    }

    let result = detection_handle.await.unwrap().unwrap();

    // Verify progress updates
    assert!(!updates.is_empty());
    assert_eq!(updates.last().unwrap().rows_processed, 10);

    // Verify increasing progress
    for window in updates.windows(2) {
        assert!(window[1].rows_processed >= window[0].rows_processed);
    }

    assert_eq!(result.rows_scanned, Some(10));
}

#[tokio::test]
async fn test_streaming_early_stop() {
    let detector = Arc::new(PatternDetector::new(0.01));
    let sampler = Arc::new(AdaptiveSampler::new(SamplingConfig::default()));

    let config = StreamDetectionConfig {
        max_concurrent_batches: 4,
        channel_buffer_size: 100,
        batch_size: 10,
        use_adaptive_sampling: true,
        confidence_threshold: 0.5, // Low threshold for early stop
        max_rows_to_scan: 0,
    };

    let processor = StreamDetectionProcessor::new(detector, sampler, config);

    // Create stream with high PII density
    let batches: Vec<RecordBatch> = (0..20)
        .map(|i| {
            (0..10)
                .map(|j| {
                    json!({
                        "email": format!("user{}@example.com", i * 10 + j),
                        "ssn": format!("{:03}-{:02}-{:04}", i, j, i * j),
                        "phone": format!("+1-555-{:03}-{:04}", i, j),
                    })
                })
                .collect()
        })
        .collect();

    let stream = stream::iter(batches.into_iter().map(Ok));

    let (progress_tx, mut progress_rx) = mpsc::channel(10);

    let result = processor
        .detect_stream(stream, Some(progress_tx))
        .await
        .unwrap();

    // Should stop early
    assert!(result.rows_scanned.unwrap() < 200); // Less than total (20 * 10)
    assert_eq!(result.sampling_strategy, Some(SamplingStrategy::EarlyStop));

    // Check for early stop signal in progress
    let mut found_stop = false;
    while let Ok(update) = progress_rx.try_recv() {
        if update.should_stop {
            found_stop = true;
            break;
        }
    }
    assert!(found_stop);
}

#[tokio::test]
async fn test_streaming_max_rows_limit() {
    let detector = Arc::new(PatternDetector::new(0.1));
    let sampler = Arc::new(AdaptiveSampler::new(SamplingConfig::default()));

    let config = StreamDetectionConfig {
        max_concurrent_batches: 4,
        channel_buffer_size: 100,
        batch_size: 10,
        use_adaptive_sampling: false,
        confidence_threshold: 0.95,
        max_rows_to_scan: 25, // Limit to 25 rows
    };

    let processor = StreamDetectionProcessor::new(detector, sampler, config);

    // Create stream with 50 rows total
    let batches: Vec<RecordBatch> = (0..10)
        .map(|i| {
            (0..5)
                .map(|j| {
                    json!({
                        "id": i * 5 + j,
                        "email": format!("user{}@example.com", i * 5 + j),
                    })
                })
                .collect()
        })
        .collect();

    let stream = stream::iter(batches.into_iter().map(Ok));

    let result = processor.detect_stream(stream, None).await.unwrap();

    // Should respect max rows limit
    assert_eq!(result.rows_scanned, Some(25));
}

#[tokio::test]
async fn test_streaming_error_handling() {
    use datacloak_core::DataCloakError;

    let detector = Arc::new(PatternDetector::new(0.1));
    let sampler = Arc::new(AdaptiveSampler::new(SamplingConfig::default()));
    let config = StreamDetectionConfig::default();

    let processor = StreamDetectionProcessor::new(detector, sampler, config);

    // Create stream with errors
    let batches: Vec<Result<RecordBatch>> = vec![
        Ok(vec![json!({"email": "test1@example.com"})]),
        Err(DataCloakError::Other("Simulated error".to_string())),
        Ok(vec![json!({"email": "test2@example.com"})]),
    ];

    let stream = stream::iter(batches);

    let result = processor.detect_stream(stream, None).await.unwrap();

    // Should continue processing after error
    assert_eq!(result.rows_scanned, Some(2));
    assert_eq!(result.pattern_counts[&PatternType::Email], 2);
}

#[tokio::test]
async fn test_concurrent_batch_processing() {
    use std::time::{Duration, Instant};

    let detector = Arc::new(PatternDetector::new(0.1));
    let sampler = Arc::new(AdaptiveSampler::new(SamplingConfig::default()));

    // Configure for high concurrency
    let config = StreamDetectionConfig {
        max_concurrent_batches: 8,
        channel_buffer_size: 100,
        batch_size: 100,
        use_adaptive_sampling: false,
        confidence_threshold: 0.95,
        max_rows_to_scan: 0,
    };

    let processor = StreamDetectionProcessor::new(detector, sampler, config);

    // Create large stream
    let batches: Vec<RecordBatch> = (0..20)
        .map(|i| {
            (0..100)
                .map(|j| {
                    json!({
                        "id": i * 100 + j,
                        "email": format!("user{}@example.com", i * 100 + j),
                        "data": format!("Random data {}", i * 100 + j),
                    })
                })
                .collect()
        })
        .collect();

    let stream = stream::iter(batches.into_iter().map(Ok));

    let start = Instant::now();
    let result = processor.detect_stream(stream, None).await.unwrap();
    let duration = start.elapsed();

    assert_eq!(result.rows_scanned, Some(2000));
    assert_eq!(result.pattern_counts[&PatternType::Email], 2000);

    // Should process efficiently with concurrency
    println!("Processed 2000 rows in {:?}", duration);
    assert!(duration < Duration::from_secs(5));
}

#[tokio::test]
async fn test_streaming_pipeline() {
    let detector = Arc::new(PatternDetector::new(0.1));
    let sampler = Arc::new(AdaptiveSampler::new(SamplingConfig::default()));
    let config = StreamDetectionConfig::default();

    let processor = StreamDetectionProcessor::new(detector, sampler, config);

    // Create test stream
    let batches = vec![
        vec![json!({"email": "test@example.com"})],
        vec![json!({"ssn": "123-45-6789"})],
    ];

    let input_stream = stream::iter(batches.into_iter().map(Ok));

    // Create pipeline
    let mut result_stream = Box::pin(processor.create_pipeline(input_stream));

    // Collect results
    let mut results = vec![];
    while let Some(result) = result_stream.next().await {
        results.push(result);
    }

    assert_eq!(results.len(), 1);
    let detection_result = results[0].as_ref().unwrap();
    assert_eq!(detection_result.rows_scanned, Some(2));
}

#[tokio::test]
async fn test_column_specific_streaming() {
    let detector = Arc::new(PatternDetector::new(0.1));
    let sampler = Arc::new(AdaptiveSampler::new(SamplingConfig::default()));
    let config = StreamDetectionConfig::default();

    let processor = StreamDetectionProcessor::new(detector, sampler, config);

    // Create stream with column-specific patterns
    let batches = vec![vec![
        json!({
            "user_email": "user1@example.com",
            "contact_email": "contact1@example.com",
            "user_ssn": "123-45-6789",
            "data": "non-pii data"
        }),
        json!({
            "user_email": "user2@example.com",
            "contact_email": "contact2@example.com",
            "user_ssn": "987-65-4321",
            "data": "more non-pii data"
        }),
    ]];

    let stream = stream::iter(batches.into_iter().map(Ok));

    let result = processor.detect_stream(stream, None).await.unwrap();

    // Check column mappings
    assert!(result.column_patterns.contains_key("user_email"));
    assert!(result.column_patterns.contains_key("contact_email"));
    assert!(result.column_patterns.contains_key("user_ssn"));
    assert!(!result.column_patterns.contains_key("data"));

    // Verify pattern types per column
    assert!(result.column_patterns["user_email"].contains(&PatternType::Email));
    assert!(result.column_patterns["contact_email"].contains(&PatternType::Email));
    assert!(result.column_patterns["user_ssn"].contains(&PatternType::SSN));
}

// Import DataCloakError for error handling test
use datacloak_core::DataCloakError;
use datacloak_core::SamplingStrategy;
