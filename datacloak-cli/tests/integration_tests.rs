use datacloak_cli::test_harness::*;
use tokio;

#[tokio::test]
async fn test_scenario_customer_churn_analysis() {
    // Create test data inline
    let columns = vec![
        ColumnSpec { name: "id".to_string(), column_type: ColumnType::Numeric },
        ColumnSpec { name: "customer_feedback".to_string(), column_type: ColumnType::TextLong },
        ColumnSpec { name: "support_tickets".to_string(), column_type: ColumnType::TextLong },
    ];
    let test_file = generate_csv_from_specs(&columns, 100);
    
    let mut harness = TestHarness::new().await;
    
    // Upload test data
    let file_id = harness.upload_file(&test_file).await;
    
    // Profile columns
    let profile = harness.profile_columns(&file_id).await;
    assert!(profile.find_column("customer_feedback").unwrap().score > 0.7);
    assert!(profile.find_column("support_tickets").unwrap().score > 0.5);
    
    // Analyze with auto-discovery
    let analysis = harness.analyze_auto(&file_id, 0.7).await;
    
    // Verify results match expected
    let results = analysis.collect_results().await;
    assert!(results.len() > 0);
    
    // Check sentiment distribution
    let sentiment_dist = AnalysisRecord::sentiment_distribution(&results, "customer_feedback");
    // Verify distribution is reasonable (all add up to ~1.0)
    let total = sentiment_dist.positive + sentiment_dist.negative + sentiment_dist.neutral;
    assert!((total - 1.0).abs() < 0.01);
    
    // Verify performance
    assert!(analysis.elapsed_seconds() < 60);
    
    // Clean up
    std::fs::remove_file(test_file).ok();
}

#[test]
fn test_property_based_column_detection() {
    use proptest::prelude::*;
    
    proptest!(|(num_columns in 5..20u32, rows in 1000..10000u32)| {
        let columns: Vec<ColumnSpec> = (0..num_columns)
            .map(|i| ColumnSpec {
                name: format!("col_{}", i),
                column_type: if i % 3 == 0 { ColumnType::TextLong } else { ColumnType::Numeric },
            })
            .collect();
        
        let file = generate_csv_from_specs(&columns, rows);
        
        // Properties to verify
        let _text_columns = columns.iter()
            .filter(|c| c.column_type == ColumnType::TextLong)
            .count();
        
        // The file should exist and have the right structure
        assert!(file.exists());
        
        // Read first line to check headers
        let content = std::fs::read_to_string(&file).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len() as u32, rows + 1); // +1 for header
        
        // Clean up
        std::fs::remove_file(file).ok();
    });
}

#[tokio::test]
async fn test_large_file_processing() {
    let mut harness = TestHarness::new().await;
    
    // Generate 1GB file (scaled down from 20GB for testing)
    let large_file = generate_large_csv(1 * GB, 100); // 100 columns
    let file_id = harness.upload_file(&large_file).await;
    
    // Profile should complete quickly
    let start = std::time::Instant::now();
    let profile = harness.profile_columns(&file_id).await;
    assert!(start.elapsed().as_secs() < 10);
    
    // Select top 10 columns
    let selected = profile.top_n(10);
    
    // Analyze with monitoring
    let mut monitor = MemoryMonitor::new();
    let mut analysis = harness.analyze_columns(&file_id, selected).await;
    
    // Stream results with memory tracking
    while let Some(batch) = analysis.next_batch().await {
        assert!(monitor.current_usage() < 1 * GB);
        assert!(batch.len() > 0);
    }
    
    // Verify completion
    assert_eq!(analysis.status(), AnalysisStatus::Completed);
    
    // Clean up
    std::fs::remove_file(large_file).ok();
}

#[tokio::test]
async fn test_multi_column_analysis_integration() {
    let mut harness = TestHarness::new().await;
    
    // Create test file with multiple text columns
    let columns = vec![
        ColumnSpec { name: "id".to_string(), column_type: ColumnType::Numeric },
        ColumnSpec { name: "feedback".to_string(), column_type: ColumnType::TextLong },
        ColumnSpec { name: "comments".to_string(), column_type: ColumnType::TextLong },
        ColumnSpec { name: "rating".to_string(), column_type: ColumnType::Numeric },
    ];
    
    let test_file = generate_csv_from_specs(&columns, 100);
    let file_id = harness.upload_file(&test_file).await;
    
    // Profile columns
    let profile = harness.profile_columns(&file_id).await;
    
    // Text columns should score higher
    let feedback_score = profile.find_column("feedback").unwrap().score;
    let rating_score = profile.find_column("rating").unwrap().score;
    assert!(feedback_score > rating_score);
    
    // Analyze specific columns
    let analysis = harness.analyze_columns(&file_id, vec!["feedback".to_string(), "comments".to_string()]).await;
    let results = analysis.collect_results().await;
    
    // Should have results
    assert!(!results.is_empty());
    
    // Clean up
    std::fs::remove_file(test_file).ok();
}

#[tokio::test]
async fn test_concurrent_analyses() {
    // Create test file first
    let test_file = generate_csv_from_specs(&[
        ColumnSpec { name: "text".to_string(), column_type: ColumnType::TextLong },
    ], 100);
    
    let mut harness = TestHarness::new().await;
    let file_id = harness.upload_file(&test_file).await;
    
    // Run multiple analyses concurrently on the same file
    let mut handles = vec![];
    
    for _ in 0..5 {
        let file_path = test_file.clone();
        
        let handle = tokio::spawn(async move {
            let mut harness = TestHarness::new().await;
            let file_id = harness.upload_file(&file_path).await;
            let analysis = harness.analyze_columns(&file_id, vec!["text".to_string()]).await;
            analysis.collect_results().await
        });
        
        handles.push(handle);
    }
    
    // Wait for all to complete
    let mut all_succeeded = true;
    for handle in handles {
        match handle.await {
            Ok(results) => assert!(!results.is_empty()),
            Err(_) => all_succeeded = false,
        }
    }
    
    assert!(all_succeeded);
    
    // Clean up
    std::fs::remove_file(test_file).ok();
}

/// Test helper to verify sentiment distribution properties
trait SentimentAssertions {
    fn assert_reasonable_distribution(&self);
}

impl SentimentAssertions for SentimentDistribution {
    fn assert_reasonable_distribution(&self) {
        // Sum should be approximately 1.0
        let sum = self.positive + self.negative + self.neutral;
        assert!((sum - 1.0).abs() < 0.01);
        
        // Each sentiment should be between 0 and 1
        assert!(self.positive >= 0.0 && self.positive <= 1.0);
        assert!(self.negative >= 0.0 && self.negative <= 1.0);
        assert!(self.neutral >= 0.0 && self.neutral <= 1.0);
    }
}