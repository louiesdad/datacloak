use datacloak_cli::test_environment::*;
use datacloak_cli::test_harness::*;
use tokio;
use std::time::Duration;

#[tokio::test]
async fn test_e2e_auto_discovery_workflow() {
    let env = TestEnvironment::new().await;
    
    // Start all services
    env.start_services(&[
        "mock-llm",
    ]).await.unwrap();
    
    // Create test data
    let columns = vec![
        ColumnSpec { name: "id".to_string(), column_type: ColumnType::Numeric },
        ColumnSpec { name: "customer_feedback".to_string(), column_type: ColumnType::TextLong },
        ColumnSpec { name: "support_notes".to_string(), column_type: ColumnType::TextLong },
        ColumnSpec { name: "order_id".to_string(), column_type: ColumnType::Numeric },
        ColumnSpec { name: "comments".to_string(), column_type: ColumnType::TextLong },
    ];
    let test_file = generate_csv_from_specs(&columns, 100);
    
    // User uploads file via CLI (simulated)
    let file_id = test_file.to_str().unwrap();
    
    // User profiles columns
    let output = env.run_cli(&[
        "profile",
        "--file", file_id,
        "--output", "json"
    ]).await;
    
    // Parse profile output
    let profile: serde_json::Value = serde_json::from_str(&output).unwrap();
    let candidates = profile["candidates"].as_array().unwrap();
    
    // Verify text columns are identified
    let text_columns: Vec<&str> = candidates.iter()
        .filter(|c| c["score"].as_f64().unwrap() > 0.7)
        .map(|c| c["name"].as_str().unwrap())
        .collect();
    
    assert!(text_columns.contains(&"customer_feedback"));
    assert!(text_columns.contains(&"comments"));
    
    // User runs analysis with auto-discovery
    let output = env.run_cli(&[
        "analyze",
        "--file", file_id,
        "--auto-discover",
        "--threshold", "0.75",
        "--output", "json",
        "--mock-llm"
    ]).await;
    
    // Verify analysis results
    let results: Vec<serde_json::Value> = serde_json::from_str(&output).unwrap();
    assert!(results.len() > 0);
    
    // Each result should have sentiment analysis
    for result in &results {
        assert!(result["sentiment"].is_string());
        assert!(result["confidence"].is_number());
    }
    
    // Clean up
    std::fs::remove_file(test_file).ok();
}

#[tokio::test]
async fn test_e2e_failure_recovery() {
    let env = TestEnvironment::new().await;
    env.start_services(&["mock-llm"]).await.unwrap();
    
    // Create large test file
    let test_file = generate_large_csv(100 * MB, 10); // 100MB, 10 columns
    let file_id = test_file.to_str().unwrap();
    
    // Start large analysis
    let run_id = env.start_analysis(file_id, vec!["col_1", "col_2", "col_3"]).await;
    
    // Wait for progress
    env.wait_for_progress(&run_id, 25).await.unwrap();
    
    // Simulate service interruption
    env.kill_service("mock-llm").await.unwrap();
    tokio::time::sleep(Duration::from_secs(2)).await;
    
    // Restart service
    env.start_service("mock-llm").await.unwrap();
    
    // Verify analysis can resume
    let status = env.get_analysis_status(&run_id).await;
    assert_eq!(status.state, "resuming");
    
    // Wait for completion
    env.wait_for_completion(&run_id).await.unwrap();
    
    // Verify no data loss
    let results = env.get_results(&run_id).await;
    assert_eq!(results.total_rows, results.processed_rows);
    
    // Clean up
    std::fs::remove_file(test_file).ok();
}

#[tokio::test]
async fn test_e2e_multi_user_concurrent_workflow() {
    let env = TestEnvironment::new().await;
    env.start_services(&["mock-llm"]).await.unwrap();
    
    // Create test files for different users
    let user_files: Vec<_> = (0..3)
        .map(|i| {
            let columns = vec![
                ColumnSpec { name: format!("user_{}_text", i), column_type: ColumnType::TextLong },
                ColumnSpec { name: format!("user_{}_data", i), column_type: ColumnType::Numeric },
            ];
            generate_csv_from_specs(&columns, 50)
        })
        .collect();
    
    // Simulate concurrent users
    let mut handles = vec![];
    
    for (i, file) in user_files.iter().enumerate() {
        let env_clone = TestEnvironment::new().await;
        let file_path = file.to_str().unwrap().to_string();
        
        let handle = tokio::spawn(async move {
            // Each user profiles their file
            let output = env_clone.run_cli(&[
                "profile",
                "--file", &file_path,
                "--output", "json"
            ]).await;
            
            let profile: serde_json::Value = serde_json::from_str(&output).unwrap();
            assert!(profile["candidates"].as_array().unwrap().len() > 0);
            
            // Each user runs analysis
            let output = env_clone.run_cli(&[
                "analyze",
                "--file", &file_path,
                "--columns", &format!("user_{}_text", i),
                "--output", "json",
                "--mock-llm"
            ]).await;
            
            let results: Vec<serde_json::Value> = serde_json::from_str(&output).unwrap();
            assert!(results.len() > 0);
        });
        
        handles.push(handle);
    }
    
    // Wait for all users to complete
    for handle in handles {
        handle.await.unwrap();
    }
    
    // Verify system metrics
    let metrics = env.get_metrics().await;
    assert!(metrics["datacloak_columns_analyzed"] >= 3.0);
    
    // Clean up
    for file in user_files {
        std::fs::remove_file(file).ok();
    }
}

#[tokio::test]
async fn test_e2e_streaming_workflow() {
    let env = TestEnvironment::new().await;
    env.start_services(&["mock-llm"]).await.unwrap();
    
    // Create test file
    let columns = vec![
        ColumnSpec { name: "id".to_string(), column_type: ColumnType::Numeric },
        ColumnSpec { name: "message".to_string(), column_type: ColumnType::TextLong },
    ];
    let test_file = generate_csv_from_specs(&columns, 200);
    
    // Run analysis with streaming output
    let output = env.run_cli(&[
        "analyze",
        "--file", test_file.to_str().unwrap(),
        "--columns", "message",
        "--output", "stream",
        "--mock-llm"
    ]).await;
    
    // Verify streaming output
    let lines: Vec<&str> = output.lines().collect();
    assert!(lines.len() >= 200); // One line per record
    
    // Each line should be valid JSON
    for line in lines {
        if !line.trim().is_empty() {
            let result: serde_json::Value = serde_json::from_str(line)
                .expect("Each line should be valid JSON");
            assert!(result["record_id"].is_string());
            assert!(result["sentiment"].is_string());
        }
    }
    
    // Clean up
    std::fs::remove_file(test_file).ok();
}

#[tokio::test]
async fn test_e2e_error_handling_workflow() {
    let env = TestEnvironment::new().await;
    
    // Test 1: Invalid file path
    let output = env.run_cli(&[
        "profile",
        "--file", "/nonexistent/file.csv",
        "--output", "json"
    ]).await;
    
    // Should contain error message
    assert!(output.contains("error") || output.is_empty());
    
    // Test 2: Invalid column names
    let test_file = generate_csv_from_specs(&[
        ColumnSpec { name: "col1".to_string(), column_type: ColumnType::TextLong },
    ], 10);
    
    let output = env.run_cli(&[
        "analyze",
        "--file", test_file.to_str().unwrap(),
        "--columns", "nonexistent_column",
        "--output", "json"
    ]).await;
    
    // Should contain error about column not found
    assert!(output.contains("not found") || output.is_empty());
    
    // Test 3: LLM service unavailable
    let output = env.run_cli(&[
        "analyze",
        "--file", test_file.to_str().unwrap(),
        "--columns", "col1",
        "--output", "json"
        // Note: no --mock-llm flag and no real LLM configured
    ]).await;
    
    // Should handle gracefully
    assert!(output.contains("error") || output.contains("neutral"));
    
    // Clean up
    std::fs::remove_file(test_file).ok();
}

#[tokio::test]
async fn test_e2e_performance_monitoring() {
    let env = TestEnvironment::new().await;
    env.start_services(&["mock-llm"]).await.unwrap();
    
    // Create files of different sizes
    let small_file = generate_csv_from_specs(&[
        ColumnSpec { name: "text".to_string(), column_type: ColumnType::TextLong },
    ], 100);
    
    let large_file = generate_csv_from_specs(&[
        ColumnSpec { name: "text".to_string(), column_type: ColumnType::TextLong },
    ], 10000);
    
    // Profile small file
    let start = std::time::Instant::now();
    let _ = env.run_cli(&[
        "profile",
        "--file", small_file.to_str().unwrap(),
        "--output", "json"
    ]).await;
    let small_duration = start.elapsed();
    
    // Profile large file
    let start = std::time::Instant::now();
    let _ = env.run_cli(&[
        "profile",
        "--file", large_file.to_str().unwrap(),
        "--output", "json"
    ]).await;
    let large_duration = start.elapsed();
    
    // Both operations should complete in reasonable time
    assert!(large_duration.as_secs() < 30); // Should complete in reasonable time
    assert!(small_duration.as_secs() < 30); // Small file should also be reasonable
    
    // Log durations for debugging
    eprintln!("Small file profiling took: {:?}", small_duration);
    eprintln!("Large file profiling took: {:?}", large_duration);
    
    // Check metrics
    let metrics = env.get_metrics().await;
    assert!(metrics["max_memory_gb"] < 4.0); // Memory usage should be bounded
    
    // Clean up
    std::fs::remove_file(small_file).ok();
    std::fs::remove_file(large_file).ok();
}

#[tokio::test]
async fn test_e2e_csv_output_format() {
    let env = TestEnvironment::new().await;
    env.start_services(&["mock-llm"]).await.unwrap();
    
    // Create test file with multiple columns
    let columns = vec![
        ColumnSpec { name: "id".to_string(), column_type: ColumnType::Numeric },
        ColumnSpec { name: "feedback".to_string(), column_type: ColumnType::TextLong },
        ColumnSpec { name: "notes".to_string(), column_type: ColumnType::TextLong },
    ];
    let test_file = generate_csv_from_specs(&columns, 5);
    
    // Run analysis with CSV output
    let output = env.run_cli(&[
        "analyze",
        "--file", test_file.to_str().unwrap(),
        "--columns", "feedback,notes",
        "--output", "csv",
        "--mock-llm"
    ]).await;
    
    // Parse CSV output
    let lines: Vec<&str> = output.lines().collect();
    assert!(lines.len() >= 6); // Header + 5 data rows
    
    // Check header
    let header = lines[0];
    assert!(header.contains("id"));
    assert!(header.contains("feedback_sentiment"));
    assert!(header.contains("feedback_confidence"));
    assert!(header.contains("notes_sentiment"));
    assert!(header.contains("notes_confidence"));
    
    // Check data rows have correct number of columns
    let header_columns = header.split(',').count();
    for i in 1..6 {
        let columns = lines[i].split(',').count();
        assert_eq!(columns, header_columns);
    }
    
    // Clean up
    std::fs::remove_file(test_file).ok();
}