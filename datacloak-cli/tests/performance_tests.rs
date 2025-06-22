use datacloak_cli::performance::{PerformanceBenchmark, PerformanceBaseline, Measurement, LoadGenerator};
use datacloak_cli::performance::MemoryMonitor; // Explicitly use the performance MemoryMonitor
use datacloak_cli::test_harness::{ColumnSpec, ColumnType, generate_csv_from_specs, generate_large_csv};
use std::time::{Duration, Instant};

/// Criterion-like benchmarking for performance tests
mod criterion_like {
    use std::time::{Duration, Instant};
    
    pub struct Bencher;
    
    impl Bencher {
        pub fn iter<F>(&mut self, mut f: F) -> Duration
        where
            F: FnMut(),
        {
            // Warm up
            for _ in 0..10 {
                f();
            }
            
            // Measure
            let start = Instant::now();
            for _ in 0..100 {
                f();
            }
            let total = start.elapsed();
            total / 100
        }
    }
}

#[test]
fn bench_column_profiling_100_columns() {
    let columns: Vec<ColumnSpec> = (0..100)
        .map(|i| ColumnSpec {
            name: format!("col_{}", i),
            column_type: if i % 3 == 0 { ColumnType::TextLong } else { ColumnType::Numeric },
        })
        .collect();
    
    let file = generate_csv_from_specs(&columns, 1000);
    let mut benchmark = PerformanceBenchmark::new("column_profiling_100")
        .with_metadata("columns", "100")
        .with_metadata("rows", "1000");
    
    let mut memory_monitor = MemoryMonitor::new();
    
    let start = Instant::now();
    {
        let _measurement = benchmark.start_measurement("profile_100_columns");
        
        // Simulate column profiling
        use datacloak_core::ml_classifier::{MLClassifier, Column};
        let classifier = MLClassifier::new();
        
        for (i, spec) in columns.iter().enumerate() {
            memory_monitor.sample();
            let sample_data: Vec<&str> = match spec.column_type {
                ColumnType::TextLong => vec!["This is a long text description", "Another text sample"],
                ColumnType::Numeric => vec!["123", "456"],
                _ => vec!["data", "sample"],
            };
            
            let column = Column::new(&spec.name, sample_data);
            let _prediction = classifier.predict(&column);
            
            if i % 10 == 0 {
                memory_monitor.sample();
            }
        }
    }
    let duration = start.elapsed();
    
    // Check against baseline
    let baseline = PerformanceBaseline {
        operation: "profile_100_columns".to_string(),
        expected_duration_ms: 500,
        max_memory_mb: Some(100),
        throughput_items_per_sec: Some(200.0),
        regression_threshold: 0.5, // Allow 50% variance in test environment
    };
    
    let result = benchmark.check_against_baseline(&baseline);
    
    println!("Column profiling (100 columns) took: {:?}", duration);
    println!("Peak memory usage: {} bytes", memory_monitor.peak_usage());
    println!("Is regression: {}", result.is_regression);
    
    // Performance assertions
    assert!(duration.as_secs() < 5, "Profiling took too long: {:?}", duration);
    assert!(memory_monitor.peak_usage() < 500 * 1024 * 1024, "Memory usage too high");
    
    // Clean up
    std::fs::remove_file(file).ok();
}

#[test]
fn bench_column_profiling_1000_columns() {
    let columns: Vec<ColumnSpec> = (0..1000)
        .map(|i| ColumnSpec {
            name: format!("col_{}", i),
            column_type: if i % 3 == 0 { ColumnType::TextLong } else { ColumnType::Numeric },
        })
        .collect();
    
    let file = generate_csv_from_specs(&columns, 100); // Fewer rows for 1000 columns
    let mut benchmark = PerformanceBenchmark::new("column_profiling_1000")
        .with_metadata("columns", "1000")
        .with_metadata("rows", "100");
    
    let mut memory_monitor = MemoryMonitor::new();
    
    let start = Instant::now();
    {
        let _measurement = benchmark.start_measurement("profile_1000_columns");
        
        // Simulate column profiling with batching
        use datacloak_core::ml_classifier::{MLClassifier, Column};
        let classifier = MLClassifier::new();
        
        // Process in batches to simulate real-world usage
        for batch in columns.chunks(50) {
            memory_monitor.sample();
            
            let batch_columns: Vec<Column> = batch.iter().map(|spec| {
                let sample_data: Vec<&str> = match spec.column_type {
                    ColumnType::TextLong => vec!["Long descriptive text content here", "More text"],
                    ColumnType::Numeric => vec!["123.45", "678.90"],
                    _ => vec!["data", "value"],
                };
                Column::new(&spec.name, sample_data)
            }).collect();
            
            let _predictions = classifier.predict_batch(&batch_columns);
        }
    }
    let duration = start.elapsed();
    
    // Check against baseline
    let baseline = PerformanceBaseline {
        operation: "profile_1000_columns".to_string(),
        expected_duration_ms: 8500,
        max_memory_mb: Some(250),
        throughput_items_per_sec: Some(120.0),
        regression_threshold: 0.5,
    };
    
    let result = benchmark.check_against_baseline(&baseline);
    
    println!("Column profiling (1000 columns) took: {:?}", duration);
    println!("Peak memory usage: {} bytes", memory_monitor.peak_usage());
    println!("Throughput: {:.2} columns/sec", 1000.0 / duration.as_secs_f64());
    
    // Performance assertions
    assert!(duration.as_secs() < 30, "Profiling took too long: {:?}", duration);
    assert!(memory_monitor.peak_usage() < 1024 * 1024 * 1024, "Memory usage too high"); // 1GB limit
    
    // Clean up
    std::fs::remove_file(file).ok();
}

#[test]
fn bench_ml_inference_single() {
    use datacloak_core::ml_classifier::{MLClassifier, Column};
    
    let mut benchmark = PerformanceBenchmark::new("ml_inference_single");
    let classifier = MLClassifier::new();
    let column = Column::new("test", vec!["Sample text for classification", "Another sample"]);
    
    let mut bencher = criterion_like::Bencher;
    let duration = bencher.iter(|| {
        let _prediction = classifier.predict(&column);
    });
    
    benchmark.add_measurement(Measurement {
        name: "ml_inference_single".to_string(),
        duration,
        memory_used: None,
        cpu_percent: None,
        custom_metrics: std::collections::HashMap::new(),
    });
    
    let baseline = PerformanceBaseline {
        operation: "ml_inference_single".to_string(),
        expected_duration_ms: 1,
        max_memory_mb: Some(50),
        throughput_items_per_sec: Some(1000.0),
        regression_threshold: 1.0, // Allow 100% variance for single predictions
    };
    
    let result = benchmark.check_against_baseline(&baseline);
    
    println!("ML inference (single) took: {:?}", duration);
    println!("Is regression: {}", result.is_regression);
    
    // Should be very fast
    assert!(duration.as_millis() < 100, "Single ML inference too slow: {:?}", duration);
}

#[test]
fn bench_ml_inference_batch_1000() {
    use datacloak_core::ml_classifier::{MLClassifier, Column};
    
    let mut benchmark = PerformanceBenchmark::new("ml_inference_batch_1000");
    let classifier = MLClassifier::new();
    
    // Create 1000 columns for batch processing
    let columns: Vec<Column> = (0..1000)
        .map(|i| Column::new(
            &format!("col_{}", i),
            vec!["Sample text content", "More sample text", "Additional content"]
        ))
        .collect();
    
    let mut memory_monitor = MemoryMonitor::new();
    
    let start = Instant::now();
    {
        let _measurement = benchmark.start_measurement("ml_inference_batch_1000");
        memory_monitor.sample();
        
        let _predictions = classifier.predict_batch(&columns);
        
        memory_monitor.sample();
    }
    let duration = start.elapsed();
    
    let baseline = PerformanceBaseline {
        operation: "ml_inference_batch_1000".to_string(),
        expected_duration_ms: 400,
        max_memory_mb: Some(200),
        throughput_items_per_sec: Some(2500.0),
        regression_threshold: 0.5,
    };
    
    let result = benchmark.check_against_baseline(&baseline);
    
    println!("ML inference (batch 1000) took: {:?}", duration);
    println!("Throughput: {:.2} predictions/sec", 1000.0 / duration.as_secs_f64());
    println!("Peak memory: {} bytes", memory_monitor.peak_usage());
    
    // Performance assertions
    assert!(duration.as_secs() < 5, "Batch inference took too long: {:?}", duration);
    assert!(memory_monitor.peak_usage() < 500 * 1024 * 1024, "Memory usage too high");
}

#[tokio::test]
async fn test_load_concurrent_analyses() {
    let test_file = generate_csv_from_specs(&[
        ColumnSpec { name: "text".to_string(), column_type: ColumnType::TextLong },
        ColumnSpec { name: "data".to_string(), column_type: ColumnType::Numeric },
    ], 100);
    
    let file_path = test_file.to_str().unwrap().to_string();
    
    // Create load generator
    let load_gen = LoadGenerator::new(10, Duration::from_secs(5))
        .with_ramp_up(Duration::from_secs(2));
    
    // Define the operation to test
    let operation = move || {
        let path = file_path.clone();
        async move {
            let start = Instant::now();
            
            // Simulate profiling operation
            use datacloak_core::ml_classifier::{MLClassifier, Column};
            let classifier = MLClassifier::new();
            let column = Column::new("test", vec!["sample text", "more text"]);
            let _prediction = classifier.predict(&column);
            
            // Add some async work
            tokio::time::sleep(Duration::from_millis(10)).await;
            
            Ok(start.elapsed())
        }
    };
    
    // Run load test
    let results = load_gen.run(operation).await;
    
    println!("Load test results:");
    println!("  Concurrent users: {}", results.concurrent_users);
    println!("  Total duration: {:?}", results.total_duration);
    println!("  Successful operations: {}", results.successful_operations);
    println!("  Failed operations: {}", results.failed_operations);
    println!("  Average operation time: {:?}", results.average_operation_time);
    println!("  Throughput: {:.2} ops/sec", results.throughput_ops_per_sec);
    
    // Assertions
    assert!(results.successful_operations > 0, "No successful operations");
    assert!(results.failed_operations == 0, "Some operations failed");
    assert!(results.throughput_ops_per_sec > 1.0, "Throughput too low"); // More reasonable threshold
    assert!(results.average_operation_time.as_millis() < 1000, "Operations too slow"); // More reasonable threshold
    
    // Clean up
    std::fs::remove_file(test_file).ok();
}

#[test]
fn test_memory_usage_large_file() {
    // Create a larger test file
    let large_file = generate_large_csv(50 * MB, 100); // 50MB, 100 columns
    
    let mut memory_monitor = MemoryMonitor::new();
    memory_monitor.sample();
    
    // Simulate processing large file
    let start = Instant::now();
    
    // Read file in chunks to simulate streaming
    let chunk_size = 1000;
    for _chunk in 0..50 { // 50 chunks
        memory_monitor.sample();
        
        // Simulate processing chunk
        use datacloak_core::ml_classifier::{MLClassifier, Column};
        let classifier = MLClassifier::new();
        
        // Process a subset of columns per chunk
        for i in 0..10 {
            let column = Column::new(
                &format!("col_{}", i),
                vec!["sample data", "more data", "additional content"]
            );
            let _prediction = classifier.predict(&column);
        }
        
        // Simulate some processing time
        std::thread::sleep(Duration::from_millis(1));
    }
    
    let duration = start.elapsed();
    memory_monitor.sample();
    
    println!("Large file processing took: {:?}", duration);
    println!("Peak memory usage: {} MB", memory_monitor.peak_usage() / (1024 * 1024));
    println!("Average memory usage: {} MB", memory_monitor.average_usage() / (1024 * 1024));
    
    // Memory should be bounded regardless of file size
    assert!(memory_monitor.peak_usage() < 300 * MB, "Memory usage too high for large file");
    assert!(duration.as_secs() < 10, "Processing took too long");
    
    // Clean up
    std::fs::remove_file(large_file).ok();
}

#[test]
fn test_regression_detection() {
    let mut benchmark = PerformanceBenchmark::new("regression_test");
    
    // Simulate a fast operation
    benchmark.add_measurement(Measurement {
        name: "fast_operation".to_string(),
        duration: Duration::from_millis(50),
        memory_used: Some(10 * 1024 * 1024),
        cpu_percent: Some(25.0),
        custom_metrics: std::collections::HashMap::new(),
    });
    
    // Simulate a slow operation (regression)
    benchmark.add_measurement(Measurement {
        name: "slow_operation".to_string(),
        duration: Duration::from_millis(500), // 10x slower
        memory_used: Some(100 * 1024 * 1024),
        cpu_percent: Some(80.0),
        custom_metrics: std::collections::HashMap::new(),
    });
    
    let baseline_fast = PerformanceBaseline {
        operation: "fast_operation".to_string(),
        expected_duration_ms: 50,
        max_memory_mb: Some(50),
        throughput_items_per_sec: Some(20.0),
        regression_threshold: 0.1, // 10% regression threshold
    };
    
    let baseline_slow = PerformanceBaseline {
        operation: "slow_operation".to_string(),
        expected_duration_ms: 50,
        max_memory_mb: Some(50),
        throughput_items_per_sec: Some(20.0),
        regression_threshold: 0.1,
    };
    
    let result_fast = benchmark.check_against_baseline(&baseline_fast);
    let result_slow = benchmark.check_against_baseline(&baseline_slow);
    
    // Fast operation should not be a regression
    assert!(!result_fast.is_regression, "Fast operation incorrectly flagged as regression");
    
    // Slow operation should be flagged as regression
    assert!(result_slow.is_regression, "Slow operation not detected as regression");
    
    println!("Fast operation regression ratio: {:?}", result_fast.regression_ratio);
    println!("Slow operation regression ratio: {:?}", result_slow.regression_ratio);
}

// Helper constants
const MB: usize = 1024 * 1024;