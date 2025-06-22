use std::time::{Duration, Instant};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use anyhow::Result;

/// Performance benchmarking utilities
pub struct PerformanceBenchmark {
    pub name: String,
    pub start_time: Instant,
    pub measurements: Vec<Measurement>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Measurement {
    pub name: String,
    pub duration: Duration,
    pub memory_used: Option<usize>,
    pub cpu_percent: Option<f64>,
    pub custom_metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    pub operation: String,
    pub expected_duration_ms: u64,
    pub max_memory_mb: Option<usize>,
    pub throughput_items_per_sec: Option<f64>,
    pub regression_threshold: f64, // 0.1 = 10% regression allowed
}

impl PerformanceBenchmark {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            start_time: Instant::now(),
            measurements: Vec::new(),
            metadata: HashMap::new(),
        }
    }
    
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
    
    pub fn start_measurement(&mut self, name: &str) -> MeasurementHandle {
        MeasurementHandle {
            name: name.to_string(),
            start_time: Instant::now(),
            benchmark: self,
        }
    }
    
    pub fn add_measurement(&mut self, measurement: Measurement) {
        self.measurements.push(measurement);
    }
    
    pub fn total_duration(&self) -> Duration {
        self.start_time.elapsed()
    }
    
    pub fn get_measurement(&self, name: &str) -> Option<&Measurement> {
        self.measurements.iter().find(|m| m.name == name)
    }
    
    pub fn average_duration(&self, operation: &str) -> Option<Duration> {
        let durations: Vec<Duration> = self.measurements.iter()
            .filter(|m| m.name == operation)
            .map(|m| m.duration)
            .collect();
        
        if durations.is_empty() {
            None
        } else {
            let total_nanos: u64 = durations.iter().map(|d| d.as_nanos() as u64).sum();
            Some(Duration::from_nanos(total_nanos / durations.len() as u64))
        }
    }
    
    pub fn check_against_baseline(&self, baseline: &PerformanceBaseline) -> PerformanceResult {
        let avg_duration = self.average_duration(&baseline.operation);
        let is_regression = if let Some(duration) = avg_duration {
            let actual_ms = duration.as_millis() as u64;
            let expected_ms = baseline.expected_duration_ms;
            let regression_ratio = (actual_ms as f64) / (expected_ms as f64);
            regression_ratio > (1.0 + baseline.regression_threshold)
        } else {
            false
        };
        
        PerformanceResult {
            benchmark_name: self.name.clone(),
            baseline: baseline.clone(),
            actual_duration: avg_duration,
            is_regression,
            regression_ratio: avg_duration.map(|d| {
                (d.as_millis() as f64) / (baseline.expected_duration_ms as f64)
            }),
        }
    }
}

pub struct MeasurementHandle<'a> {
    name: String,
    start_time: Instant,
    benchmark: &'a mut PerformanceBenchmark,
}

impl<'a> Drop for MeasurementHandle<'a> {
    fn drop(&mut self) {
        let measurement = Measurement {
            name: self.name.clone(),
            duration: self.start_time.elapsed(),
            memory_used: get_current_memory_usage(),
            cpu_percent: None, // Would need system APIs
            custom_metrics: HashMap::new(),
        };
        self.benchmark.add_measurement(measurement);
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceResult {
    pub benchmark_name: String,
    pub baseline: PerformanceBaseline,
    pub actual_duration: Option<Duration>,
    pub is_regression: bool,
    pub regression_ratio: Option<f64>,
}

/// Memory monitor for performance testing
pub struct MemoryMonitor {
    start_memory: usize,
    peak_memory: usize,
    samples: Vec<(Instant, usize)>,
}

impl MemoryMonitor {
    pub fn new() -> Self {
        let start_memory = get_current_memory_usage().unwrap_or(0);
        Self {
            start_memory,
            peak_memory: start_memory,
            samples: Vec::new(),
        }
    }
    
    pub fn sample(&mut self) {
        let current = get_current_memory_usage().unwrap_or(0);
        if current > self.peak_memory {
            self.peak_memory = current;
        }
        self.samples.push((Instant::now(), current));
    }
    
    pub fn peak_usage(&self) -> usize {
        self.peak_memory.saturating_sub(self.start_memory)
    }
    
    pub fn current_usage(&self) -> usize {
        get_current_memory_usage().unwrap_or(0).saturating_sub(self.start_memory)
    }
    
    pub fn average_usage(&self) -> usize {
        if self.samples.is_empty() {
            return 0;
        }
        
        let total: usize = self.samples.iter()
            .map(|(_, mem)| mem.saturating_sub(self.start_memory))
            .sum();
        total / self.samples.len()
    }
}

/// Load generator for performance testing
pub struct LoadGenerator {
    pub concurrent_users: usize,
    pub duration: Duration,
    pub ramp_up_time: Duration,
}

impl LoadGenerator {
    pub fn new(concurrent_users: usize, duration: Duration) -> Self {
        Self {
            concurrent_users,
            duration,
            ramp_up_time: Duration::from_secs(10),
        }
    }
    
    pub fn with_ramp_up(mut self, ramp_up: Duration) -> Self {
        self.ramp_up_time = ramp_up;
        self
    }
    
    pub async fn run<F, Fut>(&self, operation: F) -> LoadTestResults
    where
        F: Fn() -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future<Output = Result<Duration>> + Send,
    {
        let mut handles = Vec::new();
        let start_time = Instant::now();
        
        // Gradual ramp-up
        for i in 0..self.concurrent_users {
            let delay = self.ramp_up_time.as_millis() as u64 * i as u64 / self.concurrent_users as u64;
            let op = operation.clone();
            
            let handle = tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(delay)).await;
                op().await
            });
            handles.push(handle);
        }
        
        // Collect results
        let mut successful_operations = 0;
        let mut failed_operations = 0;
        let mut operation_times = Vec::new();
        
        for handle in handles {
            match handle.await {
                Ok(Ok(duration)) => {
                    successful_operations += 1;
                    operation_times.push(duration);
                }
                _ => {
                    failed_operations += 1;
                }
            }
        }
        
        let total_duration = start_time.elapsed();
        let average_time = if !operation_times.is_empty() {
            operation_times.iter().sum::<Duration>() / operation_times.len() as u32
        } else {
            Duration::ZERO
        };
        
        LoadTestResults {
            concurrent_users: self.concurrent_users,
            total_duration,
            successful_operations,
            failed_operations,
            average_operation_time: average_time,
            throughput_ops_per_sec: successful_operations as f64 / total_duration.as_secs_f64(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoadTestResults {
    pub concurrent_users: usize,
    pub total_duration: Duration,
    pub successful_operations: usize,
    pub failed_operations: usize,
    pub average_operation_time: Duration,
    pub throughput_ops_per_sec: f64,
}

/// Performance baselines for regression testing
pub struct PerformanceBaselines;

impl PerformanceBaselines {
    pub fn column_profiling() -> Vec<PerformanceBaseline> {
        vec![
            PerformanceBaseline {
                operation: "profile_100_columns".to_string(),
                expected_duration_ms: 500,
                max_memory_mb: Some(100),
                throughput_items_per_sec: Some(200.0),
                regression_threshold: 0.2, // 20% regression allowed
            },
            PerformanceBaseline {
                operation: "profile_1000_columns".to_string(),
                expected_duration_ms: 8500,
                max_memory_mb: Some(250),
                throughput_items_per_sec: Some(120.0),
                regression_threshold: 0.2,
            },
            PerformanceBaseline {
                operation: "profile_10000_columns".to_string(),
                expected_duration_ms: 95000,
                max_memory_mb: Some(500),
                throughput_items_per_sec: Some(100.0),
                regression_threshold: 0.3, // Allow more variance for large datasets
            },
        ]
    }
    
    pub fn ml_inference() -> Vec<PerformanceBaseline> {
        vec![
            PerformanceBaseline {
                operation: "ml_inference_single".to_string(),
                expected_duration_ms: 1, // Should be sub-millisecond
                max_memory_mb: Some(50),
                throughput_items_per_sec: Some(1000.0),
                regression_threshold: 0.5, // ML can vary more
            },
            PerformanceBaseline {
                operation: "ml_inference_batch_100".to_string(),
                expected_duration_ms: 50,
                max_memory_mb: Some(100),
                throughput_items_per_sec: Some(2000.0),
                regression_threshold: 0.3,
            },
            PerformanceBaseline {
                operation: "ml_inference_batch_1000".to_string(),
                expected_duration_ms: 400,
                max_memory_mb: Some(200),
                throughput_items_per_sec: Some(2500.0),
                regression_threshold: 0.3,
            },
        ]
    }
    
    pub fn streaming_throughput() -> Vec<PerformanceBaseline> {
        vec![
            PerformanceBaseline {
                operation: "stream_single_column".to_string(),
                expected_duration_ms: 1000, // 1 second for 150MB
                max_memory_mb: Some(800),
                throughput_items_per_sec: Some(150.0), // 150MB/s
                regression_threshold: 0.2,
            },
            PerformanceBaseline {
                operation: "stream_10_columns".to_string(),
                expected_duration_ms: 1250, // Slightly slower with more columns
                max_memory_mb: Some(1000),
                throughput_items_per_sec: Some(120.0), // 120MB/s
                regression_threshold: 0.2,
            },
        ]
    }
}

fn get_current_memory_usage() -> Option<usize> {
    // Simplified memory usage - in real implementation would use system APIs
    // For now, return a mock value
    Some(100 * 1024 * 1024) // 100MB
}

/// Macro for easy performance testing
#[macro_export]
macro_rules! benchmark {
    ($name:expr, $code:block) => {{
        let start = std::time::Instant::now();
        let result = $code;
        let duration = start.elapsed();
        println!("Benchmark '{}' took {:?}", $name, duration);
        (result, duration)
    }};
}

pub use benchmark;