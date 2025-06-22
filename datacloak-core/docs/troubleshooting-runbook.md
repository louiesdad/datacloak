# DataCloak ML Troubleshooting Runbook

## Overview

This runbook provides comprehensive troubleshooting guidance for DataCloak's ML-based column classification system. It covers common issues, diagnostic procedures, and resolution strategies for production deployments.

## Quick Diagnostic Checklist

When encountering issues, run through this checklist first:

```bash
# 1. Check system health
cargo test --test ml_pipeline_integration_tests
cargo bench --bench ml_graph_benchmark -- --test

# 2. Verify model files
ls -la models/
file models/*.onnx

# 3. Check dependencies
cargo tree | grep -E "(ort|candle|ndarray)"

# 4. Monitor resource usage
top -p $(pgrep datacloak)
free -m

# 5. Check logs
tail -f /var/log/datacloak/ml_classifier.log
```

## Common Issues and Solutions

### 1. Model Loading Issues

#### Issue: "Failed to load ONNX model"

**Symptoms:**
- Error message: `"Failed to load ONNX model from path"`
- Classification falls back to rule-based methods
- Reduced accuracy compared to expected performance

**Diagnostic Steps:**

```rust
// Test model loading directly
use crate::onnx_model::OnnxModel;

pub fn diagnose_model_loading(model_path: &str) {
    println!("Diagnosing model loading for: {}", model_path);
    
    // Check file existence
    if !std::path::Path::new(model_path).exists() {
        println!("❌ Model file does not exist: {}", model_path);
        return;
    }
    
    // Check file permissions
    match std::fs::metadata(model_path) {
        Ok(metadata) => {
            println!("✅ Model file exists, size: {} bytes", metadata.len());
            if metadata.len() == 0 {
                println!("❌ Model file is empty");
                return;
            }
        }
        Err(e) => {
            println!("❌ Cannot access model file: {}", e);
            return;
        }
    }
    
    // Test ONNX loading
    match OnnxModel::load(model_path) {
        Ok(_) => println!("✅ Model loaded successfully"),
        Err(e) => {
            println!("❌ Model loading failed: {}", e);
            
            // Additional diagnostics
            if e.to_string().contains("version") {
                println!("💡 Suggestion: Check ONNX opset version compatibility");
            }
            if e.to_string().contains("provider") {
                println!("💡 Suggestion: Check ONNX Runtime execution providers");
            }
        }
    }
}
```

**Common Resolutions:**

1. **Missing Model File:**
   ```bash
   # Download the correct model
   wget https://releases.datacloak.com/models/column_classifier_v1.2.0.onnx
   mv column_classifier_v1.2.0.onnx models/
   ```

2. **Incorrect ONNX Runtime Version:**
   ```toml
   # Update Cargo.toml
   [dependencies]
   ort = "2.0.0-rc.2"  # Use compatible version
   ```

3. **Permission Issues:**
   ```bash
   chmod 644 models/*.onnx
   chown datacloak:datacloak models/*.onnx
   ```

4. **Corrupted Model File:**
   ```bash
   # Verify model integrity
   onnx-simplifier --check models/column_classifier.onnx
   
   # Re-download if corrupted
   curl -L https://releases.datacloak.com/models/column_classifier_v1.2.0.onnx \
        -o models/column_classifier.onnx
   ```

#### Issue: "ONNX Runtime provider not available"

**Symptoms:**
- Error about CUDA or other execution providers
- Model loads but uses CPU instead of GPU
- Slower than expected inference

**Resolution:**
```rust
// Configure ONNX Runtime with fallback providers
use ort::{SessionBuilder, ExecutionProvider};

pub fn create_robust_session(model_path: &str) -> Result<Session, Box<dyn std::error::Error>> {
    let mut builder = SessionBuilder::new()?;
    
    // Try providers in order of preference
    let providers = vec![
        ExecutionProvider::cuda(), // Try CUDA first
        ExecutionProvider::cpu(),  // Fallback to CPU
    ];
    
    for provider in providers {
        if let Ok(session) = builder.clone().with_execution_providers([provider])?.commit_from_file(model_path) {
            println!("✅ Using execution provider: {:?}", provider);
            return Ok(session);
        }
    }
    
    // Final fallback - basic CPU
    builder.commit_from_file(model_path)
}
```

### 2. Performance Issues

#### Issue: Slow Classification Performance

**Symptoms:**
- Classification taking >50ms per column
- High CPU usage
- Memory usage constantly increasing

**Diagnostic Steps:**

```rust
use std::time::Instant;
use crate::ml_classifier::MLClassifier;

pub fn benchmark_classification_performance() {
    let classifier = MLClassifier::new();
    let test_columns = create_test_columns(100);
    
    // Single column performance
    let start = Instant::now();
    for column in &test_columns[..10] {
        let _ = classifier.predict(column);
    }
    let single_avg = start.elapsed().as_millis() / 10;
    println!("Average single column time: {}ms", single_avg);
    
    // Batch performance
    let start = Instant::now();
    let _ = classifier.predict_batch(&test_columns);
    let batch_total = start.elapsed().as_millis();
    let batch_avg = batch_total / test_columns.len() as u128;
    println!("Average batch column time: {}ms", batch_avg);
    
    // Performance analysis
    if single_avg > 10 {
        println!("⚠️  Single column performance is slow");
        println!("💡 Consider enabling model quantization");
    }
    
    if batch_avg > single_avg * 2 {
        println!("⚠️  Batch processing is inefficient");
        println!("💡 Check for threading or memory issues");
    }
}
```

**Performance Optimization Steps:**

1. **Enable Model Quantization:**
   ```rust
   use crate::model_optimization::QuantizationLevel;
   
   let classifier = MLClassifier::with_quantized_model(
       "models/column_classifier.onnx",
       QuantizationLevel::Int8
   )?;
   ```

2. **Optimize Batch Size:**
   ```rust
   // Find optimal batch size for your system
   pub fn find_optimal_batch_size() -> usize {
       let classifier = MLClassifier::new();
       let test_data = create_test_columns(1000);
       
       let batch_sizes = vec![8, 16, 32, 64, 128];
       let mut best_throughput = 0.0;
       let mut best_batch_size = 32;
       
       for &batch_size in &batch_sizes {
           let start = Instant::now();
           
           for chunk in test_data.chunks(batch_size) {
               let _ = classifier.predict_batch(chunk);
           }
           
           let throughput = test_data.len() as f64 / start.elapsed().as_secs_f64();
           
           if throughput > best_throughput {
               best_throughput = throughput;
               best_batch_size = batch_size;
           }
       }
       
       println!("Optimal batch size: {} (throughput: {:.1} cols/sec)", 
                best_batch_size, best_throughput);
       best_batch_size
   }
   ```

3. **Enable Feature Caching:**
   ```rust
   use crate::feature_extractor::FeatureCache;
   
   let cache = FeatureCache::new(10000); // Cache 10k feature vectors
   let extractor = FeatureExtractor::with_cache(cache);
   ```

#### Issue: High Memory Usage

**Symptoms:**
- Memory usage grows over time
- Out of memory errors
- System becomes unresponsive

**Memory Diagnostic Tools:**

```rust
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

// Memory usage tracker
pub struct MemoryTracker {
    allocated: AtomicUsize,
    peak_usage: AtomicUsize,
}

impl MemoryTracker {
    pub fn current_usage_mb(&self) -> f64 {
        self.allocated.load(Ordering::Relaxed) as f64 / 1024.0 / 1024.0
    }
    
    pub fn peak_usage_mb(&self) -> f64 {
        self.peak_usage.load(Ordering::Relaxed) as f64 / 1024.0 / 1024.0
    }
    
    pub fn check_memory_health(&self) -> MemoryHealth {
        let current = self.current_usage_mb();
        
        if current > 1000.0 {
            MemoryHealth::Critical
        } else if current > 500.0 {
            MemoryHealth::Warning
        } else {
            MemoryHealth::Good
        }
    }
}

#[derive(Debug)]
pub enum MemoryHealth {
    Good,
    Warning,
    Critical,
}

// Memory leak detection
pub fn detect_memory_leaks() {
    let classifier = MLClassifier::new();
    let test_column = create_test_column();
    let initial_memory = get_current_memory_usage();
    
    // Run many predictions
    for _ in 0..1000 {
        let _ = classifier.predict(&test_column);
    }
    
    let final_memory = get_current_memory_usage();
    let memory_increase = final_memory - initial_memory;
    
    if memory_increase > 50 * 1024 * 1024 { // 50MB increase
        println!("⚠️  Potential memory leak detected: {}MB increase", 
                 memory_increase / 1024 / 1024);
    } else {
        println!("✅ No significant memory increase detected");
    }
}
```

**Memory Optimization Solutions:**

1. **Enable Memory Pooling:**
   ```rust
   use crate::performance::memory_pool::MemoryPool;
   
   thread_local! {
       static MEMORY_POOL: RefCell<MemoryPool> = RefCell::new(
           MemoryPool::new(10 * 1024 * 1024, 376 * 4) // 10MB pool, 376 float features
       );
   }
   ```

2. **Implement Cache Eviction:**
   ```rust
   use lru::LruCache;
   
   pub struct BoundedFeatureCache {
       cache: LruCache<u64, Vec<f32>>,
       max_memory_mb: usize,
       current_memory: usize,
   }
   
   impl BoundedFeatureCache {
       pub fn insert(&mut self, key: u64, value: Vec<f32>) {
           let value_size = value.len() * std::mem::size_of::<f32>();
           
           // Evict if necessary
           while self.current_memory + value_size > self.max_memory_mb * 1024 * 1024 {
               if let Some((_, evicted)) = self.cache.pop_lru() {
                   self.current_memory -= evicted.len() * std::mem::size_of::<f32>();
               } else {
                   break;
               }
           }
           
           self.cache.put(key, value);
           self.current_memory += value_size;
       }
   }
   ```

### 3. Accuracy Issues

#### Issue: Lower Than Expected Classification Accuracy

**Symptoms:**
- Accuracy below 85% on known datasets
- Many text columns classified as non-text
- Inconsistent predictions for similar columns

**Accuracy Diagnostic Steps:**

```rust
use crate::ml_classifier::{MLClassifier, Column, ColumnType};

pub fn diagnose_accuracy_issues() {
    let classifier = MLClassifier::new();
    
    // Test with known samples
    let test_cases = vec![
        (Column::new("product_description", vec![
            "High-quality wireless headphones with excellent sound quality",
            "Professional-grade equipment for audio enthusiasts"
        ]), ColumnType::TextLong),
        
        (Column::new("price", vec!["19.99", "29.99", "39.99"]), ColumnType::Numeric),
        
        (Column::new("category", vec!["Electronics", "Clothing", "Books"]), ColumnType::Categorical),
    ];
    
    let mut correct = 0;
    let mut total = 0;
    
    for (column, expected_type) in test_cases {
        let prediction = classifier.predict(&column);
        let is_correct = match expected_type {
            ColumnType::TextLong | ColumnType::TextShort => {
                matches!(prediction.column_type, ColumnType::TextLong | ColumnType::TextShort)
            }
            _ => prediction.column_type == expected_type
        };
        
        if is_correct {
            correct += 1;
        } else {
            println!("❌ Misclassified '{}': predicted {:?}, expected {:?} (confidence: {:.3})",
                     column.name, prediction.column_type, expected_type, prediction.confidence);
        }
        total += 1;
    }
    
    let accuracy = correct as f64 / total as f64;
    println!("Test accuracy: {:.1}% ({}/{})", accuracy * 100.0, correct, total);
    
    if accuracy < 0.8 {
        println!("⚠️  Accuracy is below expected threshold");
        println!("💡 Consider retraining model or checking feature extraction");
    }
}
```

**Accuracy Improvement Steps:**

1. **Verify Feature Extraction:**
   ```rust
   pub fn validate_feature_extraction() {
       let extractor = FeatureExtractor::new();
       let test_column = Column::new("test", vec!["sample text", "another example"]);
       let features = extractor.extract_all_features(&test_column);
       
       println!("Feature vector length: {}", features.len());
       assert_eq!(features.len(), 376, "Feature vector length mismatch");
       
       // Check for NaN or infinite values
       let invalid_count = features.iter()
           .filter(|&&f| f.is_nan() || f.is_infinite())
           .count();
       
       if invalid_count > 0 {
           println!("⚠️  Found {} invalid feature values", invalid_count);
       } else {
           println!("✅ All features are valid numbers");
       }
       
       // Check feature range
       let min_val = features.iter().fold(f32::INFINITY, |a, &b| a.min(b));
       let max_val = features.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
       println!("Feature range: [{:.3}, {:.3}]", min_val, max_val);
   }
   ```

2. **Model Confidence Calibration:**
   ```rust
   pub fn analyze_prediction_confidence() {
       let classifier = MLClassifier::new();
       let test_data = load_validation_dataset();
       
       let mut confidence_buckets = vec![0; 10]; // 0.0-0.1, 0.1-0.2, etc.
       let mut correct_in_bucket = vec![0; 10];
       
       for (column, expected_type) in test_data {
           let prediction = classifier.predict(&column);
           let bucket = (prediction.confidence * 10.0).floor() as usize;
           let bucket = bucket.min(9);
           
           confidence_buckets[bucket] += 1;
           
           if prediction.column_type == expected_type {
               correct_in_bucket[bucket] += 1;
           }
       }
       
       println!("Confidence calibration:");
       for i in 0..10 {
           let range_start = i as f32 * 0.1;
           let range_end = (i + 1) as f32 * 0.1;
           let accuracy = if confidence_buckets[i] > 0 {
               correct_in_bucket[i] as f32 / confidence_buckets[i] as f32
           } else {
               0.0
           };
           
           println!("{:.1}-{:.1}: {} samples, {:.1}% accuracy", 
                    range_start, range_end, confidence_buckets[i], accuracy * 100.0);
       }
   }
   ```

### 4. Integration Issues

#### Issue: Graph Integration Errors

**Symptoms:**
- Errors in graph construction
- Inconsistent ranking results
- Crashes during similarity calculation

**Graph Diagnostic Steps:**

```rust
use crate::ml_graph_integration::MLGraphRanker;
use crate::graph::ColumnGraph;

pub fn diagnose_graph_integration() {
    let ranker = MLGraphRanker::new();
    let test_columns = vec![
        Column::new("col1", vec!["text data", "more text"]),
        Column::new("col2", vec!["123", "456"]),
        Column::new("col3", vec!["similar text", "also text"]),
    ];
    
    // Test graph construction
    match ranker.build_similarity_graph(&test_columns).await {
        Ok(graph) => {
            println!("✅ Graph construction successful");
            println!("   Nodes: {}", graph.node_count());
            println!("   Edges: {}", graph.edge_count());
            
            // Validate graph structure
            if graph.node_count() != test_columns.len() {
                println!("⚠️  Node count mismatch: expected {}, got {}", 
                         test_columns.len(), graph.node_count());
            }
            
            // Check for isolated nodes
            let isolated_nodes = graph.isolated_node_count();
            if isolated_nodes > 0 {
                println!("ℹ️  {} isolated nodes found", isolated_nodes);
            }
        }
        Err(e) => {
            println!("❌ Graph construction failed: {}", e);
            
            // Additional diagnostics
            if e.contains("similarity") {
                println!("💡 Suggestion: Check similarity calculation implementation");
            }
            if e.contains("memory") {
                println!("💡 Suggestion: Reduce batch size or enable memory pooling");
            }
        }
    }
    
    // Test ranking
    let candidates = ranker.rank_columns_with_graph(&test_columns, 0.7);
    println!("Ranking results: {} candidates", candidates.len());
    
    for (i, candidate) in candidates.iter().enumerate() {
        println!("  {}: {} (score: {:.3})", i + 1, candidate.name, candidate.final_score);
    }
}
```

**Graph Integration Fixes:**

1. **Handle Empty Graphs:**
   ```rust
   impl MLGraphRanker {
       pub fn safe_rank_columns(&self, columns: &[Column]) -> Vec<ColumnCandidate> {
           if columns.is_empty() {
               return Vec::new();
           }
           
           if columns.len() == 1 {
               // Single column - no graph needed
               let prediction = self.classifier.predict(&columns[0]);
               return vec![ColumnCandidate {
                   name: columns[0].name.clone(),
                   column_type: prediction.column_type,
                   confidence: prediction.confidence,
                   final_score: prediction.confidence,
               }];
           }
           
           // Normal graph-based ranking
           self.rank_columns_with_graph(columns, 0.7)
       }
   }
   ```

2. **Similarity Calculation Safeguards:**
   ```rust
   fn safe_calculate_similarity(&self, col1: &Column, col2: &Column) -> f32 {
       if col1.values.is_empty() || col2.values.is_empty() {
           return 0.0;
       }
       
       let features1 = self.feature_extractor.extract_all_features(col1);
       let features2 = self.feature_extractor.extract_all_features(col2);
       
       // Check for valid features
       if features1.iter().any(|f| f.is_nan() || f.is_infinite()) ||
          features2.iter().any(|f| f.is_nan() || f.is_infinite()) {
           return 0.0;
       }
       
       self.calculate_cosine_similarity(&features1, &features2)
   }
   ```

### 5. Dependency Issues

#### Issue: ONNX Runtime Compilation Errors

**Symptoms:**
- Build failures with ONNX-related errors
- Linker errors about missing symbols
- Version compatibility issues

**Resolution Steps:**

1. **Check Rust Version:**
   ```bash
   rustc --version
   # Should be 1.70.0 or later
   rustup update
   ```

2. **Update Dependencies:**
   ```bash
   cargo update
   cargo clean
   cargo build --features ml
   ```

3. **Platform-Specific Issues:**
   ```bash
   # macOS: Install required system libraries
   brew install libomp protobuf
   
   # Linux: Install ONNX Runtime dependencies
   sudo apt-get install libonnxruntime-dev
   
   # Windows: Use vcpkg
   vcpkg install onnxruntime
   ```

4. **Feature Flag Issues:**
   ```toml
   # Ensure correct feature flags in Cargo.toml
   [dependencies]
   ort = { version = "2.0.0-rc.2", features = ["half"] }
   ndarray = { version = "0.15", optional = true }
   
   [features]
   default = []
   ml = ["ort", "ndarray"]
   ```

## Monitoring and Alerting

### Health Check Endpoints

```rust
use serde_json::json;
use tokio::time::{timeout, Duration};

#[derive(Debug)]
pub struct HealthChecker {
    classifier: Arc<MLClassifier>,
    last_check: Instant,
    health_status: HealthStatus,
}

#[derive(Debug, Clone)]
pub enum HealthStatus {
    Healthy,
    Degraded(String),
    Unhealthy(String),
}

impl HealthChecker {
    pub async fn comprehensive_health_check(&mut self) -> HealthStatus {
        let mut issues = Vec::new();
        
        // Test 1: Model loading
        if let Err(e) = self.test_model_loading().await {
            issues.push(format!("Model loading: {}", e));
        }
        
        // Test 2: Feature extraction
        if let Err(e) = self.test_feature_extraction().await {
            issues.push(format!("Feature extraction: {}", e));
        }
        
        // Test 3: Classification performance
        if let Err(e) = self.test_classification_performance().await {
            issues.push(format!("Classification performance: {}", e));
        }
        
        // Test 4: Memory usage
        if let Err(e) = self.test_memory_usage().await {
            issues.push(format!("Memory usage: {}", e));
        }
        
        // Determine overall health
        let status = if issues.is_empty() {
            HealthStatus::Healthy
        } else if issues.len() <= 2 {
            HealthStatus::Degraded(issues.join("; "))
        } else {
            HealthStatus::Unhealthy(issues.join("; "))
        };
        
        self.health_status = status.clone();
        self.last_check = Instant::now();
        
        status
    }
    
    async fn test_model_loading(&self) -> Result<(), String> {
        timeout(Duration::from_secs(5), async {
            let test_column = Column::new("test", vec!["sample"]);
            self.classifier.predict(&test_column);
            Ok(())
        })
        .await
        .map_err(|_| "Model loading timeout".to_string())?
    }
    
    async fn test_classification_performance(&self) -> Result<(), String> {
        let test_columns = (0..10)
            .map(|i| Column::new(&format!("test_{}", i), vec!["sample text"]))
            .collect::<Vec<_>>();
        
        let start = Instant::now();
        
        timeout(Duration::from_secs(2), async {
            self.classifier.predict_batch(&test_columns);
            Ok(())
        })
        .await
        .map_err(|_| "Classification timeout".to_string())?;
        
        let elapsed = start.elapsed();
        if elapsed.as_millis() > 1000 {
            return Err(format!("Classification too slow: {}ms", elapsed.as_millis()));
        }
        
        Ok(())
    }
    
    pub fn to_json(&self) -> serde_json::Value {
        json!({
            "status": match &self.health_status {
                HealthStatus::Healthy => "healthy",
                HealthStatus::Degraded(_) => "degraded", 
                HealthStatus::Unhealthy(_) => "unhealthy"
            },
            "last_check": self.last_check.elapsed().as_secs(),
            "details": match &self.health_status {
                HealthStatus::Healthy => "All systems operational",
                HealthStatus::Degraded(msg) | HealthStatus::Unhealthy(msg) => msg
            }
        })
    }
}
```

### Alerting Configuration

```yaml
# alerts.yml
alerts:
  - name: "ml_classifier_high_latency"
    condition: "avg_classification_time_ms > 50"
    threshold: 50
    duration: "5m"
    severity: "warning"
    notification: ["slack", "email"]
    
  - name: "ml_classifier_accuracy_drop"
    condition: "classification_accuracy < 0.8"
    threshold: 0.8
    duration: "10m"
    severity: "critical"
    notification: ["pagerduty", "slack"]
    
  - name: "ml_classifier_memory_leak"
    condition: "memory_usage_mb > 1000"
    threshold: 1000
    duration: "15m"
    severity: "critical"
    notification: ["pagerduty"]
    
  - name: "ml_classifier_error_rate"
    condition: "error_rate_percent > 5"
    threshold: 5
    duration: "5m"
    severity: "warning"
    notification: ["slack"]
```

## Emergency Procedures

### 1. Complete System Failure

**Immediate Actions:**
1. Switch to fallback mode (rule-based classification only)
2. Restart ML service
3. Check system resources
4. Review recent changes

**Fallback Configuration:**
```rust
pub struct EmergencyClassifier {
    rule_based_only: bool,
}

impl EmergencyClassifier {
    pub fn emergency_mode() -> Self {
        Self { rule_based_only: true }
    }
    
    pub fn predict(&self, column: &Column) -> Prediction {
        if self.rule_based_only {
            self.rule_based_predict(column)
        } else {
            // Normal ML prediction
            self.ml_predict(column)
        }
    }
    
    fn rule_based_predict(&self, column: &Column) -> Prediction {
        // Simplified rule-based classification
        let avg_length = column.values.iter()
            .map(|v| v.len())
            .sum::<usize>() as f32 / column.values.len() as f32;
        
        let numeric_ratio = column.values.iter()
            .filter(|v| v.parse::<f64>().is_ok())
            .count() as f32 / column.values.len() as f32;
        
        if numeric_ratio > 0.8 {
            Prediction {
                column_type: ColumnType::Numeric,
                confidence: numeric_ratio,
            }
        } else if avg_length > 20.0 {
            Prediction {
                column_type: ColumnType::TextLong,
                confidence: 0.7,
            }
        } else {
            Prediction {
                column_type: ColumnType::TextShort,
                confidence: 0.6,
            }
        }
    }
}
```

### 2. Memory Exhaustion

**Immediate Actions:**
1. Enable memory-bounded caching
2. Reduce batch sizes
3. Force garbage collection
4. Restart with memory limits

**Memory Recovery Script:**
```bash
#!/bin/bash
# emergency_memory_recovery.sh

echo "Starting emergency memory recovery..."

# Kill high-memory processes
pkill -f "datacloak.*high-memory"

# Clear system caches
echo 3 > /proc/sys/vm/drop_caches

# Restart with memory limits
systemctl stop datacloak-ml
sleep 5

# Set memory limits
echo "512M" > /sys/fs/cgroup/memory/datacloak/memory.limit_in_bytes

systemctl start datacloak-ml

echo "Emergency recovery completed"
```

### 3. Model Corruption

**Detection:**
```rust
pub fn detect_model_corruption(model_path: &str) -> bool {
    // Quick corruption check
    if let Ok(metadata) = std::fs::metadata(model_path) {
        if metadata.len() == 0 {
            return true; // Empty file
        }
    }
    
    // Try loading model
    match OnnxModel::load(model_path) {
        Ok(model) => {
            // Test with known input
            let test_features = vec![0.5f32; 376];
            match model.predict(&test_features) {
                Ok(output) => {
                    // Check for valid output
                    !output.iter().any(|&x| x.is_nan() || x.is_infinite())
                }
                Err(_) => true, // Prediction failed - likely corrupted
            }
        }
        Err(_) => true, // Loading failed - corrupted
    }
}
```

**Recovery:**
```bash
#!/bin/bash
# recover_corrupted_model.sh

MODEL_PATH="models/column_classifier.onnx"
BACKUP_URL="https://releases.datacloak.com/models/column_classifier_v1.2.0.onnx"

echo "Detecting model corruption..."

if cargo run --bin detect_corruption "$MODEL_PATH"; then
    echo "Model corruption detected - recovering..."
    
    # Backup corrupted model
    mv "$MODEL_PATH" "${MODEL_PATH}.corrupted.$(date +%s)"
    
    # Download fresh model
    curl -L "$BACKUP_URL" -o "$MODEL_PATH"
    
    # Verify recovery
    if cargo run --bin validate_model "$MODEL_PATH"; then
        echo "Model recovery successful"
        systemctl restart datacloak-ml
    else
        echo "Model recovery failed - manual intervention required"
        exit 1
    fi
else
    echo "No corruption detected"
fi
```

## Log Analysis

### Important Log Patterns

1. **Normal Operation:**
   ```
   INFO [ml_classifier] Model loaded successfully: models/column_classifier.onnx
   INFO [feature_extractor] Extracted 376 features for column 'product_description'
   INFO [ml_classifier] Classified 'product_description' as TextLong (confidence: 0.94)
   ```

2. **Performance Issues:**
   ```
   WARN [ml_classifier] Classification took 87ms (threshold: 50ms)
   WARN [feature_cache] Cache hit rate below 50%: 0.23
   ERROR [memory_pool] Failed to allocate feature vector: out of memory
   ```

3. **Model Issues:**
   ```
   ERROR [onnx_model] Failed to load model: Invalid ONNX file format
   WARN [ml_classifier] Falling back to rule-based classification
   ERROR [model_cache] Model validation failed: output shape mismatch
   ```

### Log Analysis Tools

```bash
# Performance analysis
grep "Classification took" /var/log/datacloak/ml_classifier.log | \
  awk '{print $6}' | sed 's/ms//' | \
  awk '{sum+=$1; n++} END {print "Average:", sum/n "ms"}'

# Error analysis
grep -E "(ERROR|WARN)" /var/log/datacloak/ml_classifier.log | \
  cut -d' ' -f3- | sort | uniq -c | sort -nr

# Memory usage tracking
grep "memory_usage_mb" /var/log/datacloak/metrics.log | \
  awk '{print $2, $4}' | \
  gnuplot -e "plot '-' using 1:2 with lines"
```

## Preventive Measures

### 1. Regular Health Checks

```bash
#!/bin/bash
# daily_health_check.sh

LOG_FILE="/var/log/datacloak/health_check.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

echo "[$DATE] Starting daily health check..." >> "$LOG_FILE"

# Test model loading
if timeout 30 cargo run --bin test_model_loading; then
    echo "[$DATE] ✅ Model loading: OK" >> "$LOG_FILE"
else
    echo "[$DATE] ❌ Model loading: FAILED" >> "$LOG_FILE"
    echo "Model loading failure detected" | mail -s "DataCloak Alert" admin@company.com
fi

# Test classification performance
PERF_RESULT=$(cargo run --bin benchmark_classification | grep "Average time" | awk '{print $3}')
if (( $(echo "$PERF_RESULT < 50" | bc -l) )); then
    echo "[$DATE] ✅ Performance: ${PERF_RESULT}ms" >> "$LOG_FILE"
else
    echo "[$DATE] ⚠️  Performance: ${PERF_RESULT}ms (above 50ms threshold)" >> "$LOG_FILE"
fi

# Test memory usage
MEMORY_MB=$(ps aux | grep datacloak | awk '{sum+=$6} END {print sum/1024}')
if (( $(echo "$MEMORY_MB < 500" | bc -l) )); then
    echo "[$DATE] ✅ Memory usage: ${MEMORY_MB}MB" >> "$LOG_FILE"
else
    echo "[$DATE] ⚠️  Memory usage: ${MEMORY_MB}MB (high)" >> "$LOG_FILE"
fi

echo "[$DATE] Health check completed" >> "$LOG_FILE"
```

### 2. Automated Recovery

```rust
use tokio::time::{interval, Duration};
use std::sync::Arc;

pub struct AutoRecoveryService {
    health_checker: Arc<HealthChecker>,
    recovery_attempts: usize,
    max_attempts: usize,
}

impl AutoRecoveryService {
    pub async fn start_monitoring(&mut self) {
        let mut check_interval = interval(Duration::from_secs(60)); // Check every minute
        
        loop {
            check_interval.tick().await;
            
            match self.health_checker.comprehensive_health_check().await {
                HealthStatus::Healthy => {
                    self.recovery_attempts = 0; // Reset counter on successful check
                }
                HealthStatus::Degraded(issue) => {
                    println!("⚠️  System degraded: {}", issue);
                    self.attempt_recovery().await;
                }
                HealthStatus::Unhealthy(issue) => {
                    println!("❌ System unhealthy: {}", issue);
                    self.attempt_emergency_recovery().await;
                }
            }
        }
    }
    
    async fn attempt_recovery(&mut self) {
        if self.recovery_attempts >= self.max_attempts {
            println!("Max recovery attempts reached, escalating...");
            self.escalate_to_human().await;
            return;
        }
        
        self.recovery_attempts += 1;
        println!("Attempting recovery {}/{}", self.recovery_attempts, self.max_attempts);
        
        // Try various recovery strategies
        self.clear_caches().await;
        self.restart_components().await;
        self.reload_configuration().await;
    }
    
    async fn attempt_emergency_recovery(&mut self) {
        println!("Initiating emergency recovery procedures...");
        
        // Switch to emergency mode
        self.enable_emergency_mode().await;
        
        // Alert operations team
        self.send_emergency_alert().await;
        
        // Attempt full restart
        self.full_system_restart().await;
    }
}
```

## Support Contacts

### Escalation Path

1. **Level 1 - Automated Recovery**
   - Auto-restart services
   - Cache clearing
   - Configuration reload

2. **Level 2 - Operations Team**
   - Manual diagnostics
   - System resource investigation
   - Log analysis

3. **Level 3 - ML Engineering Team**
   - Model validation
   - Feature extraction debugging
   - Performance optimization

4. **Level 4 - Core Development Team**
   - Code-level debugging
   - Architecture changes
   - Emergency patches

### Emergency Contacts

```yaml
contacts:
  operations:
    primary: "ops-team@company.com"
    phone: "+1-555-OPS-TEAM"
    slack: "#ops-alerts"
    
  ml_engineering:
    primary: "ml-team@company.com" 
    phone: "+1-555-ML-TEAM"
    slack: "#ml-alerts"
    
  development:
    primary: "dev-team@company.com"
    phone: "+1-555-DEV-TEAM"
    slack: "#dev-alerts"
    
  management:
    escalation: "cto@company.com"
    phone: "+1-555-CTO-PHONE"
```

## Conclusion

This troubleshooting runbook provides comprehensive guidance for diagnosing and resolving issues with DataCloak's ML classification system. Regular use of the diagnostic tools and preventive measures will help maintain system health and minimize downtime.

For issues not covered in this runbook, collect relevant logs, performance metrics, and system information before contacting the appropriate support team.

## References

- [Rust Error Handling](https://doc.rust-lang.org/book/ch09-00-error-handling.html)
- [ONNX Runtime Troubleshooting](https://onnxruntime.ai/docs/troubleshooting/)
- [System Performance Analysis](https://brendangregg.com/systems-performance-2nd-edition-book.html)
- [Production ML Systems](https://developers.google.com/machine-learning/guides/rules-of-ml)