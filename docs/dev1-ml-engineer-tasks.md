# Developer 1: ML/AI Engineer Tasks

## Role Overview
Lead the implementation of the ML-based column profiling system that automatically identifies text-heavy columns suitable for sentiment analysis.

## TDD Learning Requirements
Before starting, master these TDD concepts from the reference guide:
1. **Red-Green-Refactor cycle**: Always write failing tests first
2. **Triangulation pattern**: Use multiple test cases to drive general solutions
3. **Fake It pattern**: Start with hardcoded values, then generalize
4. **Test behavior, not implementation**: Focus on what the code does, not how

## Sprint 1 Tasks (Days 1-5)

### Task 1.1: ML Classifier Foundation
**Story**: As a system, I need an ML classifier that can predict column types based on statistical features.

**TDD Approach**:
```rust
// Start with simplest test
#[test]
fn test_classifier_identifies_text_column() {
    let classifier = MLClassifier::new();
    let column = Column::new("description", vec!["This is a long text", "Another paragraph"]);
    let prediction = classifier.predict(&column);
    assert_eq!(prediction.column_type, ColumnType::TextLong);
}

// Then test numeric column
#[test]
fn test_classifier_identifies_numeric_column() {
    let classifier = MLClassifier::new();
    let column = Column::new("price", vec!["10.99", "25.50", "100.00"]);
    let prediction = classifier.predict(&column);
    assert_eq!(prediction.column_type, ColumnType::Numeric);
}

// Triangulate with edge cases
#[test]
fn test_classifier_handles_mixed_content() {
    let classifier = MLClassifier::new();
    let column = Column::new("mixed", vec!["Text123", "456Text", "789"]);
    let prediction = classifier.predict(&column);
    assert!(prediction.confidence < 0.8); // Low confidence for mixed
}
```

**Implementation Steps**:
1. Create `MLClassifier` struct with ONNX model loading
2. Implement basic `predict` method (fake it first - return hardcoded)
3. Add feature extraction (mean length, digit ratio)
4. Integrate real ONNX model
5. Refactor for batch processing

**Deliverables**:
- `ml_classifier.rs` with 95%+ test coverage
- Performance benchmark: <1ms per column prediction
- Model file in `models/column_classifier.onnx`

### Task 1.2: Feature Extraction Pipeline
**Story**: As an ML classifier, I need rich features extracted from columns to make accurate predictions.

**TDD Tests First**:
```rust
#[test]
fn test_extract_statistical_features() {
    let extractor = FeatureExtractor::new();
    let column = Column::new("test", vec!["hello", "world", "foo"]);
    let features = extractor.extract_stats(&column);
    
    assert_eq!(features.mean_length, 4.33, 0.01);
    assert_eq!(features.min_length, 3);
    assert_eq!(features.max_length, 5);
}

#[test]
fn test_extract_character_ratios() {
    let extractor = FeatureExtractor::new();
    let column = Column::new("test", vec!["abc123", "def456"]);
    let features = extractor.extract_char_ratios(&column);
    
    assert_eq!(features.digit_ratio, 0.5);
    assert_eq!(features.alpha_ratio, 0.5);
}

#[test]
fn test_extract_embeddings() {
    let extractor = FeatureExtractor::new();
    let features = extractor.extract_header_embedding("customer_description");
    
    assert_eq!(features.len(), 300); // FastText dimension
    assert!(features.iter().any(|&x| x != 0.0)); // Non-zero embedding
}
```

**Implementation**:
1. Statistical features (mean, std, entropy)
2. Character class ratios
3. FastText embeddings for headers
4. N-gram TF-IDF features
5. Information theory metrics

**Deliverables**:
- `feature_extractor.rs` with comprehensive tests
- Feature vector specification document
- Benchmark: <10ms for full feature extraction

### Task 1.3: ML Model Integration
**Story**: As a developer, I need to integrate pre-trained ONNX models for column classification.

**TDD Tests**:
```rust
#[test]
fn test_load_onnx_model() {
    let model = OnnxModel::load("models/column_classifier.onnx").unwrap();
    assert!(model.is_loaded());
}

#[test]
fn test_model_inference() {
    let model = OnnxModel::load("models/column_classifier.onnx").unwrap();
    let features = vec![0.5; 300]; // Mock features
    let output = model.predict(&features).unwrap();
    
    assert_eq!(output.len(), 8); // 8 column types
    assert!((output.iter().sum::<f32>() - 1.0).abs() < 0.01); // Probabilities sum to 1
}

#[test]
fn test_batch_inference_performance() {
    let model = OnnxModel::load("models/column_classifier.onnx").unwrap();
    let batch = vec![vec![0.5; 300]; 1000]; // 1000 columns
    
    let start = Instant::now();
    let outputs = model.predict_batch(&batch).unwrap();
    let elapsed = start.elapsed();
    
    assert!(elapsed.as_millis() < 1000); // <1 second for 1000 columns
}
```

**Deliverables**:
- ONNX runtime integration
- Model versioning system
- Performance: <1 second for 1000 columns

## Sprint 2 Tasks (Days 6-10)

### Task 1.4: Advanced ML Features
**Story**: As a classifier, I need advanced features like n-grams and entropy for better accuracy.

**TDD Approach**:
```rust
#[test]
fn test_ngram_extraction() {
    let extractor = FeatureExtractor::new();
    let text = "hello world";
    let ngrams = extractor.extract_ngrams(text, 3);
    
    assert!(ngrams.contains("hel"));
    assert!(ngrams.contains("wor"));
    assert_eq!(ngrams.len(), 8); // Correct number of 3-grams
}

#[test]
fn test_shannon_entropy() {
    let extractor = FeatureExtractor::new();
    
    // Low entropy - repetitive
    let low_entropy = extractor.calculate_entropy(&vec!["aaa", "aaa", "aaa"]);
    assert!(low_entropy < 1.0);
    
    // High entropy - diverse
    let high_entropy = extractor.calculate_entropy(&vec!["abc", "xyz", "123"]);
    assert!(high_entropy > 2.0);
}
```

**Deliverables**:
- Enhanced feature extraction
- Feature importance analysis
- A/B testing framework

### Task 1.5: Model Optimization
**Story**: As a system, I need optimized models for production performance.

**TDD Tests**:
```rust
#[test]
fn test_model_quantization() {
    let original = OnnxModel::load("models/column_classifier.onnx").unwrap();
    let quantized = original.quantize(QuantizationLevel::Int8).unwrap();
    
    // Test accuracy preservation
    let test_features = generate_test_features();
    let orig_output = original.predict(&test_features).unwrap();
    let quant_output = quantized.predict(&test_features).unwrap();
    
    for (o, q) in orig_output.iter().zip(quant_output.iter()) {
        assert!((o - q).abs() < 0.05); // Max 5% difference
    }
    
    // Test size reduction
    assert!(quantized.size_bytes() < original.size_bytes() / 2);
}

#[test]
fn test_model_caching() {
    let cache = ModelCache::new(100); // 100MB cache
    let model1 = cache.get_or_load("column_classifier").unwrap();
    let model2 = cache.get_or_load("column_classifier").unwrap();
    
    assert!(Arc::ptr_eq(&model1, &model2)); // Same instance
}
```

**Deliverables**:
- Quantized models for edge deployment
- Model caching system
- Performance monitoring

## Sprint 3 Tasks (Days 11-15)

### Task 1.6: ML Pipeline Integration Testing
**Story**: As a developer, I need comprehensive integration tests for the ML pipeline.

**Integration Tests**:
```rust
#[test]
async fn test_end_to_end_column_profiling() {
    let profiler = ColumnProfiler::new();
    let test_file = create_test_csv_with_known_columns();
    
    let candidates = profiler.profile_file(&test_file).await.unwrap();
    
    // Verify known text columns are identified
    let text_columns = candidates.iter()
        .filter(|c| c.final_score > 0.7)
        .map(|c| &c.name)
        .collect::<Vec<_>>();
    
    assert!(text_columns.contains(&"description"));
    assert!(text_columns.contains(&"comments"));
    assert!(!text_columns.contains(&"price")); // Numeric
    assert!(!text_columns.contains(&"date")); // DateTime
}

#[test]
async fn test_ml_accuracy_on_viznet_dataset() {
    let test_set = load_viznet_test_set();
    let classifier = MLClassifier::new();
    
    let mut correct = 0;
    let mut total = 0;
    
    for (column, true_label) in test_set {
        let prediction = classifier.predict(&column);
        if prediction.column_type == true_label {
            correct += 1;
        }
        total += 1;
    }
    
    let accuracy = correct as f32 / total as f32;
    assert!(accuracy >= 0.93); // Target 93% accuracy
}
```

### Task 1.7: Documentation and Knowledge Transfer
**Story**: As a team, we need comprehensive documentation for the ML system.

**Deliverables**:
1. ML Architecture document
2. Feature engineering guide
3. Model training pipeline
4. Performance tuning guide
5. Troubleshooting runbook

## Performance Requirements
- Column classification: <1ms per column
- Batch processing: <1 second for 1000 columns
- Memory usage: <500MB for classifier
- Model loading: <100ms cold start

## Testing Requirements
- Unit test coverage: 95%+
- Integration test coverage: 80%+
- Performance regression tests
- A/B testing framework
- Model drift detection

## Dependencies
- ONNX Runtime 1.16+
- FastText embeddings
- NumPy/ndarray for matrix operations
- Criterion for benchmarking

## Monitoring Metrics
```rust
pub struct MLMetrics {
    pub predictions_per_second: Gauge,
    pub model_load_time_ms: Histogram,
    pub feature_extraction_time_ms: Histogram,
    pub prediction_accuracy: Gauge,
    pub cache_hit_rate: Gauge,
}
```

## TDD Best Practices Reminder
1. **Write the test first** - No production code without a failing test
2. **One assertion per test** - Keep tests focused
3. **Test names describe behavior** - "test_classifier_identifies_text_column"
4. **Refactor under green** - Only refactor when all tests pass
5. **Delete redundant tests** - Keep test suite lean

## Definition of Done
- [ ] All tests passing (unit + integration)
- [ ] Performance benchmarks met
- [ ] Documentation complete
- [ ] Code reviewed by team
- [ ] Metrics instrumented
- [ ] Feature flags configured
- [ ] Deployment artifacts created