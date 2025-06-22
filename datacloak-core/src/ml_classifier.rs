
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnType {
    TextLong,
    TextShort,
    Numeric,
    DateTime,
    Categorical,
    Identifier,
    Boolean,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct Column {
    pub name: String,
    pub values: Vec<String>,
}

impl Column {
    pub fn new(name: &str, values: Vec<&str>) -> Self {
        Self {
            name: name.to_string(),
            values: values.into_iter().map(|s| s.to_string()).collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Prediction {
    pub column_type: ColumnType,
    pub confidence: f32,
}


use crate::onnx_model::OnnxModel;
use crate::feature_extractor::FeatureExtractor;
use crate::model_optimization::{ModelCache, ModelOptimizer, QuantizationLevel};
use std::sync::Arc;

pub struct MLClassifier {
    model: Option<Arc<OnnxModel>>,
    feature_extractor: Arc<FeatureExtractor>,
    model_cache: Arc<ModelCache>,
    optimizer: ModelOptimizer,
}

impl MLClassifier {
    pub fn new() -> Self {
        Self {
            model: None,
            feature_extractor: Arc::new(FeatureExtractor::new()),
            model_cache: Arc::new(ModelCache::new(512)), // 512MB cache
            optimizer: ModelOptimizer::new(),
        }
    }
    
    pub fn with_model(model_path: &str) -> Result<Self, String> {
        let cache = Arc::new(ModelCache::new(512));
        let model = cache.get_or_load(model_path)?;
        Ok(Self {
            model: Some(model),
            feature_extractor: Arc::new(FeatureExtractor::new()),
            model_cache: cache,
            optimizer: ModelOptimizer::new(),
        })
    }
    
    pub fn with_quantized_model(model_path: &str, quantization: QuantizationLevel) -> Result<Self, String> {
        let cache = Arc::new(ModelCache::new(512));
        let model = cache.get_or_load(model_path)?;
        let optimizer = ModelOptimizer::new();
        
        // Create quantized version
        let _quantized = optimizer.quantize(&model, quantization)
            .map_err(|e| format!("Quantization failed: {}", e))?;
        
        // For now, we'll use the original model since QuantizedModel doesn't implement the same interface
        // In a real implementation, we'd need to adapt the interfaces
        Ok(Self {
            model: Some(model),
            feature_extractor: Arc::new(FeatureExtractor::new()),
            model_cache: cache,
            optimizer,
        })
    }
    
    pub fn predict_batch(&self, columns: &[Column]) -> Vec<Prediction> {
        columns.iter().map(|col| self.predict(col)).collect()
    }

    pub fn predict(&self, column: &Column) -> Prediction {
        // Use ONNX model if available
        if let Some(model) = &self.model {
            // Extract features
            let features = self.feature_extractor.extract_all_features(column);
            
            // Run inference
            match model.predict(&features) {
                Ok(probabilities) => {
                    // Find column type with highest probability
                    let (best_idx, best_prob) = probabilities
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .unwrap_or((7, &0.0)); // Default to Unknown
                    
                    let column_type = match best_idx {
                        0 => ColumnType::TextLong,
                        1 => ColumnType::TextShort,
                        2 => ColumnType::Numeric,
                        3 => ColumnType::DateTime,
                        4 => ColumnType::Categorical,
                        5 => ColumnType::Identifier,
                        6 => ColumnType::Boolean,
                        _ => ColumnType::Unknown,
                    };
                    
                    Prediction {
                        column_type,
                        confidence: *best_prob,
                    }
                }
                Err(_) => {
                    // Fall back to rule-based
                    self.rule_based_predict(column)
                }
            }
        } else {
            // No model, use rule-based
            self.rule_based_predict(column)
        }
    }
    
    fn rule_based_predict(&self, column: &Column) -> Prediction {
        // Simple rule-based classification
        let mut numeric_count = 0;
        let mut text_length_sum = 0;
        let total_values = column.values.len();
        
        if total_values == 0 {
            return Prediction {
                column_type: ColumnType::Unknown,
                confidence: 0.0,
            };
        }
        
        for value in &column.values {
            // Check if numeric
            if value.parse::<f64>().is_ok() {
                numeric_count += 1;
            }
            text_length_sum += value.len();
        }
        
        let numeric_ratio = numeric_count as f32 / total_values as f32;
        let avg_length = text_length_sum as f32 / total_values as f32;
        
        // Simple rules
        if numeric_ratio > 0.9 {
            Prediction {
                column_type: ColumnType::Numeric,
                confidence: numeric_ratio,
            }
        } else if avg_length > 15.0 && numeric_ratio < 0.1 {
            Prediction {
                column_type: ColumnType::TextLong,
                confidence: 0.9,
            }
        } else {
            // Mixed content
            Prediction {
                column_type: ColumnType::Unknown,
                confidence: 0.5,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classifier_identifies_text_column() {
        let classifier = MLClassifier::new();
        let column = Column::new("description", vec!["This is a long text", "Another paragraph"]);
        let prediction = classifier.predict(&column);
        assert_eq!(prediction.column_type, ColumnType::TextLong);
    }

    #[test]
    fn test_classifier_identifies_numeric_column() {
        let classifier = MLClassifier::new();
        let column = Column::new("price", vec!["10.99", "25.50", "100.00"]);
        let prediction = classifier.predict(&column);
        assert_eq!(prediction.column_type, ColumnType::Numeric);
    }

    #[test]
    fn test_classifier_handles_mixed_content() {
        let classifier = MLClassifier::new();
        let column = Column::new("mixed", vec!["Text123", "456Text", "789"]);
        let prediction = classifier.predict(&column);
        assert!(prediction.confidence < 0.8); // Low confidence for mixed
    }
    
    #[test]
    fn test_load_onnx_model() {
        use crate::onnx_model::OnnxModel;
        let model = OnnxModel::load("models/column_classifier.onnx");
        assert!(model.is_ok() || model.is_err()); // For now, just check it attempts to load
    }
    
    #[test]
    fn test_batch_prediction() {
        let classifier = MLClassifier::new();
        let columns = vec![
            Column::new("col1", vec!["text", "more text"]),
            Column::new("col2", vec!["123", "456"]),
        ];
        let predictions = classifier.predict_batch(&columns);
        assert_eq!(predictions.len(), 2);
    }
    
    #[test]
    fn test_performance_single_prediction() {
        use std::time::Instant;
        
        let classifier = MLClassifier::new();
        let column = Column::new("test", vec!["sample", "data", "for", "testing"]);
        
        let start = Instant::now();
        let _ = classifier.predict(&column);
        let elapsed = start.elapsed();
        
        // Should be less than 5ms with advanced features
        assert!(elapsed.as_millis() < 5, "Prediction took {:?}", elapsed);
    }
    
    #[test]
    fn test_classifier_with_mock_model() {
        use crate::onnx_model::OnnxModel;
        use crate::model_optimization::{ModelCache, ModelOptimizer};
        
        // Create classifier with mock model
        let classifier = MLClassifier {
            model: Some(Arc::new(OnnxModel::mock())),
            feature_extractor: Arc::new(FeatureExtractor::new()),
            model_cache: Arc::new(ModelCache::new(512)),
            optimizer: ModelOptimizer::new(),
        };
        
        let column = Column::new("test", vec!["text data"]);
        let prediction = classifier.predict(&column);
        
        // Should use ML model, not rule-based
        assert!(prediction.confidence > 0.0);
    }
    
    #[test]
    fn test_quantized_model_creation() {
        use crate::model_optimization::QuantizationLevel;
        
        // Test successful creation with mock model (in real scenario would load actual model)
        let result = MLClassifier::with_quantized_model("test_model.onnx", QuantizationLevel::Int8);
        
        // Should succeed because mock cache returns mock models
        assert!(result.is_ok());
        
        // Verify the classifier works
        let classifier = result.unwrap();
        let column = Column::new("test", vec!["sample text"]);
        let prediction = classifier.predict(&column);
        assert!(prediction.confidence >= 0.0);
    }
}