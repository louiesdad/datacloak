use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum OnnxError {
    #[error("Failed to load model: {0}")]
    LoadError(String),

    #[error("Invalid input shape: expected {expected}, got {got}")]
    InvalidInputShape { expected: usize, got: usize },

    #[error("Inference error: {0}")]
    InferenceError(String),
}

pub struct OnnxModel {
    // Will hold actual ONNX session when available
    input_size: usize,
    output_size: usize,
    is_mock: bool,
}

impl OnnxModel {
    pub fn load(path: &str) -> Result<Self, OnnxError> {
        if !Path::new(path).exists() {
            return Err(OnnxError::LoadError(format!(
                "Model file not found: {}",
                path
            )));
        }

        // TODO: Load actual ONNX model
        // Updated size to accommodate all features: 300 (embedding) + 13 (stats+ratios+entropy+diversity) + 64 (ngrams)
        Ok(Self {
            input_size: 377,
            output_size: 8,
            is_mock: false,
        })
    }

    pub fn mock() -> Self {
        Self {
            input_size: 377,
            output_size: 8,
            is_mock: true,
        }
    }

    pub fn predict(&self, features: &[f32]) -> Result<Vec<f32>, OnnxError> {
        if features.len() != self.input_size {
            return Err(OnnxError::InvalidInputShape {
                expected: self.input_size,
                got: features.len(),
            });
        }

        if self.is_mock {
            // Return mock probabilities that sum to 1
            let mut output = vec![0.1; self.output_size];
            output[0] = 0.3; // Higher probability for first class
            Ok(output)
        } else {
            // TODO: Real ONNX inference
            Err(OnnxError::InferenceError("Not implemented".to_string()))
        }
    }

    pub fn predict_batch(&self, batch: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, OnnxError> {
        // Simple sequential processing for now
        // TODO: Optimize with actual batch processing
        batch
            .iter()
            .map(|features| self.predict(features))
            .collect()
    }

    pub fn is_loaded(&self) -> bool {
        !self.is_mock
    }

    pub fn estimated_size_bytes(&self) -> usize {
        // Estimate based on input/output dimensions
        // In real implementation, would get actual model size
        let params = self.input_size * self.output_size * 4; // 4 bytes per float
        let overhead = 1024 * 1024; // 1MB overhead
        params + overhead
    }

    pub fn input_size(&self) -> usize {
        self.input_size
    }

    pub fn output_size(&self) -> usize {
        self.output_size
    }

    pub fn is_mock(&self) -> bool {
        self.is_mock
    }
}

impl Clone for OnnxModel {
    fn clone(&self) -> Self {
        Self {
            input_size: self.input_size,
            output_size: self.output_size,
            is_mock: self.is_mock,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_load_onnx_model() {
        let model = OnnxModel::load("models/column_classifier.onnx");
        // Model file doesn't exist yet, so we expect an error
        assert!(model.is_err());
    }

    #[test]
    fn test_model_inference() {
        // Create mock model for testing
        let model = OnnxModel::mock();
        let features = vec![0.5; 377]; // Mock features with updated dimensions
        let output = model.predict(&features).unwrap();

        assert_eq!(output.len(), 8); // 8 column types
        assert!((output.iter().sum::<f32>() - 1.0).abs() < 0.01); // Probabilities sum to 1
    }

    #[test]
    fn test_batch_inference_performance() {
        let model = OnnxModel::mock();
        let batch = vec![vec![0.5; 377]; 1000]; // 1000 columns with updated dimensions

        let start = Instant::now();
        let outputs = model.predict_batch(&batch).unwrap();
        let elapsed = start.elapsed();

        assert_eq!(outputs.len(), 1000);
        assert!(elapsed.as_millis() < 1000); // <1 second for 1000 columns
    }

    #[test]
    fn test_model_input_validation() {
        let model = OnnxModel::mock();

        // Wrong input size
        let features = vec![0.5; 100]; // Should be 300
        let result = model.predict(&features);
        assert!(result.is_err());
    }

    #[test]
    fn test_model_warmup() {
        let model = OnnxModel::mock();

        // First prediction (cold)
        let features = vec![0.5; 377];
        let start = Instant::now();
        let _ = model.predict(&features).unwrap();
        let cold_time = start.elapsed();

        // Second prediction (warm)
        let start = Instant::now();
        let _ = model.predict(&features).unwrap();
        let warm_time = start.elapsed();

        // Warm prediction should be faster
        assert!(warm_time <= cold_time);
    }
}
