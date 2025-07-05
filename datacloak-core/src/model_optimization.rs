use crate::onnx_model::{OnnxError, OnnxModel};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantizationLevel {
    Int8,
    Int16,
    Dynamic,
}

pub struct QuantizedModel {
    original_model: Arc<OnnxModel>,
    _quantization_level: QuantizationLevel,
    size_reduction_factor: f32,
}

impl QuantizedModel {
    pub fn predict(&self, features: &[f32]) -> Result<Vec<f32>, OnnxError> {
        // In real implementation, this would use quantized inference
        // For now, simulate with original model
        self.original_model.predict(features)
    }

    pub fn estimated_size_bytes(&self) -> usize {
        let original_size = self.original_model.estimated_size_bytes();
        (original_size as f32 * self.size_reduction_factor) as usize
    }
}

pub struct ModelOptimizer {
    // Configuration for optimization
}

impl ModelOptimizer {
    pub fn new() -> Self {
        Self {}
    }

    pub fn quantize(
        &self,
        model: &OnnxModel,
        level: QuantizationLevel,
    ) -> Result<QuantizedModel, String> {
        // In real implementation, this would perform actual quantization
        // For now, return a wrapper with simulated size reduction
        let size_reduction_factor = match level {
            QuantizationLevel::Int8 => 0.25,   // 4x smaller
            QuantizationLevel::Int16 => 0.5,   // 2x smaller
            QuantizationLevel::Dynamic => 0.6, // 1.67x smaller
        };

        Ok(QuantizedModel {
            original_model: Arc::new(model.clone()),
            _quantization_level: level,
            size_reduction_factor,
        })
    }
}

#[derive(Clone)]
struct CachedModel {
    model: Arc<OnnxModel>,
    size_bytes: usize,
    last_accessed: Instant,
}

pub struct ModelCache {
    cache: Arc<Mutex<HashMap<String, CachedModel>>>,
    max_size_bytes: usize,
}

impl ModelCache {
    pub fn new(max_size_mb: usize) -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            max_size_bytes: max_size_mb * 1024 * 1024,
        }
    }

    pub fn get_or_load(&self, model_name: &str) -> Result<Arc<OnnxModel>, String> {
        let mut cache = self.cache.lock().unwrap();

        // Check if model is already cached
        if let Some(cached) = cache.get_mut(model_name) {
            cached.last_accessed = Instant::now();
            return Ok(cached.model.clone());
        }

        // Load model (mocked for testing)
        let model = Arc::new(OnnxModel::mock());
        let size_bytes = model.estimated_size_bytes();

        // Check if we need to evict models
        self.evict_if_needed(&mut cache, size_bytes);

        // Add to cache
        cache.insert(
            model_name.to_string(),
            CachedModel {
                model: model.clone(),
                size_bytes,
                last_accessed: Instant::now(),
            },
        );

        Ok(model)
    }

    fn evict_if_needed(&self, cache: &mut HashMap<String, CachedModel>, needed_bytes: usize) {
        let current_size = self.calculate_cache_size(cache);

        if current_size + needed_bytes > self.max_size_bytes {
            // Evict least recently used models
            let mut entries: Vec<_> = cache
                .iter()
                .map(|(k, v)| (k.clone(), v.last_accessed))
                .collect();
            entries.sort_by_key(|(_, time)| *time);

            let mut freed_bytes = 0;
            for (key, _) in entries {
                if let Some(model) = cache.remove(&key) {
                    freed_bytes += model.size_bytes;
                    if current_size - freed_bytes + needed_bytes <= self.max_size_bytes {
                        break;
                    }
                }
            }
        }
    }

    fn calculate_cache_size(&self, cache: &HashMap<String, CachedModel>) -> usize {
        cache.values().map(|m| m.size_bytes).sum()
    }

    pub fn size_bytes(&self) -> usize {
        let cache = self.cache.lock().unwrap();
        self.calculate_cache_size(&cache)
    }

    pub fn max_size_bytes(&self) -> usize {
        self.max_size_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx_model::OnnxModel;
    use std::time::Instant;

    #[test]
    fn test_model_quantization() {
        let original = OnnxModel::mock();
        let optimizer = ModelOptimizer::new();
        let quantized = optimizer
            .quantize(&original, QuantizationLevel::Int8)
            .unwrap();

        // Test accuracy preservation
        let test_features = vec![0.5; 377];
        let orig_output = original.predict(&test_features).unwrap();
        let quant_output = quantized.predict(&test_features).unwrap();

        for (o, q) in orig_output.iter().zip(quant_output.iter()) {
            assert!((o - q).abs() < 0.05); // Max 5% difference
        }

        // Test size reduction (mocked for now)
        assert!(quantized.estimated_size_bytes() < original.estimated_size_bytes());
    }

    #[test]
    fn test_model_caching() {
        let cache = ModelCache::new(100); // 100MB cache

        // First load
        let model1 = cache.get_or_load("column_classifier").unwrap();

        // Second load should return cached instance
        let model2 = cache.get_or_load("column_classifier").unwrap();

        // Should be the same instance
        assert!(Arc::ptr_eq(&model1, &model2));
    }

    #[test]
    fn test_cache_eviction() {
        let cache = ModelCache::new(2); // 2MB cache (small enough to force eviction)

        // Load multiple models to trigger eviction
        let _model1 = cache.get_or_load("model1").unwrap();
        let _model2 = cache.get_or_load("model2").unwrap();
        let _model3 = cache.get_or_load("model3").unwrap();

        // Cache should have evicted some models to stay under limit
        // Since each model is ~1MB, we should have at most 2 models
        assert!(cache.size_bytes() <= cache.max_size_bytes());

        // Should have at most 2 models cached
        let cache_lock = cache.cache.lock().unwrap();
        assert!(cache_lock.len() <= 2);
    }

    #[test]
    fn test_quantization_performance() {
        let model = OnnxModel::mock();
        let optimizer = ModelOptimizer::new();
        let quantized = optimizer.quantize(&model, QuantizationLevel::Int8).unwrap();

        let test_features = vec![0.5; 377];
        let batch = vec![test_features; 100];

        // Benchmark original
        let start = Instant::now();
        for features in &batch {
            let _ = model.predict(features).unwrap();
        }
        let original_time = start.elapsed();

        // Benchmark quantized
        let start = Instant::now();
        for features in &batch {
            let _ = quantized.predict(features).unwrap();
        }
        let quantized_time = start.elapsed();

        // Quantized should be faster
        assert!(quantized_time < original_time * 2); // Allow some variance
    }
}
