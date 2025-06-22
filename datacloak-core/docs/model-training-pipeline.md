# Model Training Pipeline Documentation

## Overview

This document describes the complete machine learning model training pipeline for DataCloak's column classification system. The pipeline transforms raw training data into production-ready ONNX models optimized for sentiment analysis column identification.

## Training Architecture

```
Raw Data → Feature Engineering → Model Training → Validation → ONNX Export → Optimization
    ↓             ↓                    ↓             ↓           ↓             ↓
Training     376-dim          ML Models      Accuracy     Production    Quantized
Dataset      Features         (Various)      Metrics      Model         Models
```

## Training Data Requirements

### Data Collection

#### Training Dataset Structure
```
training_data/
├── labeled_columns/
│   ├── text_long/          # Long text columns (reviews, descriptions)
│   ├── text_short/         # Short text columns (categories, tags)
│   ├── numeric/            # Numeric columns (prices, counts)
│   ├── datetime/           # Date/time columns
│   ├── categorical/        # Categorical columns
│   ├── identifier/         # ID columns (UUIDs, keys)
│   ├── boolean/            # Boolean columns
│   └── unknown/            # Unlabeled/mixed columns
├── validation/             # Validation split (20%)
├── test/                   # Test split (20%)
└── metadata.json          # Dataset metadata
```

#### Data Quality Requirements

1. **Minimum Samples per Class**:
   - TextLong: 5,000 samples
   - TextShort: 3,000 samples
   - Numeric: 2,000 samples
   - DateTime: 1,500 samples
   - Categorical: 2,000 samples
   - Identifier: 1,500 samples
   - Boolean: 1,000 samples
   - Unknown: 1,000 samples

2. **Data Diversity**:
   - Multiple domains (e-commerce, social media, enterprise)
   - Various languages (primarily English, with international support)
   - Different data formats and structures
   - Balanced representation across industries

3. **Quality Metrics**:
   - Label accuracy > 95%
   - Class balance within 2:1 ratio
   - No duplicate columns across splits
   - Representative sample sizes

### Synthetic Data Generation

For augmenting training data:

```python
# Synthetic data generator (Python script)
import numpy as np
import pandas as pd
from faker import Faker

class SyntheticDataGenerator:
    def __init__(self):
        self.fake = Faker()
    
    def generate_text_long(self, n_samples=1000):
        """Generate long text samples for reviews/descriptions"""
        samples = []
        for _ in range(n_samples):
            # Product descriptions
            description = f"{self.fake.sentence()} {self.fake.text(max_nb_chars=200)}"
            samples.append(description)
        return samples
    
    def generate_text_short(self, n_samples=1000):
        """Generate short text samples for categories"""
        categories = ['Electronics', 'Clothing', 'Books', 'Sports', 'Home & Garden']
        samples = [self.fake.random_element(categories) for _ in range(n_samples)]
        return samples
    
    def generate_numeric(self, n_samples=1000):
        """Generate numeric samples"""
        samples = []
        for _ in range(n_samples):
            if np.random.random() < 0.3:
                # Integers
                samples.append(str(np.random.randint(1, 10000)))
            else:
                # Floats
                samples.append(f"{np.random.uniform(0.01, 9999.99):.2f}")
        return samples
```

## Feature Engineering Pipeline

### 1. Feature Extraction

```rust
// Feature extraction for training
pub struct TrainingFeatureExtractor {
    base_extractor: FeatureExtractor,
    feature_stats: FeatureStatistics,
}

impl TrainingFeatureExtractor {
    pub fn extract_training_features(&self, dataset: &TrainingDataset) -> TrainingFeatures {
        let mut features = Vec::new();
        let mut labels = Vec::new();
        
        for (column, label) in dataset.samples() {
            let feature_vector = self.base_extractor.extract_all_features(column);
            features.push(feature_vector);
            labels.push(self.encode_label(label));
        }
        
        TrainingFeatures { features, labels }
    }
    
    fn encode_label(&self, label: &str) -> usize {
        match label {
            "text_long" => 0,
            "text_short" => 1,
            "numeric" => 2,
            "datetime" => 3,
            "categorical" => 4,
            "identifier" => 5,
            "boolean" => 6,
            _ => 7, // unknown
        }
    }
}
```

### 2. Feature Normalization

```rust
pub struct FeatureNormalizer {
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl FeatureNormalizer {
    pub fn fit(&mut self, features: &[Vec<f32>]) {
        let n_features = features[0].len();
        self.mean = vec![0.0; n_features];
        self.std = vec![1.0; n_features];
        
        // Calculate mean
        for feature_vec in features {
            for (i, &value) in feature_vec.iter().enumerate() {
                self.mean[i] += value;
            }
        }
        for mean_val in &mut self.mean {
            *mean_val /= features.len() as f32;
        }
        
        // Calculate standard deviation
        for feature_vec in features {
            for (i, &value) in feature_vec.iter().enumerate() {
                let diff = value - self.mean[i];
                self.std[i] += diff * diff;
            }
        }
        for (i, std_val) in self.std.iter_mut().enumerate() {
            *std_val = (*std_val / features.len() as f32).sqrt().max(1e-8);
        }
    }
    
    pub fn transform(&self, features: &mut [f32]) {
        for (i, feature) in features.iter_mut().enumerate() {
            *feature = (*feature - self.mean[i]) / self.std[i];
        }
    }
}
```

## Model Training

### 1. Model Architectures

#### Random Forest (Primary Model)
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class RandomForestTrainer:
    def __init__(self):
        self.param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
    
    def train(self, X_train, y_train, X_val, y_val):
        # Grid search for best parameters
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            rf, self.param_grid, 
            cv=5, scoring='f1_macro', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Validate on holdout set
        best_model = grid_search.best_estimator_
        val_score = best_model.score(X_val, y_val)
        
        return best_model, val_score
```

#### Neural Network (Alternative Model)
```python
import tensorflow as tf
from tensorflow.keras import layers, models

class NeuralNetworkTrainer:
    def __init__(self, input_dim=376, num_classes=8):
        self.input_dim = input_dim
        self.num_classes = num_classes
    
    def build_model(self):
        model = models.Sequential([
            layers.Dense(512, activation='relu', input_shape=(self.input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(), 
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100):
        model = self.build_model()
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
        ]
        
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history
```

### 2. Training Pipeline

```python
class ModelTrainingPipeline:
    def __init__(self, config):
        self.config = config
        self.feature_extractor = TrainingFeatureExtractor()
        self.normalizer = FeatureNormalizer()
        
    def run_training(self, dataset_path):
        print("Loading training data...")
        dataset = self.load_dataset(dataset_path)
        
        print("Extracting features...")
        features, labels = self.feature_extractor.extract_training_features(dataset)
        
        print("Splitting data...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(features, labels)
        
        print("Normalizing features...")
        self.normalizer.fit(X_train)
        X_train_norm = self.normalizer.transform(X_train)
        X_val_norm = self.normalizer.transform(X_val)
        X_test_norm = self.normalizer.transform(X_test)
        
        print("Training models...")
        models = {}
        
        # Train Random Forest
        rf_trainer = RandomForestTrainer()
        rf_model, rf_score = rf_trainer.train(X_train_norm, y_train, X_val_norm, y_val)
        models['random_forest'] = rf_model
        
        # Train Neural Network
        nn_trainer = NeuralNetworkTrainer()
        nn_model, nn_history = nn_trainer.train(X_train_norm, y_train, X_val_norm, y_val)
        models['neural_network'] = nn_model
        
        print("Evaluating models...")
        results = self.evaluate_models(models, X_test_norm, y_test)
        
        print("Selecting best model...")
        best_model = self.select_best_model(models, results)
        
        print("Exporting to ONNX...")
        onnx_path = self.export_to_onnx(best_model)
        
        return best_model, results, onnx_path
```

## Model Evaluation

### Evaluation Metrics

1. **Classification Metrics**:
   - Accuracy (overall correct predictions)
   - Precision (per-class precision)
   - Recall (per-class recall)
   - F1-score (harmonic mean of precision/recall)
   - Macro-averaged F1 (average across classes)

2. **Confusion Matrix Analysis**:
   ```python
   from sklearn.metrics import confusion_matrix, classification_report
   
   def evaluate_model(model, X_test, y_test):
       y_pred = model.predict(X_test)
       
       # Confusion matrix
       cm = confusion_matrix(y_test, y_pred)
       
       # Classification report
       report = classification_report(y_test, y_pred, 
                                    target_names=CLASS_NAMES)
       
       return cm, report
   ```

3. **Sentiment Analysis Relevance**:
   - TextLong/TextShort precision (primary targets)
   - False positive rate for non-text columns
   - Confidence calibration for uncertain predictions

### Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold

def cross_validate_model(model, X, y, cv_folds=5):
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        model.fit(X_train_fold, y_train_fold)
        score = model.score(X_val_fold, y_val_fold)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)
```

## ONNX Model Export

### Model Conversion

```python
import tf2onnx
import onnx

def export_tensorflow_to_onnx(tf_model, output_path):
    """Convert TensorFlow model to ONNX"""
    spec = (tf.TensorSpec((None, 376), tf.float32, name="input"),)
    
    output_path = output_path.replace('.onnx', '.onnx')
    model_proto, _ = tf2onnx.convert.from_keras(tf_model, input_signature=spec)
    
    with open(output_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    
    return output_path

def export_sklearn_to_onnx(sklearn_model, X_sample, output_path):
    """Convert scikit-learn model to ONNX"""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, X_sample.shape[1]]))]
    onnx_model = convert_sklearn(sklearn_model, initial_types=initial_type)
    
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    return output_path
```

### Model Validation

```python
import onnxruntime as ort

def validate_onnx_model(onnx_path, test_features, expected_predictions):
    """Validate ONNX model produces correct outputs"""
    session = ort.InferenceSession(onnx_path)
    
    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Run inference
    result = session.run([output_name], {input_name: test_features.astype(np.float32)})
    predictions = np.argmax(result[0], axis=1)
    
    # Compare with expected
    accuracy = np.mean(predictions == expected_predictions)
    print(f"ONNX model accuracy: {accuracy:.4f}")
    
    return accuracy > 0.95  # Should match original model closely
```

## Model Optimization

### Quantization

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_onnx_model(input_path, output_path, quantization_type='int8'):
    """Apply dynamic quantization to ONNX model"""
    if quantization_type == 'int8':
        quant_type = QuantType.QInt8
    elif quantization_type == 'int16':
        quant_type = QuantType.QInt16
    else:
        raise ValueError(f"Unsupported quantization type: {quantization_type}")
    
    quantize_dynamic(
        input_path,
        output_path,
        weight_type=quant_type
    )
    
    return output_path
```

### Model Pruning

```python
import tensorflow_model_optimization as tfmot

def prune_tensorflow_model(model, X_train, y_train, target_sparsity=0.5):
    """Apply magnitude-based pruning to TensorFlow model"""
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    
    # Define pruning schedule
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=target_sparsity,
            begin_step=0,
            end_step=1000
        )
    }
    
    # Apply pruning to model layers
    model_for_pruning = prune_low_magnitude(model, **pruning_params)
    
    # Compile and train
    model_for_pruning.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Fine-tune with pruning
    model_for_pruning.fit(X_train, y_train, epochs=10, verbose=1)
    
    # Remove pruning wrappers
    final_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    
    return final_model
```

## Production Deployment

### Model Versioning

```yaml
# model_config.yaml
model_metadata:
  name: "datacloak_column_classifier"
  version: "1.2.0"
  training_date: "2024-01-15"
  framework: "tensorflow"
  accuracy: 0.923
  features:
    count: 376
    version: "1.0"
  
model_files:
  onnx: "models/column_classifier_v1.2.0.onnx"
  quantized_int8: "models/column_classifier_v1.2.0_int8.onnx"
  quantized_int16: "models/column_classifier_v1.2.0_int16.onnx"
  
performance_benchmarks:
  inference_time_ms: 2.1
  memory_usage_mb: 45
  accuracy_test_set: 0.918
  
deployment:
  min_rust_version: "1.70.0"
  min_onnx_runtime: "1.15.0"
  recommended_cache_size_mb: 512
```

### Continuous Training Pipeline

```python
class ContinuousTrainingPipeline:
    def __init__(self, config):
        self.config = config
        self.model_store = ModelStore(config.model_store_path)
        
    def retrain_model(self, new_data_path, current_model_path):
        """Retrain model with new data"""
        # Load current model performance
        current_metrics = self.load_model_metrics(current_model_path)
        
        # Load and combine training data
        new_data = self.load_dataset(new_data_path)
        existing_data = self.load_dataset(self.config.training_data_path)
        combined_data = self.combine_datasets(existing_data, new_data)
        
        # Train new model
        new_model, new_metrics, onnx_path = self.run_training(combined_data)
        
        # Compare performance
        if self.is_better_model(new_metrics, current_metrics):
            # Deploy new model
            self.deploy_model(onnx_path, new_metrics)
            print("New model deployed successfully")
        else:
            print("New model performance insufficient, keeping current model")
    
    def is_better_model(self, new_metrics, current_metrics):
        """Compare model performance"""
        accuracy_improvement = new_metrics['accuracy'] - current_metrics['accuracy']
        f1_improvement = new_metrics['f1_macro'] - current_metrics['f1_macro']
        
        # Require minimum improvement to deploy
        return (accuracy_improvement > 0.01 and f1_improvement > 0.01)
```

## Training Scripts

### Main Training Script

```python
#!/usr/bin/env python3
"""
DataCloak Model Training Pipeline
Usage: python train_model.py --config config.yaml --data-path /path/to/data
"""

import argparse
import yaml
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Train DataCloak column classifier')
    parser.add_argument('--config', type=str, required=True, help='Training configuration file')
    parser.add_argument('--data-path', type=str, required=True, help='Path to training data')
    parser.add_argument('--output-dir', type=str, default='./models', help='Output directory for models')
    parser.add_argument('--model-type', type=str, choices=['rf', 'nn', 'both'], default='both')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize training pipeline
    pipeline = ModelTrainingPipeline(config)
    
    # Run training
    best_model, results, onnx_path = pipeline.run_training(args.data_path)
    
    print(f"Training completed. Best model saved to: {onnx_path}")
    print(f"Model performance: {results}")

if __name__ == "__main__":
    main()
```

### Configuration Template

```yaml
# training_config.yaml
training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  validation_split: 0.2
  test_split: 0.2
  random_seed: 42

models:
  random_forest:
    n_estimators: 200
    max_depth: 20
    min_samples_split: 5
    min_samples_leaf: 2
    max_features: "sqrt"
  
  neural_network:
    hidden_layers: [512, 256, 128, 64]
    dropout_rate: 0.3
    batch_normalization: true
    activation: "relu"

data:
  feature_version: "1.0"
  normalization: "standard"
  augmentation: true
  synthetic_ratio: 0.2

evaluation:
  metrics: ["accuracy", "precision", "recall", "f1_macro"]
  cross_validation_folds: 5
  min_accuracy_threshold: 0.85

export:
  onnx_opset_version: 14
  optimize_for_inference: true
  quantization_levels: ["int8", "int16"]
```

## Monitoring and Maintenance

### Model Performance Monitoring

```python
class ModelMonitor:
    def __init__(self, model_path, metrics_store):
        self.model = load_onnx_model(model_path)
        self.metrics_store = metrics_store
        
    def log_prediction(self, features, prediction, confidence, ground_truth=None):
        """Log prediction for monitoring"""
        metrics = {
            'timestamp': datetime.now(),
            'prediction': prediction,
            'confidence': confidence,
            'features_hash': hash(tuple(features)),
        }
        
        if ground_truth is not None:
            metrics['ground_truth'] = ground_truth
            metrics['correct'] = (prediction == ground_truth)
        
        self.metrics_store.log(metrics)
    
    def check_model_drift(self, recent_window_days=7):
        """Check for model performance drift"""
        recent_metrics = self.metrics_store.get_recent(recent_window_days)
        
        if len(recent_metrics) < 100:
            return False  # Insufficient data
        
        recent_accuracy = sum(m['correct'] for m in recent_metrics if 'correct' in m) / len([m for m in recent_metrics if 'correct' in m])
        baseline_accuracy = self.get_baseline_accuracy()
        
        drift_detected = abs(recent_accuracy - baseline_accuracy) > 0.05
        
        if drift_detected:
            self.alert_model_drift(recent_accuracy, baseline_accuracy)
        
        return drift_detected
```

## Best Practices

### 1. Data Quality
- Maintain high-quality labeled datasets
- Regular data validation and cleaning
- Balanced class representation
- Domain diversity in training data

### 2. Model Validation
- Cross-validation with multiple folds
- Hold-out test sets for final evaluation
- Performance monitoring in production
- A/B testing for model updates

### 3. Version Control
- Track model versions with metadata
- Reproducible training pipelines
- Configuration management
- Model artifact versioning

### 4. Performance Optimization
- Feature selection based on importance
- Model quantization for inference speed
- Batch processing for efficiency
- Caching strategies for repeated predictions

### 5. Continuous Improvement
- Regular model retraining with new data
- Performance monitoring and alerting
- Feedback loop integration
- Automated pipeline execution

## Troubleshooting

### Common Training Issues

1. **Overfitting**:
   - Increase regularization
   - Add more training data
   - Use cross-validation
   - Implement early stopping

2. **Class Imbalance**:
   - Use class weights
   - Apply SMOTE or similar techniques
   - Collect more minority class samples
   - Use stratified sampling

3. **Feature Issues**:
   - Check feature scaling/normalization
   - Validate feature extraction pipeline
   - Remove correlated features
   - Handle missing values properly

4. **ONNX Export Issues**:
   - Verify model compatibility
   - Check ONNX opset version
   - Validate input/output shapes
   - Test converted model accuracy

## References

- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [scikit-learn Model Selection](https://scikit-learn.org/stable/model_selection.html)
- [TensorFlow Model Optimization](https://www.tensorflow.org/model_optimization)
- [MLOps Best Practices](https://ml-ops.org/)