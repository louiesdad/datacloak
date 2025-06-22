# ML Models Directory

This directory contains ONNX models for column classification.

## Models

- `column_classifier.onnx` - Main column type classifier model
  - Input: Feature vector (300 dimensions)
  - Output: Probabilities for 8 column types
  - Expected accuracy: 93%+

## Model Training

The column classifier model should be trained on diverse datasets including:
- Various text lengths and patterns
- Numeric data in different formats
- Date/time patterns
- Mixed content examples

## Model Versioning

Models are versioned using semantic versioning:
- Major: Breaking changes to input/output format
- Minor: Improvements to accuracy/features
- Patch: Bug fixes or minor tweaks