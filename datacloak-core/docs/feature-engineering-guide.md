# DataCloak Feature Engineering Guide

## Overview

This guide provides comprehensive documentation for the feature extraction pipeline used in DataCloak's ML-based column classification system. The feature extractor transforms raw column data into a 376-dimensional feature vector optimized for sentiment analysis column identification.

## Feature Categories

### 1. Statistical Features (8 dimensions)

Statistical features capture basic numerical properties of column data:

#### Text Length Statistics (4 features)
```rust
// Implementation in FeatureExtractor::extract_text_stats()
features[0] = mean_length;     // Average character count per value
features[1] = median_length;   // Median character count
features[2] = std_length;      // Standard deviation of lengths
features[3] = length_variance; // Variance in text lengths
```

**Usage**: Distinguishes between short categorical values and long descriptive text.

#### Data Quality Metrics (4 features)
```rust
features[4] = unique_ratio;    // Unique values / Total values
features[5] = null_ratio;      // Null/empty values / Total values  
features[6] = numeric_ratio;   // Numeric values / Total values
features[7] = length_range;    // Max length - Min length
```

**Usage**: Indicates data quality and type consistency.

### 2. Character-based Features (26 dimensions)

Character analysis provides insights into content composition:

#### Character Category Ratios (5 features)
```rust
// Ratios of different character types
features[8]  = alphabetic_ratio;   // Letters / Total chars
features[9]  = numeric_ratio;      // Digits / Total chars  
features[10] = punctuation_ratio;  // Punctuation / Total chars
features[11] = whitespace_ratio;   // Spaces/tabs / Total chars
features[12] = special_ratio;      // Special chars / Total chars
```

#### Unicode Category Distribution (16 features)
```rust
// Unicode character categories
features[13] = uppercase_letter_ratio;
features[14] = lowercase_letter_ratio;
features[15] = titlecase_letter_ratio;
features[16] = decimal_number_ratio;
features[17] = letter_number_ratio;
features[18] = other_number_ratio;
features[19] = connect_punctuation_ratio;
features[20] = dash_punctuation_ratio;
features[21] = close_punctuation_ratio;
features[22] = final_punctuation_ratio;
features[23] = initial_punctuation_ratio;
features[24] = other_punctuation_ratio;
features[25] = open_punctuation_ratio;
features[26] = currency_symbol_ratio;
features[27] = modifier_symbol_ratio;
features[28] = other_symbol_ratio;
```

#### Case Analysis (5 features)
```rust
features[29] = uppercase_ratio;    // All-caps values ratio
features[30] = lowercase_ratio;    // All-lowercase values ratio
features[31] = titlecase_ratio;    // Title Case values ratio
features[32] = mixed_case_ratio;   // Mixed case values ratio
features[33] = case_consistency;   // Case pattern consistency
```

**Usage**: Identifies naming conventions, proper nouns, and text formatting patterns.

### 3. N-gram Features (256 dimensions)

N-gram analysis captures sequential patterns and semantic content:

#### Character N-grams (128 features)
```rust
// Extract character n-grams with feature hashing
pub fn extract_hashed_ngrams(&self, texts: &[&str], n: usize, feature_dim: usize) -> Vec<f32> {
    let mut feature_counts = vec![0u32; feature_dim];
    
    for text in texts {
        for ngram in self.extract_char_ngrams(text, n) {
            let hash = self.hash_ngram(ngram) % feature_dim;
            feature_counts[hash] = feature_counts[hash].saturating_add(1);
        }
    }
    
    // Normalize by total n-gram count
    let total: u32 = feature_counts.iter().sum();
    if total > 0 {
        feature_counts.into_iter()
            .map(|count| count as f32 / total as f32)
            .collect()
    } else {
        vec![0.0; feature_dim]
    }
}
```

**N-gram Types**:
- **2-grams**: Character pairs (captures language patterns)
- **3-grams**: Character triplets (captures common sequences)
- **4-grams**: Longer patterns (captures distinctive sequences)

#### Token N-grams (128 features)
```rust
// Word-level n-grams for semantic analysis
pub fn extract_token_ngrams(&self, column: &Column, n: usize) -> Vec<String> {
    let mut ngrams = Vec::new();
    
    for value in &column.values {
        let tokens: Vec<&str> = value.split_whitespace().collect();
        for window in tokens.windows(n) {
            ngrams.push(window.join(" "));
        }
    }
    
    ngrams
}
```

**Usage**: Captures semantic patterns, common phrases, and domain-specific terminology.

### 4. Entropy and Diversity Metrics (6 dimensions)

Information-theoretic measures of content diversity:

#### Shannon Entropy (3 features)
```rust
pub fn calculate_shannon_entropy(&self, column: &Column) -> f32 {
    let mut char_counts = std::collections::HashMap::new();
    let mut total_chars = 0;
    
    for value in &column.values {
        for ch in value.chars() {
            *char_counts.entry(ch).or_insert(0) += 1;
            total_chars += 1;
        }
    }
    
    if total_chars == 0 {
        return 0.0;
    }
    
    let mut entropy = 0.0;
    for &count in char_counts.values() {
        let probability = count as f32 / total_chars as f32;
        if probability > 0.0 {
            entropy -= probability * probability.log2();
        }
    }
    
    entropy
}

pub fn calculate_normalized_entropy(&self, column: &Column) -> f32 {
    let entropy = self.calculate_shannon_entropy(column);
    let max_entropy = self.calculate_max_possible_entropy(column);
    
    if max_entropy > 0.0 {
        (entropy / max_entropy).min(1.0).max(0.0)
    } else {
        0.0
    }
}
```

**Features**:
- `features[290] = raw_entropy`: Raw Shannon entropy
- `features[291] = normalized_entropy`: Entropy normalized by maximum possible
- `features[292] = entropy_per_char`: Entropy per character

#### Simpson Diversity Index (3 features)
```rust
pub fn calculate_simpson_diversity(&self, column: &Column) -> f32 {
    let mut value_counts = std::collections::HashMap::new();
    let total_values = column.values.len();
    
    for value in &column.values {
        *value_counts.entry(value).or_insert(0) += 1;
    }
    
    if total_values <= 1 {
        return 0.0;
    }
    
    let mut sum_squares = 0.0;
    for &count in value_counts.values() {
        let probability = count as f32 / total_values as f32;
        sum_squares += probability * probability;
    }
    
    1.0 - sum_squares
}
```

**Features**:
- `features[293] = simpson_diversity`: Value diversity measure
- `features[294] = effective_alphabet_size`: Equivalent uniform distribution size
- `features[295] = diversity_ratio`: Diversity relative to maximum possible

**Usage**: Measures content variety and randomness, useful for distinguishing structured vs. free-text data.

### 5. Pattern Recognition Features (80 dimensions)

Specialized pattern detectors for common data types:

#### Email Patterns (10 features)
```rust
// Email pattern detection and analysis
features[296] = email_ratio;           // Values matching email pattern
features[297] = email_domain_diversity; // Unique domains / Total emails
features[298] = common_email_ratio;    // Common providers (gmail, etc.)
features[299] = business_email_ratio;  // Non-generic domains
features[300] = email_length_avg;      // Average email length
features[301] = local_part_diversity;  // Username diversity
features[302] = tld_diversity;         // Top-level domain variety
features[303] = email_complexity;      // Special chars in emails
features[304] = subdomain_ratio;       // Emails with subdomains
features[305] = email_validation_ratio; // Syntactically valid emails
```

#### Phone Number Patterns (10 features)
```rust
// Phone number pattern analysis
features[306] = phone_ratio;           // Phone number pattern matches
features[307] = intl_phone_ratio;      // International format
features[308] = us_phone_ratio;        // US format patterns
features[309] = phone_length_consistency; // Length consistency
features[310] = area_code_diversity;   // Unique area codes
features[311] = formatted_phone_ratio; // With formatting (dashes, parens)
features[312] = extension_ratio;       // With extensions
features[313] = mobile_prefix_ratio;   // Mobile number patterns
features[314] = toll_free_ratio;       // Toll-free number patterns
features[315] = phone_digit_entropy;   // Digit randomness
```

#### URL Patterns (10 features)
```rust
// URL pattern detection
features[316] = url_ratio;             // URL pattern matches
features[317] = domain_diversity;      // Unique domains
features[318] = protocol_diversity;    // HTTP/HTTPS/FTP variety
features[319] = path_complexity;       // URL path complexity
features[320] = query_param_ratio;     // URLs with parameters
features[321] = secure_url_ratio;      // HTTPS URLs
features[322] = subdomain_complexity;  // Subdomain depth
features[323] = port_specification_ratio; // URLs with port numbers
features[324] = fragment_ratio;        // URLs with fragments
features[325] = url_length_avg;        // Average URL length
```

#### Date/Time Patterns (20 features)
```rust
// Date and time pattern analysis
features[326] = date_ratio;            // Date pattern matches
features[327] = time_ratio;            // Time pattern matches
features[328] = datetime_ratio;        // Combined datetime
features[329] = iso_date_ratio;        // ISO 8601 format
features[330] = us_date_ratio;         // MM/DD/YYYY format
features[331] = eu_date_ratio;         // DD/MM/YYYY format
features[332] = timestamp_ratio;       // Unix timestamps
features[333] = relative_date_ratio;   // Relative dates (yesterday, etc.)
features[334] = future_date_ratio;     // Future dates
features[335] = past_date_ratio;       // Historical dates
features[336] = date_range_consistency; // Date range consistency
features[337] = timezone_ratio;        // Timezone specifications
features[338] = weekday_ratio;         // Weekday names
features[339] = month_name_ratio;      // Month names
features[340] = quarter_ratio;         // Quarter specifications
features[341] = fiscal_year_ratio;     // Fiscal year patterns
features[342] = date_separator_consistency; // Consistent separators
features[343] = two_digit_year_ratio;  // Two-digit years
features[344] = four_digit_year_ratio; // Four-digit years
features[345] = chronological_order;   // Temporal ordering
```

#### Identifier Patterns (15 features)
```rust
// ID and key pattern analysis
features[346] = uuid_ratio;            // UUID patterns
features[347] = sequential_id_ratio;   // Sequential numbers
features[348] = alphanumeric_id_ratio; // Mixed alphanumeric
features[349] = prefixed_id_ratio;     // IDs with prefixes
features[350] = checksum_ratio;        // IDs with checksums
features[351] = id_length_consistency; // Consistent ID lengths
features[352] = hex_id_ratio;          // Hexadecimal IDs
features[353] = base64_ratio;          // Base64 encoded
features[354] = hash_pattern_ratio;    // Hash-like patterns
features[355] = composite_id_ratio;    // Multi-part IDs
features[356] = zero_padded_ratio;     // Zero-padded numbers
features[357] = id_entropy;            // Randomness in IDs
features[358] = id_collision_ratio;    // Duplicate IDs
features[359] = hierarchical_id_ratio; // Hierarchical structure
features[360] = version_id_ratio;      // Version identifiers
```

#### Numeric Patterns (15 features)
```rust
// Numeric pattern analysis
features[361] = integer_ratio;         // Integer values
features[362] = decimal_ratio;         // Decimal numbers
features[363] = percentage_ratio;      // Percentage values
features[364] = currency_ratio;        // Currency amounts
features[365] = scientific_notation_ratio; // Scientific notation
features[366] = negative_ratio;        // Negative numbers
features[367] = zero_ratio;            // Zero values
features[368] = round_number_ratio;    // Round numbers (100, 1000)
features[369] = fractional_ratio;      // Fractional values (1/2, 3/4)
features[370] = range_notation_ratio;  // Range values (1-10, 5..15)
features[371] = numeric_precision;     // Decimal places consistency
features[372] = magnitude_range;       // Order of magnitude range
features[373] = numeric_distribution;  // Distribution characteristics
features[374] = outlier_ratio;         // Statistical outliers
features[375] = numeric_pattern_consistency; // Pattern consistency
```

## Feature Engineering Best Practices

### 1. Feature Normalization

All features are normalized to [0, 1] range for consistent model input:

```rust
pub fn normalize_features(&self, features: &mut [f32]) {
    for feature in features.iter_mut() {
        *feature = feature.max(0.0).min(1.0);
    }
}
```

### 2. Missing Value Handling

Handle empty or null columns gracefully:

```rust
if column.values.is_empty() {
    return vec![0.0; 376]; // Return zero vector for empty columns
}
```

### 3. Feature Hashing

Use consistent hashing for n-gram features:

```rust
fn hash_ngram(&self, ngram: &str) -> usize {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    ngram.hash(&mut hasher);
    hasher.finish() as usize
}
```

### 4. Performance Optimization

Optimize for large datasets:

```rust
// Use parallel processing for large columns
use rayon::prelude::*;

pub fn extract_batch_features(&self, columns: &[Column]) -> Vec<Vec<f32>> {
    columns.par_iter()
        .map(|col| self.extract_all_features(col))
        .collect()
}
```

## Testing and Validation

### Unit Tests

Each feature category has comprehensive unit tests:

```rust
#[test]
fn test_statistical_features() {
    let extractor = FeatureExtractor::new();
    let column = Column::new("test", vec!["short", "medium text", "very long text content"]);
    let features = extractor.extract_text_stats(&column);
    
    assert_eq!(features.len(), 8);
    assert!(features[0] > 0.0); // mean_length
    assert!(features[3] > 0.0); // unique_ratio
}
```

### Integration Tests

Validate feature extraction pipeline:

```rust
#[test]
fn test_complete_feature_extraction() {
    let extractor = FeatureExtractor::new();
    let column = Column::new("email", vec![
        "user@example.com", 
        "test@gmail.com", 
        "admin@company.org"
    ]);
    let features = extractor.extract_all_features(&column);
    
    assert_eq!(features.len(), 376);
    assert!(features[296] > 0.8); // High email ratio
}
```

### Performance Tests

Benchmark feature extraction speed:

```rust
#[bench]
fn bench_feature_extraction(b: &mut Bencher) {
    let extractor = FeatureExtractor::new();
    let large_column = create_large_test_column(1000);
    
    b.iter(|| {
        black_box(extractor.extract_all_features(&large_column))
    });
}
```

## Feature Importance Analysis

Based on model training and validation:

### High-Importance Features (Top 20)

1. **Text Length Statistics** (features 0-3): Primary indicators of text vs. non-text
2. **Character Ratios** (features 8-12): Content composition analysis
3. **Email Patterns** (features 296-305): Strong identifier patterns
4. **Entropy Measures** (features 290-295): Content randomness/structure
5. **N-gram Distributions** (features 34-162): Semantic content patterns

### Medium-Importance Features

- Date/time patterns for temporal data identification
- Numeric patterns for numerical data classification
- ID patterns for identifier recognition
- Case analysis for naming conventions

### Low-Importance Features

- Detailed Unicode categories (useful for edge cases)
- Specific TLD distributions
- Rare pattern detectors

## Troubleshooting

### Common Issues

#### 1. High Memory Usage
```rust
// Solution: Process in batches
let batch_size = 100;
for chunk in columns.chunks(batch_size) {
    let features = extract_batch_features(chunk);
    process_features(features);
}
```

#### 2. Slow Feature Extraction
```rust
// Solution: Enable parallel processing
use rayon::prelude::*;
let features: Vec<_> = columns.par_iter()
    .map(|col| extractor.extract_all_features(col))
    .collect();
```

#### 3. Feature Dimension Mismatch
```rust
// Validation: Always check feature vector length
assert_eq!(features.len(), 376, "Feature vector length mismatch");
```

### Performance Tuning

1. **Enable SIMD**: Use `target-cpu=native` for SIMD acceleration
2. **Memory Pool**: Reuse feature vectors to reduce allocations
3. **Caching**: Cache computed features for repeated analysis
4. **Parallel Processing**: Use Rayon for multi-core feature extraction

## Advanced Topics

### Custom Feature Engineering

Add domain-specific features:

```rust
impl FeatureExtractor {
    pub fn extract_domain_features(&self, column: &Column) -> Vec<f32> {
        // Custom domain-specific feature extraction
        let mut features = Vec::new();
        
        // Add sentiment-specific features
        features.push(self.sentiment_keyword_ratio(column));
        features.push(self.emotion_word_ratio(column));
        features.push(self.opinion_marker_ratio(column));
        
        features
    }
}
```

### Feature Selection

Automatic feature selection based on importance:

```rust
pub fn select_features(&self, features: &[f32], importance_mask: &[bool]) -> Vec<f32> {
    features.iter()
        .zip(importance_mask.iter())
        .filter_map(|(f, &important)| if important { Some(*f) } else { None })
        .collect()
}
```

### Online Feature Updates

Update features with new data patterns:

```rust
pub fn update_feature_stats(&mut self, new_columns: &[Column]) {
    // Update feature statistics based on new data
    self.pattern_cache.update(new_columns);
    self.ngram_vocabulary.expand(new_columns);
}
```

## References

- [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
- [Text Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html)
- [Information Theory in Machine Learning](https://en.wikipedia.org/wiki/Information_theory)
- [N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf)