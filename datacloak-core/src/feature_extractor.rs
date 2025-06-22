use crate::ml_classifier::Column;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct StatisticalFeatures {
    pub mean_length: f32,
    pub min_length: usize,
    pub max_length: usize,
    pub std_deviation: f32,
}

#[derive(Debug, Clone)]
pub struct CharacterRatios {
    pub digit_ratio: f32,
    pub alpha_ratio: f32,
    pub special_ratio: f32,
    pub whitespace_ratio: f32,
}

#[derive(Debug, Clone)]
pub struct EntropyMetrics {
    pub shannon_entropy: f32,
    pub normalized_entropy: f32,
    pub entropy_rate: f32,
}

pub struct FeatureExtractor {
    // Will hold FastText model and other resources
}

impl FeatureExtractor {
    pub fn new() -> Self {
        Self {}
    }
    
    pub fn extract_stats(&self, column: &Column) -> StatisticalFeatures {
        let lengths: Vec<usize> = column.values.iter().map(|v| v.len()).collect();
        let mean_length = lengths.iter().sum::<usize>() as f32 / lengths.len() as f32;
        let min_length = *lengths.iter().min().unwrap_or(&0);
        let max_length = *lengths.iter().max().unwrap_or(&0);
        
        // Calculate std deviation
        let variance = lengths.iter()
            .map(|&len| {
                let diff = len as f32 - mean_length;
                diff * diff
            })
            .sum::<f32>() / lengths.len() as f32;
        let std_deviation = variance.sqrt();
        
        StatisticalFeatures {
            mean_length,
            min_length,
            max_length,
            std_deviation,
        }
    }
    
    pub fn extract_char_ratios(&self, column: &Column) -> CharacterRatios {
        let mut total_chars = 0;
        let mut digit_count = 0;
        let mut alpha_count = 0;
        let mut special_count = 0;
        let mut whitespace_count = 0;
        
        for value in &column.values {
            for ch in value.chars() {
                total_chars += 1;
                if ch.is_ascii_digit() {
                    digit_count += 1;
                } else if ch.is_alphabetic() {
                    alpha_count += 1;
                } else if ch.is_whitespace() {
                    whitespace_count += 1;
                } else {
                    special_count += 1;
                }
            }
        }
        
        let total = total_chars as f32;
        CharacterRatios {
            digit_ratio: digit_count as f32 / total,
            alpha_ratio: alpha_count as f32 / total,
            special_ratio: special_count as f32 / total,
            whitespace_ratio: whitespace_count as f32 / total,
        }
    }
    
    pub fn extract_header_embedding(&self, header: &str) -> Vec<f32> {
        // Placeholder: generate simple embedding based on header
        // In real implementation, use FastText or similar
        let mut embedding = vec![0.0; 300];
        
        // Simple hash-based embedding for now
        for (i, ch) in header.chars().enumerate() {
            let idx = (ch as usize * (i + 1)) % 300;
            embedding[idx] = 1.0;
        }
        
        embedding
    }
    
    pub fn calculate_entropy(&self, column: &Column) -> f32 {
        // Count character frequencies
        let mut char_counts = HashMap::new();
        let mut total_chars = 0;
        
        for value in &column.values {
            for ch in value.chars() {
                *char_counts.entry(ch).or_insert(0) += 1;
                total_chars += 1;
            }
        }
        
        // Calculate Shannon entropy
        let mut entropy = 0.0;
        for count in char_counts.values() {
            let probability = *count as f32 / total_chars as f32;
            if probability > 0.0 {
                entropy -= probability * probability.log2();
            }
        }
        
        entropy
    }
    
    pub fn extract_char_ngrams<'a>(&self, text: &'a str, n: usize) -> Vec<&'a str> {
        let chars: Vec<char> = text.chars().collect();
        let mut ngrams = Vec::new();
        
        if chars.len() >= n {
            for i in 0..=(chars.len() - n) {
                let start = text.char_indices().nth(i).map(|(idx, _)| idx).unwrap_or(0);
                let end = text.char_indices().nth(i + n).map(|(idx, _)| idx).unwrap_or(text.len());
                ngrams.push(&text[start..end]);
            }
        }
        
        ngrams
    }
    
    pub fn extract_hashed_ngrams(&self, texts: &[&str], n: usize, feature_dim: usize) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut features = vec![0.0; feature_dim];
        
        for text in texts {
            let ngrams = self.extract_char_ngrams(text, n);
            for ngram in ngrams {
                // Hash the n-gram to get an index
                let mut hasher = DefaultHasher::new();
                ngram.hash(&mut hasher);
                let hash = hasher.finish();
                let index = (hash as usize) % feature_dim;
                
                // Increment feature value (feature hashing)
                features[index] += 1.0;
            }
        }
        
        // Normalize by total n-grams
        let total: f32 = features.iter().sum();
        if total > 0.0 {
            for feature in &mut features {
                *feature /= total;
            }
        }
        
        features
    }
    
    pub fn extract_token_ngrams<'a>(&self, text: &'a str, n: usize) -> Vec<Vec<&'a str>> {
        let tokens: Vec<&str> = text.split_whitespace().collect();
        let mut ngrams = Vec::new();
        
        if tokens.len() >= n {
            for i in 0..=(tokens.len() - n) {
                ngrams.push(tokens[i..i+n].to_vec());
            }
        }
        
        ngrams
    }
    
    pub fn calculate_simpson_diversity(&self, column: &Column) -> f32 {
        let mut value_counts = HashMap::new();
        let mut total = 0;
        
        // Count occurrences of each unique value
        for value in &column.values {
            *value_counts.entry(value.as_str()).or_insert(0) += 1;
            total += 1;
        }
        
        if total <= 1 {
            return 0.0;
        }
        
        // Calculate Simpson's diversity index: 1 - Σ(n_i * (n_i - 1)) / (N * (N - 1))
        let mut sum = 0;
        for count in value_counts.values() {
            sum += count * (count - 1);
        }
        
        1.0 - (sum as f32 / (total * (total - 1)) as f32)
    }
    
    pub fn extract_ngram_frequencies(&self, column: &Column, n: usize) -> HashMap<String, usize> {
        let mut frequencies = HashMap::new();
        
        for value in &column.values {
            let ngrams = self.extract_char_ngrams(value, n);
            for ngram in ngrams {
                *frequencies.entry(ngram.to_string()).or_insert(0) += 1;
            }
        }
        
        frequencies
    }
    
    pub fn calculate_advanced_entropy(&self, column: &Column) -> EntropyMetrics {
        // Shannon entropy
        let shannon_entropy = self.calculate_entropy(column);
        
        // Calculate alphabet size for normalization
        let mut unique_chars = std::collections::HashSet::new();
        for value in &column.values {
            for ch in value.chars() {
                unique_chars.insert(ch);
            }
        }
        let alphabet_size = unique_chars.len() as f32;
        
        // Normalized entropy (0 to 1)
        let max_entropy = if alphabet_size > 1.0 {
            alphabet_size.log2()
        } else {
            1.0 // Avoid division by zero for single-character alphabet
        };
        
        let normalized_entropy = if max_entropy > 0.0 && shannon_entropy >= 0.0 {
            (shannon_entropy / max_entropy).min(1.0) // Ensure it doesn't exceed 1.0
        } else {
            0.0
        };
        
        // Entropy rate (considers n-gram patterns)
        let mut entropy_rate = 0.0;
        if column.values.len() > 0 {
            // Calculate bigram entropy
            let bigram_freqs = self.extract_ngram_frequencies(column, 2);
            let total_bigrams: usize = bigram_freqs.values().sum();
            
            if total_bigrams > 0 {
                for count in bigram_freqs.values() {
                    let prob = *count as f32 / total_bigrams as f32;
                    if prob > 0.0 {
                        entropy_rate -= prob * prob.log2();
                    }
                }
            }
        }
        
        EntropyMetrics {
            shannon_entropy,
            normalized_entropy,
            entropy_rate,
        }
    }
    
    pub fn extract_all_features(&self, column: &Column) -> Vec<f32> {
        let mut features = Vec::new();
        
        // Add header embedding
        features.extend(self.extract_header_embedding(&column.name));
        
        // Add statistical features
        let stats = self.extract_stats(column);
        features.push(stats.mean_length);
        features.push(stats.min_length as f32);
        features.push(stats.max_length as f32);
        features.push(stats.std_deviation);
        
        // Add character ratios
        let ratios = self.extract_char_ratios(column);
        features.push(ratios.digit_ratio);
        features.push(ratios.alpha_ratio);
        features.push(ratios.special_ratio);
        features.push(ratios.whitespace_ratio);
        
        // Add advanced entropy metrics
        let entropy_metrics = self.calculate_advanced_entropy(column);
        features.push(entropy_metrics.shannon_entropy);
        features.push(entropy_metrics.normalized_entropy);
        features.push(entropy_metrics.entropy_rate);
        
        // Add Simpson diversity
        features.push(self.calculate_simpson_diversity(column));
        
        // Add hashed n-gram features (64 dimensions)
        let texts: Vec<&str> = column.values.iter().map(|s| s.as_str()).collect();
        let ngram_features = self.extract_hashed_ngrams(&texts, 3, 64);
        features.extend(ngram_features);
        
        features
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml_classifier::Column;

    #[test]
    fn test_extract_statistical_features() {
        let extractor = FeatureExtractor::new();
        let column = Column::new("test", vec!["hello", "world", "foo"]);
        let features = extractor.extract_stats(&column);
        
        assert!((features.mean_length - 4.33).abs() < 0.01);
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
    
    #[test]
    fn test_shannon_entropy() {
        let extractor = FeatureExtractor::new();
        
        // Low entropy - repetitive
        let low_entropy_column = Column::new("test", vec!["aaa", "aaa", "aaa"]);
        let low_entropy = extractor.calculate_entropy(&low_entropy_column);
        assert!(low_entropy < 1.0);
        
        // High entropy - diverse
        let high_entropy_column = Column::new("test", vec!["abc", "xyz", "123"]);
        let high_entropy = extractor.calculate_entropy(&high_entropy_column);
        assert!(high_entropy > 2.0);
    }
    
    #[test]
    fn test_full_feature_vector() {
        let extractor = FeatureExtractor::new();
        let column = Column::new("customer_name", vec!["John Doe", "Jane Smith", "Bob Wilson"]);
        let features = extractor.extract_all_features(&column);
        
        // Should have all features combined
        assert!(features.len() > 300); // Embedding + stats + ratios
    }
    
    #[test]
    fn test_feature_extraction_performance() {
        use std::time::Instant;
        
        let extractor = FeatureExtractor::new();
        let column = Column::new(
            "description",
            vec![
                "This is a long description text",
                "Another paragraph with more content",
                "Yet another sentence for testing",
                "Final text to test performance"
            ]
        );
        
        let start = Instant::now();
        let _ = extractor.extract_all_features(&column);
        let elapsed = start.elapsed();
        
        // Should be less than 10ms for full feature extraction
        assert!(elapsed.as_millis() < 10, "Feature extraction took {:?}", elapsed);
    }
    
    #[test]
    fn test_character_ngrams() {
        let extractor = FeatureExtractor::new();
        let text = "hello";
        let ngrams = extractor.extract_char_ngrams(text, 3);
        assert!(ngrams.contains(&"hel"));
        assert!(ngrams.contains(&"ell"));
        assert!(ngrams.contains(&"llo"));
    }
    
    #[test]
    fn test_hashed_ngram_features() {
        let extractor = FeatureExtractor::new();
        let features = extractor.extract_hashed_ngrams(&["hello world"], 3, 256);
        assert_eq!(features.len(), 256); // Fixed dimension
    }
    
    #[test]
    fn test_ngram_frequencies() {
        let extractor = FeatureExtractor::new();
        let column = Column::new("test", vec!["the the cat", "the dog"]);
        let frequencies = extractor.extract_ngram_frequencies(&column, 3);
        
        // "the" appears 3 times
        assert_eq!(*frequencies.get("the").unwrap_or(&0), 3);
        // "he " appears 3 times (from "the ")
        assert_eq!(*frequencies.get("he ").unwrap_or(&0), 3);
    }
    
    #[test]
    fn test_advanced_entropy_metrics() {
        let extractor = FeatureExtractor::new();
        let column = Column::new("test", vec!["aaa", "bbb", "ccc"]);
        
        let metrics = extractor.calculate_advanced_entropy(&column);
        
        // Should have multiple entropy metrics
        assert!(metrics.shannon_entropy > 0.0);
        assert!(metrics.normalized_entropy >= 0.0 && metrics.normalized_entropy <= 1.0);
        assert!(metrics.entropy_rate >= 0.0);
    }
    
    #[test]
    fn test_simpson_diversity_index() {
        let extractor = FeatureExtractor::new();
        let column = Column::new("test", vec!["apple", "banana", "apple", "cherry"]);
        let diversity = extractor.calculate_simpson_diversity(&column);
        
        // Diversity should be between 0 and 1
        assert!(diversity >= 0.0 && diversity <= 1.0);
        // With 3 different values out of 4, diversity should be relatively high
        assert!(diversity > 0.5);
    }
    
    #[test]
    fn test_token_ngrams() {
        let extractor = FeatureExtractor::new();
        let text = "the quick brown fox";
        let token_ngrams = extractor.extract_token_ngrams(text, 2);
        
        assert!(token_ngrams.contains(&vec!["the", "quick"]));
        assert!(token_ngrams.contains(&vec!["quick", "brown"]));
        assert!(token_ngrams.contains(&vec!["brown", "fox"]));
    }
}