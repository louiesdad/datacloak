use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnData {
    pub embedding: Vec<f32>,
    pub tokens: Vec<String>,
}

#[derive(Clone)]
pub struct SimilarityCalculator;

impl SimilarityCalculator {
    pub fn new() -> Self {
        Self
    }

    pub fn cosine_similarity(&self, vec1: &[f32], vec2: &[f32]) -> f32 {
        if vec1.len() != vec2.len() || vec1.is_empty() {
            return 0.0;
        }

        let mut dot_product = 0.0;
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;

        for i in 0..vec1.len() {
            dot_product += vec1[i] * vec2[i];
            norm1 += vec1[i] * vec1[i];
            norm2 += vec2[i] * vec2[i];
        }

        let norm1 = norm1.sqrt();
        let norm2 = norm2.sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1 * norm2)
        }
    }

    pub fn jaccard_similarity(&self, set1: &[&str], set2: &[&str]) -> f32 {
        if set1.is_empty() && set2.is_empty() {
            return 1.0;
        }
        
        if set1.is_empty() || set2.is_empty() {
            return 0.0;
        }

        let s1: HashSet<&str> = set1.iter().cloned().collect();
        let s2: HashSet<&str> = set2.iter().cloned().collect();

        let intersection = s1.intersection(&s2).count();
        let union = s1.union(&s2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    pub fn combined_similarity(
        &self,
        col1: &ColumnData,
        col2: &ColumnData,
        embedding_weight: f32,
        token_weight: f32,
    ) -> f32 {
        let embedding_sim = self.cosine_similarity(&col1.embedding, &col2.embedding);
        
        let tokens1: Vec<&str> = col1.tokens.iter().map(|s| s.as_str()).collect();
        let tokens2: Vec<&str> = col2.tokens.iter().map(|s| s.as_str()).collect();
        let token_sim = self.jaccard_similarity(&tokens1, &tokens2);

        embedding_weight * embedding_sim + token_weight * token_sim
    }
    
    #[cfg(feature = "similarity-search")]
    pub fn cosine_similarity_simd(&self, vec1: &[f32], vec2: &[f32]) -> f32 {
        super::similarity_simd::cosine_similarity_simd(vec1, vec2)
    }
}