use datacloak_core::graph::{ColumnData, SimilarityCalculator};

#[test]
fn test_cosine_similarity() {
    let calc = SimilarityCalculator::new();
    let vec1 = vec![1.0, 0.0, 0.0];
    let vec2 = vec![1.0, 0.0, 0.0];

    assert_eq!(calc.cosine_similarity(&vec1, &vec2), 1.0); // Identical
}

#[test]
fn test_cosine_similarity_orthogonal() {
    let calc = SimilarityCalculator::new();
    let vec1 = vec![1.0, 0.0];
    let vec2 = vec![0.0, 1.0];

    assert_eq!(calc.cosine_similarity(&vec1, &vec2), 0.0); // Orthogonal
}

#[test]
fn test_cosine_similarity_normalized() {
    let calc = SimilarityCalculator::new();
    let vec1 = vec![3.0, 4.0]; // magnitude = 5
    let vec2 = vec![0.6, 0.8]; // magnitude = 1 (normalized)

    let similarity = calc.cosine_similarity(&vec1, &vec2);
    assert!((similarity - 1.0).abs() < 0.0001); // Should be 1.0
}

#[test]
fn test_cosine_similarity_negative() {
    let calc = SimilarityCalculator::new();
    let vec1 = vec![1.0, 0.0];
    let vec2 = vec![-1.0, 0.0];

    assert_eq!(calc.cosine_similarity(&vec1, &vec2), -1.0); // Opposite directions
}

#[test]
fn test_jaccard_similarity() {
    let calc = SimilarityCalculator::new();
    let set1 = vec!["hello", "world"];
    let set2 = vec!["hello", "rust"];

    let similarity = calc.jaccard_similarity(&set1, &set2);
    assert!((similarity - 0.333333).abs() < 0.01); // 1/3
}

#[test]
fn test_jaccard_similarity_identical() {
    let calc = SimilarityCalculator::new();
    let set1 = vec!["hello", "world", "rust"];
    let set2 = vec!["hello", "world", "rust"];

    assert_eq!(calc.jaccard_similarity(&set1, &set2), 1.0);
}

#[test]
fn test_jaccard_similarity_disjoint() {
    let calc = SimilarityCalculator::new();
    let set1 = vec!["hello", "world"];
    let set2 = vec!["foo", "bar"];

    assert_eq!(calc.jaccard_similarity(&set1, &set2), 0.0);
}

#[test]
fn test_jaccard_similarity_empty() {
    let calc = SimilarityCalculator::new();
    let set1: Vec<&str> = vec![];
    let set2 = vec!["hello"];

    assert_eq!(calc.jaccard_similarity(&set1, &set2), 0.0);

    let set3: Vec<&str> = vec![];
    let set4: Vec<&str> = vec![];
    assert_eq!(calc.jaccard_similarity(&set3, &set4), 1.0); // Empty sets are identical
}

#[test]
fn test_combined_similarity() {
    let calc = SimilarityCalculator::new();
    let col1 = ColumnData {
        embedding: vec![1.0, 0.0],
        tokens: vec!["hello".to_string(), "world".to_string()],
    };
    let col2 = ColumnData {
        embedding: vec![0.7, 0.7],
        tokens: vec!["hello".to_string(), "rust".to_string()],
    };

    let similarity = calc.combined_similarity(&col1, &col2, 0.6, 0.4);
    assert!(similarity > 0.0 && similarity < 1.0);
}

#[test]
fn test_combined_similarity_weights() {
    let calc = SimilarityCalculator::new();
    let col1 = ColumnData {
        embedding: vec![1.0, 0.0],
        tokens: vec!["hello".to_string()],
    };
    let col2 = ColumnData {
        embedding: vec![0.0, 1.0],         // Orthogonal embeddings
        tokens: vec!["hello".to_string()], // Identical tokens
    };

    // Full weight on embeddings (orthogonal)
    let sim1 = calc.combined_similarity(&col1, &col2, 1.0, 0.0);
    assert_eq!(sim1, 0.0);

    // Full weight on tokens (identical)
    let sim2 = calc.combined_similarity(&col1, &col2, 0.0, 1.0);
    assert_eq!(sim2, 1.0);

    // Equal weights
    let sim3 = calc.combined_similarity(&col1, &col2, 0.5, 0.5);
    assert!((sim3 - 0.5).abs() < 0.0001);
}

#[test]
fn test_cosine_similarity_large_vectors() {
    let calc = SimilarityCalculator::new();
    let vec1: Vec<f32> = (0..300).map(|i| i as f32).collect();
    let vec2: Vec<f32> = (0..300).map(|i| (i as f32) * 2.0).collect();

    let similarity = calc.cosine_similarity(&vec1, &vec2);
    assert!((similarity - 1.0).abs() < 0.0001); // Parallel vectors
}

#[cfg(feature = "similarity-search")]
#[test]
fn test_simd_cosine_similarity() {
    let calc = SimilarityCalculator::new();
    let vec1: Vec<f32> = vec![1.0; 1024];
    let vec2: Vec<f32> = vec![0.5; 1024];

    let scalar_result = calc.cosine_similarity(&vec1, &vec2);
    let simd_result = calc.cosine_similarity_simd(&vec1, &vec2);

    assert!((scalar_result - simd_result).abs() < 0.0001);
}
