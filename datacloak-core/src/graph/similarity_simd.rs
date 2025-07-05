#[cfg(feature = "similarity-search")]
use packed_simd_2::{f32x8, SimdVector};

#[cfg(feature = "similarity-search")]
pub fn cosine_similarity_simd(vec1: &[f32], vec2: &[f32]) -> f32 {
    if vec1.len() != vec2.len() || vec1.is_empty() {
        return 0.0;
    }

    let len = vec1.len();
    let chunks = len / 8;
    let remainder = len % 8;

    let mut dot_product = f32x8::splat(0.0);
    let mut norm1 = f32x8::splat(0.0);
    let mut norm2 = f32x8::splat(0.0);

    // Process 8 elements at a time
    for i in 0..chunks {
        let offset = i * 8;
        let v1 = f32x8::from_slice_unaligned(&vec1[offset..]);
        let v2 = f32x8::from_slice_unaligned(&vec2[offset..]);

        dot_product += v1 * v2;
        norm1 += v1 * v1;
        norm2 += v2 * v2;
    }

    // Sum the SIMD vectors
    let mut dot_sum = dot_product.sum();
    let mut norm1_sum = norm1.sum();
    let mut norm2_sum = norm2.sum();

    // Handle remaining elements
    let offset = chunks * 8;
    for i in 0..remainder {
        let idx = offset + i;
        dot_sum += vec1[idx] * vec2[idx];
        norm1_sum += vec1[idx] * vec1[idx];
        norm2_sum += vec2[idx] * vec2[idx];
    }

    let norm1_sqrt = norm1_sum.sqrt();
    let norm2_sqrt = norm2_sum.sqrt();

    if norm1_sqrt == 0.0 || norm2_sqrt == 0.0 {
        0.0
    } else {
        dot_sum / (norm1_sqrt * norm2_sqrt)
    }
}

#[cfg(feature = "similarity-search")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_vs_scalar_accuracy() {
        let vec1: Vec<f32> = (0..256).map(|i| (i as f32).sin()).collect();
        let vec2: Vec<f32> = (0..256).map(|i| (i as f32).cos()).collect();

        let scalar_result =
            crate::graph::SimilarityCalculator::new().cosine_similarity(&vec1, &vec2);
        let simd_result = cosine_similarity_simd(&vec1, &vec2);

        assert!((scalar_result - simd_result).abs() < 0.0001);
    }

    #[test]
    fn test_simd_non_aligned_sizes() {
        for size in [7, 15, 17, 31, 33, 63, 65, 127, 129].iter() {
            let vec1: Vec<f32> = (0..*size).map(|i| i as f32 / *size as f32).collect();
            let vec2: Vec<f32> = (0..*size)
                .map(|i| 1.0 - (i as f32 / *size as f32))
                .collect();

            let scalar_result =
                crate::graph::SimilarityCalculator::new().cosine_similarity(&vec1, &vec2);
            let simd_result = cosine_similarity_simd(&vec1, &vec2);

            assert!(
                (scalar_result - simd_result).abs() < 0.0001,
                "Mismatch for size {}: scalar={}, simd={}",
                size,
                scalar_result,
                simd_result
            );
        }
    }
}
