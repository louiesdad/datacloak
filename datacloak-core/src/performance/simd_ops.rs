#[cfg(feature = "similarity-search")]
use packed_simd_2::{f32x8, f32x4};
use std::alloc::{alloc, Layout};

pub struct SimdOps;

impl SimdOps {
    #[inline]
    pub fn prefetch<T>(_data: &[T]) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            // Prefetch hint - implementation varies by platform
            let _ = data.as_ptr();
        }
    }
    
    pub fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
    
    #[cfg(feature = "similarity-search")]
    pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let len = a.len();
        
        let chunks = len / 8;
        let remainder = len % 8;
        
        let mut sum = f32x8::splat(0.0);
        
        // Process 8 elements at a time
        for i in 0..chunks {
            let offset = i * 8;
            let va = f32x8::from_slice_unaligned(&a[offset..]);
            let vb = f32x8::from_slice_unaligned(&b[offset..]);
            sum += va * vb;
        }
        
        let mut result = sum.sum();
        
        // Handle remainder
        let offset = chunks * 8;
        for i in 0..remainder {
            result += a[offset + i] * b[offset + i];
        }
        
        result
    }
    
    pub fn batch_cosine_similarity(
        query: &[f32],
        targets: &[Vec<f32>],
        batch_size: usize,
    ) -> Vec<f32> {
        use rayon::prelude::*;
        
        targets
            .par_chunks(batch_size)
            .flat_map(|batch| {
                batch.iter()
                    .map(|target| {
                        let dot = Self::dot_product_scalar(query, target);
                        let norm1: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
                        let norm2: f32 = target.iter().map(|x| x * x).sum::<f32>().sqrt();
                        
                        if norm1 == 0.0 || norm2 == 0.0 {
                            0.0
                        } else {
                            dot / (norm1 * norm2)
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }
    
    pub fn allocate_aligned<T>(count: usize, alignment: usize) -> Vec<T> {
        assert!(alignment.is_power_of_two());
        
        let size = count * std::mem::size_of::<T>();
        let layout = Layout::from_size_align(size, alignment).unwrap();
        
        unsafe {
            let ptr = alloc(layout) as *mut T;
            if ptr.is_null() {
                panic!("Failed to allocate aligned memory");
            }
            
            Vec::from_raw_parts(ptr, count, count)
        }
    }
    
    #[cfg(feature = "similarity-search")]
    pub fn euclidean_distance_simd(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let len = a.len();
        
        let chunks = len / 8;
        let remainder = len % 8;
        
        let mut sum = f32x8::splat(0.0);
        
        // Process 8 elements at a time
        for i in 0..chunks {
            let offset = i * 8;
            let va = f32x8::from_slice_unaligned(&a[offset..]);
            let vb = f32x8::from_slice_unaligned(&b[offset..]);
            let diff = va - vb;
            sum += diff * diff;
        }
        
        let mut result = sum.sum();
        
        // Handle remainder
        let offset = chunks * 8;
        for i in 0..remainder {
            let diff = a[offset + i] - b[offset + i];
            result += diff * diff;
        }
        
        result.sqrt()
    }
    
    pub fn vector_add_scalar(a: &[f32], b: &[f32], result: &mut [f32]) {
        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }
    }
    
    #[cfg(feature = "similarity-search")]
    pub fn vector_add_simd(a: &[f32], b: &[f32], result: &mut [f32]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), result.len());
        
        let len = a.len();
        let chunks = len / 8;
        let remainder = len % 8;
        
        // Process 8 elements at a time
        for i in 0..chunks {
            let offset = i * 8;
            let va = f32x8::from_slice_unaligned(&a[offset..]);
            let vb = f32x8::from_slice_unaligned(&b[offset..]);
            let sum = va + vb;
            sum.write_to_slice_unaligned(&mut result[offset..]);
        }
        
        // Handle remainder
        let offset = chunks * 8;
        for i in 0..remainder {
            result[offset + i] = a[offset + i] + b[offset + i];
        }
    }
    
    pub fn normalize_vector(vec: &mut [f32]) {
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in vec.iter_mut() {
                *x /= norm;
            }
        }
    }
    
    #[cfg(feature = "similarity-search")]
    pub fn normalize_vector_simd(vec: &mut [f32]) {
        let norm = Self::dot_product_simd(vec, vec).sqrt();
        if norm > 0.0 {
            let len = vec.len();
            let chunks = len / 8;
            let remainder = len % 8;
            
            let norm_vec = f32x8::splat(norm);
            
            for i in 0..chunks {
                let offset = i * 8;
                let v = f32x8::from_slice_unaligned(&vec[offset..]);
                let normalized = v / norm_vec;
                normalized.write_to_slice_unaligned(&mut vec[offset..]);
            }
            
            let offset = chunks * 8;
            for i in 0..remainder {
                vec[offset + i] /= norm;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        
        let result = SimdOps::dot_product_scalar(&a, &b);
        assert_eq!(result, 70.0); // 1*5 + 2*6 + 3*7 + 4*8
        
        #[cfg(feature = "similarity-search")]
        {
            let simd_result = SimdOps::dot_product_simd(&a, &b);
            assert_eq!(simd_result, 70.0);
        }
    }
    
    #[test]
    fn test_normalize_vector() {
        let mut vec = vec![3.0, 4.0];
        SimdOps::normalize_vector(&mut vec);
        
        assert!((vec[0] - 0.6).abs() < 0.001);
        assert!((vec[1] - 0.8).abs() < 0.001);
        
        #[cfg(feature = "similarity-search")]
        {
            let mut vec2 = vec![3.0, 4.0];
            SimdOps::normalize_vector_simd(&mut vec2);
            assert!((vec2[0] - 0.6).abs() < 0.001);
            assert!((vec2[1] - 0.8).abs() < 0.001);
        }
    }
}