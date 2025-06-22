use datacloak_core::graph::SimilarityCalculator;
use datacloak_core::performance::{MemoryPool, CacheFriendlyGraph, SimdOps};
use std::time::Instant;

#[test]
fn test_simd_similarity() {
    let calc = SimilarityCalculator::new();
    let vec1 = vec![1.0; 1024];
    let vec2 = vec![0.5; 1024];
    
    let _scalar_result = calc.cosine_similarity(&vec1, &vec2);
    
    #[cfg(feature = "similarity-search")]
    {
        let simd_result = calc.cosine_similarity_simd(&vec1, &vec2);
        assert!((scalar_result - simd_result).abs() < 0.0001);
    }
}

#[test]
fn test_simd_performance_improvement() {
    #[cfg(feature = "similarity-search")]
    {
        let calc = SimilarityCalculator::new();
        let vec1: Vec<f32> = (0..1024).map(|i| (i as f32).sin()).collect();
        let vec2: Vec<f32> = (0..1024).map(|i| (i as f32).cos()).collect();
        
        // Scalar version timing
        let start = Instant::now();
        for _ in 0..10000 {
            calc.cosine_similarity(&vec1, &vec2);
        }
        let scalar_time = start.elapsed();
        
        // SIMD version timing
        let start = Instant::now();
        for _ in 0..10000 {
            calc.cosine_similarity_simd(&vec1, &vec2);
        }
        let simd_time = start.elapsed();
        
        println!("Scalar time: {:?}, SIMD time: {:?}", scalar_time, simd_time);
        assert!(simd_time < scalar_time / 2); // At least 2x faster
    }
}

#[test]
fn test_memory_pool_allocation() {
    let pool = MemoryPool::new(1024 * 1024); // 1MB pool
    
    // Allocate a few vectors
    let allocation1 = pool.allocate::<f32>(100).unwrap();
    let allocation2 = pool.allocate::<f32>(200).unwrap();
    
    // Check pool utilization
    let stats = pool.stats();
    assert!(stats.allocated_bytes > 0);
    assert!(stats.free_bytes > 0);
    assert_eq!(stats.total_bytes, 1024 * 1024);
    
    // Verify allocations work
    assert_eq!(allocation1.as_slice().len(), 100);
    assert_eq!(allocation2.as_slice().len(), 200);
}

#[test]
fn test_memory_pool_performance() {
    let pool = MemoryPool::new(10 * 1024 * 1024); // 10MB pool
    
    let start = Instant::now();
    let mut pool_allocations = vec![];
    for i in 0..100 {
        let allocation = pool.allocate::<f32>(100 + i).unwrap();
        pool_allocations.push(allocation);
    }
    let pool_time = start.elapsed();
    
    println!("Pool allocation time: {:?}", pool_time);
    assert!(pool_allocations.len() == 100);
}

#[test]
fn test_cache_friendly_graph() {
    let graph = CacheFriendlyGraph::new();
    
    // Add nodes
    let nodes: Vec<_> = (0..1000)
        .map(|i| graph.add_node(format!("col_{}", i), vec![i as f32; 100]))
        .collect();
    
    // Add edges in cache-friendly order
    for i in 0..1000 {
        for j in i+1..i+10.min(1000) {
            if j < 1000 {
                graph.add_edge(nodes[i], nodes[j], 0.5);
            }
        }
    }
    
    // Test cache-friendly traversal
    let start = Instant::now();
    let mut neighbor_count = 0;
    for node in graph.iter_nodes_cache_friendly() {
        neighbor_count += graph.get_neighbors(node).len();
    }
    let cache_friendly_time = start.elapsed();
    
    println!("Cache-friendly traversal: {:?}, Total neighbors: {}", 
             cache_friendly_time, neighbor_count);
    
    // Verify correctness
    assert_eq!(graph.node_count(), 1000);
    assert!(neighbor_count > 0);
}

#[test]
fn test_prefetch_optimization() {
    let data: Vec<Vec<f32>> = (0..1000)
        .map(|i| vec![i as f32; 300])
        .collect();
    
    // Test with prefetching
    let start = Instant::now();
    let mut sum = 0.0;
    for i in 0..data.len() {
        // Prefetch next data
        if i + 1 < data.len() {
            SimdOps::prefetch(&data[i + 1]);
        }
        sum += data[i].iter().sum::<f32>();
    }
    let prefetch_time = start.elapsed();
    
    // Test without prefetching
    let start = Instant::now();
    let mut sum2 = 0.0;
    for vec in &data {
        sum2 += vec.iter().sum::<f32>();
    }
    let normal_time = start.elapsed();
    
    println!("Prefetch time: {:?}, Normal time: {:?}", prefetch_time, normal_time);
    assert!((sum - sum2).abs() < 0.01);
}

#[test]
fn test_batch_similarity_computation() {
    let calc = SimilarityCalculator::new();
    let vectors: Vec<Vec<f32>> = (0..100)
        .map(|i| (0..300).map(|j| ((i * j) as f32).sin()).collect())
        .collect();
    
    // Test batch computation
    let start = Instant::now();
    let results = SimdOps::batch_cosine_similarity(&vectors[0], &vectors[1..], 8);
    let batch_time = start.elapsed();
    
    // Compare with sequential
    let start = Instant::now();
    let sequential_results: Vec<f32> = vectors[1..]
        .iter()
        .map(|v| calc.cosine_similarity(&vectors[0], v))
        .collect();
    let sequential_time = start.elapsed();
    
    println!("Batch time: {:?}, Sequential time: {:?}", batch_time, sequential_time);
    
    // Verify results match
    for (batch, seq) in results.iter().zip(sequential_results.iter()) {
        assert!((batch - seq).abs() < 0.0001);
    }
}

#[test]
fn test_lock_free_graph_operations() {
    use std::sync::Arc;
    use std::thread;
    
    let graph = Arc::new(CacheFriendlyGraph::new_lock_free());
    
    // Add nodes concurrently
    let mut handles = vec![];
    for thread_id in 0..4 {
        let graph_clone = graph.clone();
        let handle = thread::spawn(move || {
            for i in 0..250 {
                let node_id = thread_id * 250 + i;
                graph_clone.add_node(format!("col_{}", node_id), vec![node_id as f32; 100]);
            }
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    assert_eq!(graph.node_count(), 1000);
}

#[test]
fn test_simd_dot_product() {
    let a = vec![1.0_f32; 1024];
    let b = vec![2.0_f32; 1024];
    
    let _scalar_result = SimdOps::dot_product_scalar(&a, &b);
    
    #[cfg(feature = "similarity-search")]
    {
        let simd_result = SimdOps::dot_product_simd(&a, &b);
        assert!((scalar_result - simd_result).abs() < 0.01);
        assert_eq!(simd_result, 2048.0);
    }
}

#[test]
fn test_aligned_allocation() {
    let aligned_vec = SimdOps::allocate_aligned::<f32>(1024, 32);
    
    // Check alignment
    let ptr = aligned_vec.as_ptr() as usize;
    assert_eq!(ptr % 32, 0, "Vector should be 32-byte aligned");
    
    // Check size
    assert_eq!(aligned_vec.len(), 1024);
}