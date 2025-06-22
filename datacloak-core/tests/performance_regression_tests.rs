use datacloak_core::{
    graph::{ColumnGraph, ColumnNode, SimilarityCalculator},
    performance::{MemoryPool, CacheFriendlyGraph, SimdOps},
};
use std::time::Instant;

/// Performance regression test suite to ensure optimizations don't degrade
#[cfg(test)]
mod regression_tests {
    use super::*;

    #[test]
    fn test_graph_construction_performance() {
        let iterations = 3;
        let mut times = vec![];

        for _ in 0..iterations {
            let start = Instant::now();
            
            let mut graph = ColumnGraph::new();
            let nodes: Vec<_> = (0..1000)
                .map(|i| graph.add_node(ColumnNode::new(&format!("col_{}", i), vec![i as f32; 100])))
                .collect();
            
            // Add edges
            for i in 0..1000 {
                for j in i+1..i+10.min(1000) {
                    if j < 1000 {
                        graph.add_edge(nodes[i], nodes[j], 0.5);
                    }
                }
            }
            
            let elapsed = start.elapsed();
            times.push(elapsed.as_millis());
        }

        let avg_time = times.iter().sum::<u128>() / times.len() as u128;
        println!("Graph construction (1000 nodes): avg {}ms, times: {:?}", avg_time, times);
        
        // Performance requirement: <5 seconds for 1000 nodes
        assert!(avg_time < 5000, "Graph construction too slow: {}ms", avg_time);
        
        // Consistency check: variance should be reasonable (allow some variance in debug mode)
        let max_variance = times.iter().max().unwrap() - times.iter().min().unwrap();
        assert!(max_variance <= avg_time.max(2), "Performance too inconsistent: variance {}ms", max_variance);
    }

    #[test]
    fn test_pagerank_performance() {
        let mut graph = ColumnGraph::new();
        let nodes: Vec<_> = (0..1000)
            .map(|i| graph.add_node(ColumnNode::new(&format!("node_{}", i), vec![i as f32; 10])))
            .collect();

        // Create a connected graph
        for i in 0..1000 {
            for j in 0..10 {
                let target = (i + j * 100) % 1000;
                if target != i {
                    graph.add_edge(nodes[i], nodes[target], 0.5);
                }
            }
        }

        let iterations = 5;
        let mut times = vec![];

        for _ in 0..iterations {
            let start = Instant::now();
            let ranks = graph.calculate_pagerank(0.85, 100);
            let elapsed = start.elapsed();
            
            assert_eq!(ranks.len(), 1000);
            times.push(elapsed.as_millis());
        }

        let avg_time = times.iter().sum::<u128>() / times.len() as u128;
        println!("PageRank (1000 nodes): avg {}ms, times: {:?}", avg_time, times);
        
        // Performance requirement: <100ms for 1000 nodes
        assert!(avg_time < 100, "PageRank too slow: {}ms", avg_time);
    }

    #[test]
    fn test_similarity_computation_performance() {
        let calc = SimilarityCalculator::new();
        let iterations = 10000;
        
        // Test different vector sizes
        for &size in &[100, 300, 1024] {
            let vec1: Vec<f32> = (0..size).map(|i| i as f32 / size as f32).collect();
            let vec2: Vec<f32> = (0..size).map(|i| (size - i) as f32 / size as f32).collect();
            
            let start = Instant::now();
            for _ in 0..iterations {
                let _similarity = calc.cosine_similarity(&vec1, &vec2);
            }
            let elapsed = start.elapsed();
            
            let per_calculation = elapsed.as_micros() as f64 / iterations as f64;
            println!("Cosine similarity ({}D): {:.2}µs per calculation", size, per_calculation);
            
            // Performance requirement: reasonable for debug mode
            assert!(per_calculation < 50.0, "Similarity calculation too slow for {}D: {:.2}µs", size, per_calculation);
        }
    }

    #[test]
    fn test_memory_pool_performance() {
        let pool = MemoryPool::new(100 * 1024 * 1024); // 100MB pool
        let iterations = 10000;
        
        // Test allocation performance
        let start = Instant::now();
        let mut allocations = vec![];
        
        for i in 0..iterations {
            if let Ok(allocation) = pool.allocate::<f32>(100 + (i % 500)) {
                allocations.push(allocation);
            }
            
            // Periodically free some allocations
            if i % 1000 == 0 && !allocations.is_empty() {
                allocations.drain(0..allocations.len()/2);
            }
        }
        
        let elapsed = start.elapsed();
        let per_allocation = elapsed.as_micros() as f64 / iterations as f64;
        
        println!("Memory pool allocation: {:.2}µs per allocation ({} successful)", 
                per_allocation, allocations.len());
        
        // Performance requirement: reasonable allocation speed
        assert!(per_allocation < 100.0, "Memory pool allocation too slow: {:.2}µs", per_allocation);
        assert!(allocations.len() > iterations / 10, "Too many allocation failures: {} out of {}", allocations.len(), iterations);
    }

    #[test]
    fn test_cache_friendly_graph_performance() {
        let graph = CacheFriendlyGraph::new();
        let node_count = 1000;
        
        // Measure node creation
        let start = Instant::now();
        let nodes: Vec<_> = (0..node_count)
            .map(|i| graph.add_node(format!("col_{}", i), vec![i as f32; 100]))
            .collect();
        let node_creation_time = start.elapsed();
        
        // Measure edge creation
        let start = Instant::now();
        for i in 0..node_count {
            for j in i+1..i+10.min(node_count) {
                if j < node_count {
                    graph.add_edge(nodes[i], nodes[j], 0.5);
                }
            }
        }
        let edge_creation_time = start.elapsed();
        
        // Measure traversal
        let start = Instant::now();
        let mut neighbor_count = 0;
        for node in graph.iter_nodes_cache_friendly() {
            neighbor_count += graph.get_neighbors(node).len();
        }
        let traversal_time = start.elapsed();
        
        println!("Cache-friendly graph - nodes: {:?}, edges: {:?}, traversal: {:?}",
                node_creation_time, edge_creation_time, traversal_time);
        
        // Performance requirements
        assert!(node_creation_time.as_millis() < 100, "Node creation too slow: {:?}", node_creation_time);
        assert!(edge_creation_time.as_millis() < 500, "Edge creation too slow: {:?}", edge_creation_time);
        assert!(traversal_time.as_millis() < 50, "Traversal too slow: {:?}", traversal_time);
        assert!(neighbor_count > 0, "No neighbors found");
    }

    #[test]
    fn test_simd_vs_scalar_performance() {
        let vector_size = 1024;
        let iterations = 10000;
        
        let vec1: Vec<f32> = (0..vector_size).map(|i| (i as f32).sin()).collect();
        let vec2: Vec<f32> = (0..vector_size).map(|i| (i as f32).cos()).collect();
        
        // Scalar performance
        let start = Instant::now();
        for _ in 0..iterations {
            let _result = SimdOps::dot_product_scalar(&vec1, &vec2);
        }
        let scalar_time = start.elapsed();
        
        let scalar_per_op = scalar_time.as_micros() as f64 / iterations as f64;
        println!("Scalar dot product: {:.2}µs per operation", scalar_per_op);
        
        // SIMD performance (if available)
        #[cfg(feature = "similarity-search")]
        {
            let start = Instant::now();
            for _ in 0..iterations {
                let _result = SimdOps::dot_product_simd(&vec1, &vec2);
            }
            let simd_time = start.elapsed();
            
            let simd_per_op = simd_time.as_micros() as f64 / iterations as f64;
            let speedup = scalar_per_op / simd_per_op;
            
            println!("SIMD dot product: {:.2}µs per operation, speedup: {:.2}x", simd_per_op, speedup);
            
            // SIMD should provide some speedup
            assert!(speedup >= 1.2, "SIMD speedup insufficient: {:.2}x", speedup);
        }
        
        // Base performance requirement (relaxed for debug mode)
        assert!(scalar_per_op < 50.0, "Scalar operations too slow: {:.2}µs", scalar_per_op);
    }

    #[test]
    fn test_memory_usage_bounds() {
        let initial_memory = get_approximate_memory_usage();
        
        // Create large data structures
        let mut graph = ColumnGraph::new();
        let nodes: Vec<_> = (0..5000)
            .map(|i| graph.add_node(ColumnNode::new(&format!("node_{}", i), vec![i as f32; 50])))
            .collect();
        
        // Add edges
        for i in 0..5000 {
            for j in 0..5 {
                let target = (i + j * 1000) % 5000;
                if target != i {
                    graph.add_edge(nodes[i], nodes[target], 0.5);
                }
            }
        }
        
        let peak_memory = get_approximate_memory_usage();
        let memory_increase = peak_memory.saturating_sub(initial_memory);
        
        println!("Memory usage - initial: {} MB, peak: {} MB, increase: {} MB",
                initial_memory / 1024 / 1024,
                peak_memory / 1024 / 1024,
                memory_increase / 1024 / 1024);
        
        // Memory requirement: reasonable usage for 5000 nodes
        assert!(memory_increase < 1024 * 1024 * 1024, "Memory usage too high: {} MB", memory_increase / 1024 / 1024);
        
        // Test cleanup
        drop(graph);
        drop(nodes);
        
        // Allow some time for cleanup
        std::thread::sleep(std::time::Duration::from_millis(100));
        
        let final_memory = get_approximate_memory_usage();
        println!("Final memory: {} MB", final_memory / 1024 / 1024);
    }

    #[test]
    fn test_concurrent_performance() {
        use std::sync::Arc;
        use std::thread;
        
        let calc = Arc::new(SimilarityCalculator::new());
        let iterations_per_thread = 1000;
        let num_threads = 4;
        
        let vec1: Arc<Vec<f32>> = Arc::new((0..300).map(|i| i as f32 / 300.0).collect());
        let vec2: Arc<Vec<f32>> = Arc::new((0..300).map(|i| (300 - i) as f32 / 300.0).collect());
        
        let start = Instant::now();
        
        let handles: Vec<_> = (0..num_threads).map(|_| {
            let calc = Arc::clone(&calc);
            let vec1 = Arc::clone(&vec1);
            let vec2 = Arc::clone(&vec2);
            
            thread::spawn(move || {
                for _ in 0..iterations_per_thread {
                    let _similarity = calc.cosine_similarity(&vec1, &vec2);
                }
            })
        }).collect();
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        let elapsed = start.elapsed();
        let total_operations = num_threads * iterations_per_thread;
        let per_operation = elapsed.as_micros() as f64 / total_operations as f64;
        
        println!("Concurrent performance: {:.2}µs per operation ({} threads, {} ops/thread)",
                per_operation, num_threads, iterations_per_thread);
        
        // Should maintain good performance under concurrent load
        assert!(per_operation < 20.0, "Concurrent performance degradation: {:.2}µs", per_operation);
    }

    #[test]
    fn test_large_dataset_scalability() {
        // Test with progressively larger datasets
        for &size in &[100, 500, 1000, 2000] {
            let start = Instant::now();
            
            let mut graph = ColumnGraph::new();
            let nodes: Vec<_> = (0..size)
                .map(|i| graph.add_node(ColumnNode::new(&format!("node_{}", i), vec![i as f32; 10])))
                .collect();
            
            // Add sparse edges
            for i in 0..size {
                for j in 0..5.min(size - i - 1) {
                    let target = (i + j + 1) % size;
                    graph.add_edge(nodes[i], nodes[target], 0.5);
                }
            }
            
            let construction_time = start.elapsed();
            
            let start = Instant::now();
            let ranks = graph.calculate_pagerank(0.85, 50);
            let pagerank_time = start.elapsed();
            
            println!("Size {}: construction {:?}, PageRank {:?}",
                    size, construction_time, pagerank_time);
            
            assert_eq!(ranks.len(), size);
            
            // Scalability check: time should grow sub-quadratically
            let expected_max_construction = (size * size / 1000) as u128; // Rough heuristic
            let expected_max_pagerank = (size / 10) as u128; // Should be roughly linear
            
            assert!(construction_time.as_millis() < expected_max_construction.max(100),
                   "Construction doesn't scale for size {}: {}ms", size, construction_time.as_millis());
            assert!(pagerank_time.as_millis() < expected_max_pagerank.max(50),
                   "PageRank doesn't scale for size {}: {}ms", size, pagerank_time.as_millis());
        }
    }
}

fn get_approximate_memory_usage() -> usize {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb) = line.split_whitespace().nth(1).and_then(|s| s.parse::<usize>().ok()) {
                        return kb * 1024;
                    }
                }
            }
        }
    }
    
    // For other platforms, return a reasonable baseline
    100 * 1024 * 1024 // 100MB
}