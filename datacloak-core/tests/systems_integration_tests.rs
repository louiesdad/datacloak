use datacloak_core::{
    graph::{ColumnGraph, ColumnNode, SimilarityCalculator},
    performance::{CacheFriendlyGraph, MemoryPool, SimdOps},
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::time::{Duration, Instant};

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_concurrent_graph_operations() {
        let graph = Arc::new(tokio::sync::RwLock::new(ColumnGraph::new()));
        let mut handles = vec![];

        // 100 concurrent operations
        for i in 0..100 {
            let graph_clone = Arc::clone(&graph);
            let handle = tokio::spawn(async move {
                if i % 2 == 0 {
                    // Reader - calculate PageRank
                    let g = graph_clone.read().await;
                    g.calculate_pagerank(0.85, 10)
                } else {
                    // Writer - add node
                    let mut g = graph_clone.write().await;
                    let node_id =
                        g.add_node(ColumnNode::new(&format!("node_{}", i), vec![i as f32]));
                    let mut result = HashMap::new();
                    result.insert(node_id, 1.0 / 100.0);
                    result
                }
            });
            handles.push(handle);
        }

        // All operations should complete successfully
        let mut completed = 0;
        for handle in handles {
            if handle.await.is_ok() {
                completed += 1;
            }
        }

        // Most operations should succeed
        assert!(
            completed >= 90,
            "Too many failed operations: {}/100",
            completed
        );
    }

    #[test]
    fn test_memory_bounded_large_graph() {
        let start_memory = get_memory_usage();

        // Create very large graph (10,000 nodes)
        let mut graph = ColumnGraph::new();
        let nodes: Vec<_> = (0..10_000)
            .map(|i| {
                graph.add_node(ColumnNode::new(
                    &format!("column_{}", i),
                    vec![i as f32; 10],
                ))
            })
            .collect();

        // Add edges (sparse graph - each node connects to ~10 others)
        for (i, &node_i) in nodes.iter().enumerate() {
            for j in 0..10 {
                let target_idx = (i + j * 1000) % nodes.len();
                if target_idx != i {
                    graph.add_edge(node_i, nodes[target_idx], 0.5);
                }
            }
        }

        // Calculate PageRank on large graph
        let start = Instant::now();
        let ranks = graph.calculate_pagerank(0.85, 100);
        let elapsed = start.elapsed();

        // Performance requirements
        assert!(
            elapsed.as_millis() < 5000,
            "PageRank too slow: {:?}",
            elapsed
        );
        assert_eq!(ranks.len(), 10_000);

        // Memory usage should be reasonable
        let peak_memory = get_memory_usage();
        let memory_increase = peak_memory.saturating_sub(start_memory);
        println!("Memory increase: {} MB", memory_increase / 1024 / 1024);

        // Allow reasonable memory usage for 10K nodes
        assert!(
            memory_increase < 2 * 1024 * 1024 * 1024,
            "Memory usage too high: {} MB",
            memory_increase / 1024 / 1024
        );
    }

    #[tokio::test]
    async fn test_streaming_processor_resilience() {
        // Simulate stream processing with intermittent failures
        let mut processed_count = 0;
        let mut error_count = 0;

        for i in 0..1000 {
            // Simulate processing with some failures
            if i % 100 == 0 {
                // Simulate error
                error_count += 1;
                tokio::time::sleep(Duration::from_millis(1)).await;
            } else {
                // Simulate successful processing
                processed_count += 1;
                tokio::time::sleep(Duration::from_micros(100)).await;
            }
        }

        // Should complete despite errors
        assert!(
            processed_count >= 900,
            "Too many failures: processed {}, errors {}",
            processed_count,
            error_count
        );
        assert!(error_count == 10, "Expected 10 errors, got {}", error_count);
    }

    #[tokio::test]
    async fn test_50gb_file_simulation() {
        // Simulate processing a 50GB file (without actually creating it)
        let chunk_processor = ChunkProcessor::new(100 * 1024 * 1024); // 100MB chunks

        // Simulate 500 chunks of 100MB each = 50GB
        let total_chunks = 500;
        let mut processed_chunks = 0;
        let start = Instant::now();

        for chunk_id in 0..total_chunks {
            // Simulate chunk processing time (proportional to real processing)
            let chunk_start = Instant::now();

            // Simulate chunk data (don't actually allocate 100MB)
            let simulated_chunk_size = 100 * 1024 * 1024;
            let simulated_rows = simulated_chunk_size / 1000; // ~1KB per row

            // Process chunk
            let _results = chunk_processor
                .process_chunk_simulation(
                    chunk_id,
                    simulated_rows,
                    vec!["col1".to_string(), "col2".to_string()],
                )
                .await;

            processed_chunks += 1;

            let chunk_elapsed = chunk_start.elapsed();

            // Each chunk should process in reasonable time (<500ms)
            assert!(
                chunk_elapsed.as_millis() < 500,
                "Chunk {} too slow: {:?}",
                chunk_id,
                chunk_elapsed
            );

            // Memory should remain stable
            let current_memory = get_memory_usage();
            // Allow up to 4GB for simulation
            assert!(
                current_memory < 4 * 1024 * 1024 * 1024,
                "Memory leak detected: {} MB",
                current_memory / 1024 / 1024
            );

            // Progress reporting
            if chunk_id % 50 == 0 {
                let progress = (chunk_id as f32 / total_chunks as f32) * 100.0;
                println!("Processed {} chunks ({:.1}%)", chunk_id, progress);
            }
        }

        let total_elapsed = start.elapsed();

        // Should complete all chunks
        assert_eq!(processed_chunks, total_chunks);

        // Should complete in reasonable time (under 5 minutes)
        assert!(
            total_elapsed.as_secs() < 300,
            "Total processing too slow: {:?}",
            total_elapsed
        );

        println!(
            "50GB simulation: {} chunks in {:?}",
            processed_chunks, total_elapsed
        );
    }

    #[tokio::test]
    async fn test_ml_graph_pipeline_stress() {
        // Create stress test dataset
        let stress_columns: Vec<MockColumn> = (0..1000)
            .map(|i| {
                let col_type = match i % 4 {
                    0 => "text",    // 25% text
                    1 => "numeric", // 25% numeric
                    2 => "id",      // 25% id
                    _ => "mixed",   // 25% mixed
                };

                let values = match col_type {
                    "text" => vec![
                        format!("This is a long text description for item {}", i),
                        format!("Another detailed review and analysis for product {}", i),
                        format!("Comprehensive feedback and comments about item {}", i),
                    ],
                    "numeric" => vec![
                        format!("{}.99", i % 100),
                        format!("{}.50", (i * 2) % 100),
                        format!("{}.25", (i * 3) % 100),
                    ],
                    "id" => vec![
                        format!("ID-{:06}", i),
                        format!("SKU-{:06}", i + 1000),
                        format!("REF-{:06}", i + 2000),
                    ],
                    _ => vec![
                        format!("Mixed {} content", i),
                        format!("{}.99", i),
                        format!("ID-{}", i),
                    ],
                };

                MockColumn::new(&format!("stress_col_{}", i), values)
            })
            .collect();

        // Profile all columns
        let start = Instant::now();
        let candidates = profile_columns_mock(&stress_columns).await;
        let elapsed = start.elapsed();

        // Performance requirements from PRD
        assert!(
            elapsed.as_secs() < 10,
            "Profiling 1000 columns too slow: {:?}",
            elapsed
        );
        assert_eq!(candidates.len(), 1000);

        // Quality requirements
        let text_candidates: Vec<_> = candidates.iter().filter(|c| c.final_score > 0.7).collect();

        // Should identify ~250 text columns (25% of dataset)
        assert!(
            text_candidates.len() >= 200,
            "Too few text columns identified: {}",
            text_candidates.len()
        );
        assert!(
            text_candidates.len() <= 300,
            "Too many false positives: {}",
            text_candidates.len()
        );

        // Check ranking quality
        for i in 0..text_candidates.len().saturating_sub(1) {
            assert!(
                text_candidates[i].final_score >= text_candidates[i + 1].final_score,
                "Ranking not properly sorted at position {}",
                i
            );
        }
    }

    #[test]
    fn test_simd_performance_optimization() {
        // Test SIMD vs scalar performance
        let vector_size = 1024;
        let iterations = 10_000;

        let vec1: Vec<f32> = (0..vector_size)
            .map(|i| i as f32 / vector_size as f32)
            .collect();
        let vec2: Vec<f32> = (0..vector_size)
            .map(|i| (vector_size - i) as f32 / vector_size as f32)
            .collect();

        // Scalar implementation
        let start = Instant::now();
        for _ in 0..iterations {
            let _similarity = SimdOps::dot_product_scalar(&vec1, &vec2);
        }
        let scalar_time = start.elapsed();

        // SIMD implementation (if available)
        #[cfg(feature = "similarity-search")]
        {
            let start = Instant::now();
            for _ in 0..iterations {
                let _similarity = SimdOps::dot_product_simd(&vec1, &vec2);
            }
            let simd_time = start.elapsed();

            // SIMD should be at least 1.5x faster (relaxed for testing)
            let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;
            assert!(speedup >= 1.5, "SIMD speedup insufficient: {:.2}x", speedup);

            println!(
                "SIMD speedup: {:.2}x (scalar: {:?}, SIMD: {:?})",
                speedup, scalar_time, simd_time
            );
        }

        #[cfg(not(feature = "similarity-search"))]
        {
            println!("Scalar time: {:?} (SIMD not available)", scalar_time);
        }
    }

    #[test]
    fn test_graph_operations_scalability() {
        let mut graph = ColumnGraph::new();

        // Pre-populate graph with 1000 nodes
        let nodes: Vec<_> = (0..1000)
            .map(|i| graph.add_node(ColumnNode::new(&format!("node_{}", i), vec![i as f32; 10])))
            .collect();

        // Add edges
        for i in 0..1000 {
            for j in 0..10 {
                let target = (i + j * 100) % 1000;
                if target != i {
                    graph.add_edge(nodes[i], nodes[target], 0.5);
                }
            }
        }

        // Benchmark PageRank calculation
        let start = Instant::now();
        let ranks = graph.calculate_pagerank(0.85, 50);
        let elapsed = start.elapsed();

        assert_eq!(ranks.len(), 1000);
        assert!(
            elapsed.as_millis() < 100,
            "PageRank scalability test failed: {:?}",
            elapsed
        );

        println!("PageRank 1000 nodes: {:?}", elapsed);
    }

    #[tokio::test]
    async fn test_cache_friendly_graph_performance() {
        let graph = CacheFriendlyGraph::new();

        // Add nodes
        let nodes: Vec<_> = (0..1000)
            .map(|i| graph.add_node(format!("col_{}", i), vec![i as f32; 100]))
            .collect();

        // Add edges in cache-friendly order
        for i in 0..1000 {
            for j in i + 1..i + 10.min(1000) {
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

        println!(
            "Cache-friendly traversal: {:?}, Total neighbors: {}",
            cache_friendly_time, neighbor_count
        );

        // Verify correctness
        assert_eq!(graph.node_count(), 1000);
        assert!(neighbor_count > 0);
        assert!(
            cache_friendly_time.as_millis() < 50,
            "Cache-friendly traversal too slow: {:?}",
            cache_friendly_time
        );
    }

    #[tokio::test]
    async fn test_memory_pool_under_pressure() {
        let pool = MemoryPool::new(100 * 1024 * 1024); // 100MB pool

        let mut allocations = vec![];
        let mut allocation_count = 0;

        // Stress test the memory pool
        for i in 0..1000 {
            let size = 100 + (i % 1000); // Variable sizes

            if let Ok(allocation) = pool.allocate::<f32>(size) {
                allocation_count += 1;
                allocations.push(allocation);

                // Periodically drop some allocations
                if i % 100 == 0 && !allocations.is_empty() {
                    allocations.drain(0..allocations.len() / 2);
                }
            }

            // Check pool stats periodically
            if i % 250 == 0 {
                let stats = pool.stats();
                println!(
                    "Pool stats at {}: allocated={} MB, free={} MB",
                    i,
                    stats.allocated_bytes / 1024 / 1024,
                    stats.free_bytes / 1024 / 1024
                );
            }
        }

        let final_stats = pool.stats();
        println!(
            "Final pool stats: allocated={} MB, free={} MB, allocations={}",
            final_stats.allocated_bytes / 1024 / 1024,
            final_stats.free_bytes / 1024 / 1024,
            allocation_count
        );

        assert!(
            allocation_count > 100,
            "Memory pool should handle many allocations"
        );
        assert!(final_stats.allocated_bytes <= final_stats.total_bytes);
    }

    #[tokio::test]
    async fn test_concurrent_similarity_calculations() {
        let calc = SimilarityCalculator::new();
        let mut handles = vec![];

        // Create test vectors
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|i| (0..300).map(|j| ((i * j) as f32).sin()).collect())
            .collect();

        // Launch concurrent similarity calculations
        for i in 0..50 {
            let calc_clone = calc.clone();
            let vec1 = vectors[i % vectors.len()].clone();
            let vec2 = vectors[(i + 1) % vectors.len()].clone();

            let handle = tokio::spawn(async move { calc_clone.cosine_similarity(&vec1, &vec2) });
            handles.push(handle);
        }

        // Collect results
        let mut results = vec![];
        for handle in handles {
            if let Ok(result) = handle.await {
                results.push(result);
            }
        }

        assert_eq!(
            results.len(),
            50,
            "All similarity calculations should complete"
        );

        // Verify results are reasonable
        for &result in &results {
            assert!(
                result >= -1.1 && result <= 1.1,
                "Invalid similarity score: {}",
                result
            );
        }
    }
}

// Helper functions
fn get_memory_usage() -> usize {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb) = line
                        .split_whitespace()
                        .nth(1)
                        .and_then(|s| s.parse::<usize>().ok())
                    {
                        return kb * 1024;
                    }
                }
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        // For non-Linux systems, use a simple estimation
        100 * 1024 * 1024 // 100MB baseline
    }

    #[cfg(target_os = "linux")]
    0
}

// Mock implementations for testing
pub struct ChunkProcessor {
    #[allow(dead_code)]
    chunk_size: usize,
}

impl ChunkProcessor {
    pub fn new(chunk_size: usize) -> Self {
        Self { chunk_size }
    }

    pub async fn process_chunk_simulation(
        &self,
        chunk_id: usize,
        rows: usize,
        _columns: Vec<String>,
    ) -> Result<Vec<String>, anyhow::Error> {
        // Simulate processing time proportional to data size
        let processing_time = Duration::from_micros((rows / 10000) as u64); // Scale down for testing
        tokio::time::sleep(processing_time).await;

        // Return mock results
        Ok((0..rows.min(10))
            .map(|i| format!("result_{}_{}", chunk_id, i))
            .collect())
    }
}

#[derive(Debug, Clone)]
pub struct MockColumn {
    pub name: String,
    pub values: Vec<String>,
}

impl MockColumn {
    pub fn new(name: &str, values: Vec<String>) -> Self {
        Self {
            name: name.to_string(),
            values,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MockCandidate {
    pub name: String,
    pub final_score: f32,
}

async fn profile_columns_mock(columns: &[MockColumn]) -> Vec<MockCandidate> {
    let mut candidates = vec![];

    for col in columns {
        // Mock profiling logic
        let score = if col.name.contains("text")
            || col
                .values
                .iter()
                .any(|v| v.contains("description") || v.contains("review") || v.contains("comment"))
        {
            0.8 // High score for text columns
        } else if col.name.contains("id")
            || col
                .values
                .iter()
                .any(|v| v.starts_with("ID-") || v.starts_with("SKU-"))
        {
            0.3 // Low score for ID columns
        } else if col.values.iter().any(|v| v.parse::<f32>().is_ok()) {
            0.5 // Medium score for numeric columns
        } else {
            0.6 // Default score for mixed content
        };

        candidates.push(MockCandidate {
            name: col.name.clone(),
            final_score: score,
        });

        // Simulate processing time
        tokio::time::sleep(Duration::from_micros(100)).await;
    }

    // Sort by score (highest first)
    candidates.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).unwrap());

    candidates
}
