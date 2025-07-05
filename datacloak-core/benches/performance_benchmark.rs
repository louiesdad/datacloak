use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use datacloak_core::graph::{ColumnGraph, ColumnNode, SimilarityCalculator};
use datacloak_core::performance::{CacheFriendlyGraph, MemoryPool, SimdOps};
use std::time::Instant;

fn bench_memory_pool(c: &mut Criterion) {
    c.bench_function("memory_pool_allocation", |b| {
        let pool = MemoryPool::new(100 * 1024 * 1024); // 100MB pool
        b.iter(|| {
            let _allocation = pool.allocate::<f32>(black_box(1000)).unwrap();
        });
    });

    c.bench_function("memory_pool_vs_std_allocation", |b| {
        let pool = MemoryPool::new(100 * 1024 * 1024);
        b.iter_batched(
            || (),
            |_| {
                let _pool_alloc = pool.allocate::<f32>(1000).unwrap();
                let _std_alloc: Vec<f32> = vec![0.0; 1000];
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_cache_friendly_graph(c: &mut Criterion) {
    c.bench_function("cache_friendly_graph_construction", |b| {
        b.iter_batched(
            || (),
            |_| {
                let graph = CacheFriendlyGraph::new();
                let nodes: Vec<_> = (0..1000)
                    .map(|i| graph.add_node(format!("col_{}", i), vec![i as f32; 100]))
                    .collect();

                for i in 0..1000 {
                    for j in i + 1..i + 10.min(1000) {
                        if j < 1000 {
                            graph.add_edge(nodes[i], nodes[j], 0.5);
                        }
                    }
                }
                black_box(graph);
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("cache_friendly_traversal", |b| {
        let graph = CacheFriendlyGraph::new();
        let nodes: Vec<_> = (0..1000)
            .map(|i| graph.add_node(format!("col_{}", i), vec![i as f32; 100]))
            .collect();

        for i in 0..1000 {
            for j in i + 1..i + 10.min(1000) {
                if j < 1000 {
                    graph.add_edge(nodes[i], nodes[j], 0.5);
                }
            }
        }

        b.iter(|| {
            let mut neighbor_count = 0;
            for node in graph.iter_nodes_cache_friendly() {
                neighbor_count += graph.get_neighbors(black_box(node)).len();
            }
            black_box(neighbor_count);
        });
    });
}

fn bench_simd_operations(c: &mut Criterion) {
    c.bench_function("simd_dot_product_scalar", |b| {
        let a = vec![1.0f32; 1024];
        let vec_b = vec![2.0f32; 1024];
        b.iter(|| SimdOps::dot_product_scalar(black_box(&a), black_box(&vec_b)));
    });

    #[cfg(feature = "similarity-search")]
    c.bench_function("simd_dot_product_simd", |b| {
        let a = vec![1.0f32; 1024];
        let vec_b = vec![2.0f32; 1024];
        b.iter(|| SimdOps::dot_product_simd(black_box(&a), black_box(&vec_b)));
    });

    c.bench_function("simd_vector_normalization", |b| {
        b.iter_batched(
            || vec![3.0f32; 1024],
            |mut vec| {
                SimdOps::normalize_vector(&mut vec);
                black_box(vec);
            },
            BatchSize::SmallInput,
        );
    });

    #[cfg(feature = "similarity-search")]
    c.bench_function("simd_vector_normalization_simd", |b| {
        b.iter_batched(
            || vec![3.0f32; 1024],
            |mut vec| {
                SimdOps::normalize_vector_simd(&mut vec);
                black_box(vec);
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_batch_operations(c: &mut Criterion) {
    c.bench_function("batch_cosine_similarity", |b| {
        let query = vec![1.0f32; 300];
        let targets: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32 / 100.0; 300]).collect();

        b.iter(|| SimdOps::batch_cosine_similarity(black_box(&query), black_box(&targets), 8));
    });

    c.bench_function("sequential_cosine_similarity", |b| {
        let calc = SimilarityCalculator::new();
        let query = vec![1.0f32; 300];
        let targets: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32 / 100.0; 300]).collect();

        b.iter(|| {
            let results: Vec<f32> = targets
                .iter()
                .map(|target| calc.cosine_similarity(black_box(&query), black_box(target)))
                .collect();
            black_box(results);
        });
    });
}

fn bench_graph_comparison(c: &mut Criterion) {
    c.bench_function("standard_graph_construction", |b| {
        b.iter_batched(
            || (),
            |_| {
                let mut graph = ColumnGraph::new();
                let nodes: Vec<_> = (0..1000)
                    .map(|i| {
                        graph.add_node(ColumnNode::new(&format!("col_{}", i), vec![i as f32; 100]))
                    })
                    .collect();

                for i in 0..1000 {
                    for j in i + 1..i + 10.min(1000) {
                        if j < 1000 {
                            graph.add_edge(nodes[i], nodes[j], 0.5);
                        }
                    }
                }
                black_box(graph);
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("cache_friendly_graph_construction", |b| {
        b.iter_batched(
            || (),
            |_| {
                let graph = CacheFriendlyGraph::new();
                let nodes: Vec<_> = (0..1000)
                    .map(|i| graph.add_node(format!("col_{}", i), vec![i as f32; 100]))
                    .collect();

                for i in 0..1000 {
                    for j in i + 1..i + 10.min(1000) {
                        if j < 1000 {
                            graph.add_edge(nodes[i], nodes[j], 0.5);
                        }
                    }
                }
                black_box(graph);
            },
            BatchSize::SmallInput,
        );
    });
}

fn performance_regression_test() {
    println!("\n=== Performance Regression Test Suite ===");

    // Graph construction performance
    let start = Instant::now();
    let mut graph = ColumnGraph::new();
    let nodes: Vec<_> = (0..1000)
        .map(|i| graph.add_node(ColumnNode::new(&format!("col_{}", i), vec![i as f32; 100])))
        .collect();

    for i in 0..1000 {
        for j in i + 1..i + 10.min(1000) {
            if j < 1000 {
                graph.add_edge(nodes[i], nodes[j], 0.5);
            }
        }
    }
    let graph_time = start.elapsed();
    println!("Graph construction (1000 nodes): {:?}", graph_time);
    assert!(graph_time.as_millis() < 100, "Graph construction too slow");

    // PageRank performance
    let start = Instant::now();
    let _ranks = graph.calculate_pagerank(0.85, 100);
    let pagerank_time = start.elapsed();
    println!("PageRank (1000 nodes): {:?}", pagerank_time);
    assert!(pagerank_time.as_millis() < 100, "PageRank too slow");

    // Similarity computation performance
    let calc = SimilarityCalculator::new();
    let vec1 = vec![1.0f32; 1024];
    let vec2 = vec![0.5f32; 1024];

    let start = Instant::now();
    for _ in 0..10000 {
        calc.cosine_similarity(&vec1, &vec2);
    }
    let similarity_time = start.elapsed();
    let per_similarity = similarity_time.as_micros() as f64 / 10000.0;
    println!("Cosine similarity: {:.2}µs per calculation", per_similarity);
    assert!(per_similarity < 100.0, "Similarity calculation too slow");

    // Memory pool performance
    let pool = MemoryPool::new(100 * 1024 * 1024);
    let start = Instant::now();
    let mut allocations = vec![];
    for i in 0..1000 {
        if let Ok(alloc) = pool.allocate::<f32>(100 + i) {
            allocations.push(alloc);
        }
    }
    let pool_time = start.elapsed();
    println!("Memory pool allocations: {:?}", pool_time);
    println!("Successful allocations: {}", allocations.len());

    println!("=== All performance tests passed! ===");
}

criterion_group!(
    benches,
    bench_memory_pool,
    bench_cache_friendly_graph,
    bench_simd_operations,
    bench_batch_operations,
    bench_graph_comparison
);

criterion_main!(benches);
