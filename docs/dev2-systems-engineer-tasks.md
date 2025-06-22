# Developer 2: Systems Engineer Tasks

## Role Overview
Implement the graph-based ranking system, parallel processing pipeline, and performance optimizations for handling 20GB+ files with 1000+ columns.

## TDD Learning Requirements
Master these TDD patterns from the reference guide:
1. **Arrange-Act-Assert (AAA)**: Structure tests clearly
2. **Triangulation**: Use multiple cases to force general solutions
3. **Obvious Implementation**: When simple, just implement
4. **Refactoring patterns**: Extract method, remove duplication

## Sprint 1 Tasks (Days 1-5)

### Task 2.1: Graph Construction Foundation
**Story**: As a ranking system, I need to build similarity graphs from column relationships.

**TDD Tests First**:
```rust
#[test]
fn test_create_empty_graph() {
    let graph = ColumnGraph::new();
    assert_eq!(graph.node_count(), 0);
    assert_eq!(graph.edge_count(), 0);
}

#[test]
fn test_add_column_nodes() {
    let mut graph = ColumnGraph::new();
    let col1 = ColumnNode::new("description", vec![0.1, 0.2, 0.3]);
    let col2 = ColumnNode::new("comments", vec![0.2, 0.3, 0.4]);
    
    graph.add_node(col1);
    graph.add_node(col2);
    
    assert_eq!(graph.node_count(), 2);
}

#[test]
fn test_add_similarity_edge() {
    let mut graph = ColumnGraph::new();
    let node1 = graph.add_node(ColumnNode::new("col1", vec![1.0, 0.0]));
    let node2 = graph.add_node(ColumnNode::new("col2", vec![0.0, 1.0]));
    
    graph.add_edge(node1, node2, 0.5); // 50% similarity
    
    assert_eq!(graph.edge_count(), 1);
    assert_eq!(graph.get_edge_weight(node1, node2), Some(0.5));
}
```

**Implementation Steps**:
1. Create `ColumnGraph` struct using petgraph
2. Implement node addition (fake it first)
3. Add edge creation with weights
4. Implement adjacency queries
5. Refactor for performance

**Deliverables**:
- `column_graph.rs` with full test coverage
- Graph visualization utility
- Benchmark: <1ms for 1000-node graph creation

### Task 2.2: Similarity Computation
**Story**: As a graph builder, I need to compute similarity between columns efficiently.

**TDD Approach**:
```rust
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
fn test_jaccard_similarity() {
    let calc = SimilarityCalculator::new();
    let set1 = vec!["hello", "world"];
    let set2 = vec!["hello", "rust"];
    
    assert_eq!(calc.jaccard_similarity(&set1, &set2), 0.33, 0.01); // 1/3
}

#[test]
fn test_combined_similarity() {
    let calc = SimilarityCalculator::new();
    let col1 = ColumnData {
        embedding: vec![1.0, 0.0],
        tokens: vec!["hello", "world"],
    };
    let col2 = ColumnData {
        embedding: vec![0.7, 0.7],
        tokens: vec!["hello", "rust"],
    };
    
    let similarity = calc.combined_similarity(&col1, &col2, 0.6, 0.4);
    assert!(similarity > 0.0 && similarity < 1.0);
}
```

**Implementation**:
1. Cosine similarity for embeddings
2. Jaccard similarity for token sets
3. Weighted combination
4. SIMD optimization for vectors
5. Caching for repeated calculations

**Deliverables**:
- `similarity.rs` with optimized implementations
- Performance: <0.1ms per similarity calculation
- Accuracy validation against sklearn

### Task 2.3: K-NN Index for Fast Similarity Search
**Story**: As a graph builder, I need efficient k-nearest neighbor search for large column sets.

**TDD Tests**:
```rust
#[test]
fn test_build_faiss_index() {
    let index = FaissIndex::new(300); // 300-dim vectors
    let vectors = vec![
        vec![1.0; 300],
        vec![0.5; 300],
        vec![0.0; 300],
    ];
    
    index.add_batch(&vectors).unwrap();
    assert_eq!(index.size(), 3);
}

#[test]
fn test_knn_search() {
    let index = FaissIndex::new(2);
    index.add(&vec![0.0, 0.0]).unwrap();
    index.add(&vec![1.0, 0.0]).unwrap();
    index.add(&vec![0.0, 1.0]).unwrap();
    
    let query = vec![0.1, 0.1];
    let neighbors = index.search(&query, 2).unwrap();
    
    assert_eq!(neighbors.len(), 2);
    assert_eq!(neighbors[0].0, 0); // Closest is first point
}

#[test]
fn test_knn_performance() {
    let index = FaissIndex::new(300);
    let vectors: Vec<Vec<f32>> = (0..10000)
        .map(|_| (0..300).map(|_| rand::random()).collect())
        .collect();
    
    index.add_batch(&vectors).unwrap();
    
    let start = Instant::now();
    let _ = index.search(&vectors[0], 10);
    assert!(start.elapsed().as_micros() < 1000); // <1ms
}
```

**Deliverables**:
- FAISS integration for Rust
- Fallback pure-Rust implementation
- Benchmark: <1ms for 10k vectors

## Sprint 2 Tasks (Days 6-10)

### Task 2.4: PageRank Implementation
**Story**: As a ranking system, I need PageRank to identify central columns in the similarity graph.

**TDD First**:
```rust
#[test]
fn test_pagerank_simple_graph() {
    let mut graph = ColumnGraph::new();
    let a = graph.add_node("A");
    let b = graph.add_node("B");
    let c = graph.add_node("C");
    
    graph.add_edge(a, b, 1.0);
    graph.add_edge(b, c, 1.0);
    graph.add_edge(c, a, 1.0);
    
    let ranks = graph.calculate_pagerank(0.85, 100);
    
    // All nodes should have equal rank in a cycle
    assert!((ranks[&a] - 0.33).abs() < 0.01);
    assert!((ranks[&b] - 0.33).abs() < 0.01);
    assert!((ranks[&c] - 0.33).abs() < 0.01);
}

#[test]
fn test_pagerank_hub_node() {
    let mut graph = ColumnGraph::new();
    let hub = graph.add_node("hub");
    let nodes: Vec<_> = (0..5).map(|i| graph.add_node(&format!("node{}", i))).collect();
    
    // All nodes point to hub
    for &node in &nodes {
        graph.add_edge(node, hub, 1.0);
    }
    
    let ranks = graph.calculate_pagerank(0.85, 100);
    
    // Hub should have highest rank
    assert!(ranks[&hub] > ranks[&nodes[0]] * 2.0);
}
```

**Implementation**:
1. Power iteration algorithm
2. Damping factor handling
3. Convergence detection
4. Sparse matrix optimization
5. Parallel computation

**Deliverables**:
- PageRank implementation
- Convergence analysis
- Performance: <100ms for 1000 nodes

### Task 2.5: Parallel Processing Pipeline
**Story**: As a system, I need to process multiple columns in parallel while maintaining consistency.

**TDD Tests**:
```rust
#[test]
async fn test_parallel_column_processor() {
    let processor = ParallelProcessor::new(4); // 4 workers
    let columns = vec!["col1", "col2", "col3", "col4"];
    
    let results = processor.process_columns(columns, |col| {
        // Simulate work
        thread::sleep(Duration::from_millis(10));
        format!("{}_processed", col)
    }).await;
    
    assert_eq!(results.len(), 4);
    assert!(results.contains(&"col1_processed".to_string()));
}

#[test]
async fn test_work_stealing() {
    let processor = ParallelProcessor::new(2);
    let work_items: Vec<_> = (0..100).collect();
    
    let start = Instant::now();
    let results = processor.process_balanced(work_items, |item| {
        // Uneven work
        if item % 10 == 0 {
            thread::sleep(Duration::from_millis(50));
        }
        item * 2
    }).await;
    
    // Should complete faster than sequential
    assert!(start.elapsed().as_millis() < 2500); // Not 5000ms sequential
    assert_eq!(results.len(), 100);
}

#[test]
async fn test_checkpoint_recovery() {
    let processor = ParallelProcessor::new(2);
    let checkpoint = Checkpoint::new("test_run");
    
    // Simulate failure at item 50
    let work = (0..100).collect();
    processor.process_with_checkpoint(work, &checkpoint, |item| {
        if item == 50 {
            panic!("Simulated failure");
        }
        item * 2
    }).await.unwrap_err();
    
    // Resume from checkpoint
    let resumed = processor.resume_from_checkpoint(&checkpoint).await.unwrap();
    assert_eq!(resumed.completed_items, 50);
}
```

**Implementation**:
1. Work-stealing thread pool
2. Checkpoint mechanism
3. Progress tracking
4. Error recovery
5. Resource management

**Deliverables**:
- `parallel_processor.rs`
- Checkpoint system
- Throughput: 1M rows/minute

### Task 2.6: Streaming Architecture
**Story**: As a system, I need to process 20GB+ files without loading them into memory.

**TDD Approach**:
```rust
#[test]
async fn test_streaming_reader() {
    let file = create_test_file_gb(1); // 1GB test file
    let reader = StreamingReader::new(file, 100 * MB);
    
    let mut chunks = 0;
    let mut total_rows = 0;
    
    while let Some(chunk) = reader.next_chunk().await.unwrap() {
        chunks += 1;
        total_rows += chunk.row_count();
        assert!(chunk.memory_usage() <= 100 * MB);
    }
    
    assert!(chunks >= 10); // At least 10 chunks for 1GB
    assert_eq!(total_rows, expected_rows);
}

#[test]
async fn test_memory_bounded_processing() {
    let processor = MemoryBoundedProcessor::new(500 * MB);
    let large_file = create_test_file_gb(5);
    
    let monitor = MemoryMonitor::new();
    let result = processor.process_file(large_file, |chunk| {
        assert!(monitor.current_usage() <= 600 * MB); // Some overhead
        chunk.transform()
    }).await;
    
    assert!(result.is_ok());
}
```

**Deliverables**:
- Streaming CSV reader
- Memory-bounded processing
- Zero-copy optimizations

## Sprint 3 Tasks (Days 11-15)

### Task 2.7: Performance Optimization
**Story**: As a developer, I need the system optimized for production workloads.

**Performance Tests**:
```rust
#[bench]
fn bench_graph_construction_1000_columns(b: &mut Bencher) {
    let columns = generate_test_columns(1000);
    b.iter(|| {
        let graph = GraphRanker::build_graph(&columns);
        black_box(graph);
    });
}

#[bench]
fn bench_pagerank_convergence(b: &mut Bencher) {
    let graph = create_test_graph(1000);
    b.iter(|| {
        let ranks = graph.calculate_pagerank(0.85, 100);
        black_box(ranks);
    });
}

#[test]
fn test_simd_similarity() {
    let vec1 = vec![1.0; 1024];
    let vec2 = vec![0.5; 1024];
    
    let scalar_result = cosine_similarity_scalar(&vec1, &vec2);
    let simd_result = cosine_similarity_simd(&vec1, &vec2);
    
    assert!((scalar_result - simd_result).abs() < 0.0001);
    
    // Benchmark
    let start = Instant::now();
    for _ in 0..10000 {
        cosine_similarity_simd(&vec1, &vec2);
    }
    let simd_time = start.elapsed();
    
    let start = Instant::now();
    for _ in 0..10000 {
        cosine_similarity_scalar(&vec1, &vec2);
    }
    let scalar_time = start.elapsed();
    
    assert!(simd_time < scalar_time / 2); // At least 2x faster
}
```

**Optimization Areas**:
1. SIMD for vector operations
2. Cache-friendly data layouts
3. Lock-free data structures
4. Memory pool allocation
5. Compiler optimizations

### Task 2.8: Integration and Load Testing
**Story**: As a team, we need comprehensive system tests under production loads.

**Load Tests**:
```rust
#[test]
async fn test_concurrent_graph_operations() {
    let graph = Arc::new(RwLock::new(ColumnGraph::new()));
    let mut handles = vec![];
    
    // 100 concurrent operations
    for i in 0..100 {
        let graph_clone = graph.clone();
        let handle = tokio::spawn(async move {
            if i % 2 == 0 {
                // Reader
                let g = graph_clone.read().await;
                g.calculate_pagerank(0.85, 10)
            } else {
                // Writer
                let mut g = graph_clone.write().await;
                g.add_node(format!("node_{}", i))
            }
        });
        handles.push(handle);
    }
    
    for handle in handles {
        assert!(handle.await.is_ok());
    }
}

#[test]
async fn test_50gb_file_processing() {
    let large_file = create_test_file_gb(50);
    let processor = StreamingProcessor::new();
    
    let start = Instant::now();
    let result = processor.process_file(large_file).await.unwrap();
    let elapsed = start.elapsed();
    
    assert_eq!(result.rows_processed, 500_000_000); // Expected rows
    assert!(elapsed.as_secs() < 3600); // Under 1 hour
    
    // Memory never exceeded 1GB
    assert!(result.peak_memory_mb < 1024);
}
```

## Performance Requirements
- Graph construction: <5 seconds for 1000 columns
- PageRank: <100ms for 1000 nodes
- Similarity search: <1ms per query
- Stream processing: 100MB/s throughput
- Memory usage: <1GB for any file size

## Testing Requirements
- Unit test coverage: 90%+
- Integration tests for all components
- Load tests with production data sizes
- Chaos testing for failure scenarios
- Performance regression suite

## Dependencies
- petgraph for graph algorithms
- faiss for similarity search
- rayon for CPU parallelism
- tokio for async I/O
- SIMD intrinsics for optimization

## Monitoring Metrics
```rust
pub struct SystemMetrics {
    pub graph_build_time_ms: Histogram,
    pub pagerank_iterations: Gauge,
    pub similarity_cache_hits: Counter,
    pub parallel_efficiency: Gauge,
    pub memory_usage_mb: Gauge,
    pub checkpoint_saves: Counter,
}
```

## TDD Best Practices for Systems
1. **Test performance early** - Include benchmarks in TDD cycle
2. **Test concurrency** - Multi-threaded tests from the start
3. **Test failure modes** - Panic recovery, OOM handling
4. **Test at scale** - Use property-based testing for large inputs
5. **Profile under test** - Catch performance regressions

## Architecture Decisions
1. **Graph Library**: petgraph for flexibility
2. **Parallelism**: Rayon for CPU, Tokio for I/O
3. **Memory Management**: Explicit bounds and monitoring
4. **Checkpointing**: Write-ahead log pattern
5. **Optimization**: SIMD where beneficial

## Definition of Done
- [ ] All unit tests passing
- [ ] Integration tests passing
- [ ] Load tests meeting SLAs
- [ ] Memory usage within bounds
- [ ] Documentation complete
- [ ] Performance benchmarks recorded
- [ ] Monitoring instrumented
- [ ] Chaos tests passing