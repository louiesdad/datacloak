# Integration Guide: Graph Components

## Overview
Dev 2 has completed the graph construction foundation and similarity computation components. This guide helps other developers integrate with these components.

## Available Components from Dev 2

### 1. ColumnGraph (in `datacloak-core/src/graph/`)
```rust
use datacloak_core::graph::{ColumnGraph, ColumnNode};

// Create a graph
let mut graph = ColumnGraph::new();

// Add nodes
let node1 = graph.add_node(ColumnNode::new("description", embedding_vec));
let node2 = graph.add_node(ColumnNode::new("comments", embedding_vec));

// Add similarity edge
graph.add_edge(node1, node2, 0.85); // 85% similarity

// Query the graph
let neighbors = graph.get_neighbors(node1);
let edge_weight = graph.get_edge_weight(node1, node2);
```

### 2. SimilarityCalculator (in `datacloak-core/src/graph/similarity.rs`)
```rust
use datacloak_core::graph::SimilarityCalculator;

let calculator = SimilarityCalculator::new();

// Cosine similarity for embeddings
let similarity = calculator.cosine_similarity(&vec1, &vec2);

// Jaccard similarity for token sets
let similarity = calculator.jaccard_similarity(&tokens1, &tokens2);

// Combined similarity (weighted)
let combined = calculator.combined_similarity(
    &col_data1,
    &col_data2,
    0.6, // embedding weight
    0.4  // token weight
);
```

## Integration Points

### For Dev 1 (ML Engineer)
When implementing the ML+Graph profiler:

```rust
// After ML classification
let ml_predictions = classifier.predict_batch(&columns);

// Build similarity graph using Dev 2's components
let mut graph = ColumnGraph::new();
let similarity_calc = SimilarityCalculator::new();

// Add text-heavy columns to graph
for prediction in ml_predictions.iter().filter(|p| p.is_text_probable()) {
    let node = graph.add_node(ColumnNode {
        name: prediction.column_name.clone(),
        embedding: prediction.embedding.clone(),
        ml_score: prediction.probability,
    });
}

// Calculate similarities and add edges
for (i, node_i) in graph.nodes().enumerate() {
    for (j, node_j) in graph.nodes().enumerate().skip(i + 1) {
        let similarity = similarity_calc.combined_similarity(
            &node_i.data,
            &node_j.data,
            0.6,
            0.4
        );
        
        if similarity > 0.3 { // threshold
            graph.add_edge(i, j, similarity);
        }
    }
}

// Use graph for ranking (PageRank will be implemented by Dev 2 in Task 2.4)
```

### For Dev 3 (Backend Engineer)
When implementing the profile endpoint:

```rust
use datacloak_core::graph::{ColumnGraph, SimilarityCalculator};
use datacloak_core::ml_classifier::MLClassifier;

#[post("/api/v2/profile")]
async fn profile_columns(
    file_id: web::Json<FileId>,
    ml_classifier: web::Data<MLClassifier>,
    graph_builder: web::Data<GraphBuilder>,
) -> Result<HttpResponse> {
    // 1. Extract sample
    let sample = extract_sample(&file_id.path).await?;
    
    // 2. ML classification (Dev 1's component)
    let ml_predictions = ml_classifier.predict_batch(&sample).await?;
    
    // 3. Build similarity graph (Dev 2's component)
    let graph = graph_builder.build_from_predictions(&ml_predictions).await?;
    
    // 4. Calculate rankings (PageRank pending from Dev 2)
    // For now, use simple degree-based ranking
    let rankings = calculate_simple_rankings(&graph);
    
    // 5. Combine scores
    let candidates = merge_ml_and_graph_scores(ml_predictions, rankings);
    
    Ok(HttpResponse::Ok().json(ProfileResponse { candidates }))
}
```

### For Dev 4 (Integration Engineer)
When testing the integrated system:

```rust
#[test]
fn test_graph_similarity_accuracy() {
    let calc = SimilarityCalculator::new();
    
    // Test known similar columns
    let desc_embedding = vec![0.8, 0.2, 0.1, ...];
    let comment_embedding = vec![0.7, 0.3, 0.1, ...];
    
    let similarity = calc.cosine_similarity(&desc_embedding, &comment_embedding);
    assert!(similarity > 0.8, "Similar columns should have high similarity");
}

#[bench]
fn bench_graph_construction_with_similarity(b: &mut Bencher) {
    let columns = generate_test_columns(1000);
    let calc = SimilarityCalculator::new();
    
    b.iter(|| {
        let mut graph = ColumnGraph::new();
        // Build graph with similarities
        build_similarity_graph(&mut graph, &columns, &calc);
    });
}
```

## Performance Characteristics
- Graph construction: O(n) for n nodes
- Edge addition: O(1)
- Similarity calculation: <0.1ms per pair
- SIMD-optimized cosine similarity available with `similarity-search` feature

## Next Steps
- Dev 2 will implement K-NN index (Task 2.3) for faster similarity search
- Dev 2 will implement PageRank (Task 2.4) for graph-based ranking
- Dev 1 can start integrating graph components into ML profiler
- Dev 3 can begin implementing the profile endpoint using these components