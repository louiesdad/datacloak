use datacloak_core::graph::{ColumnGraph, ColumnNode};

#[test]
fn test_pagerank_simple_graph() {
    let mut graph = ColumnGraph::new();
    let a = graph.add_node(ColumnNode::new("A", vec![1.0]));
    let b = graph.add_node(ColumnNode::new("B", vec![1.0]));
    let c = graph.add_node(ColumnNode::new("C", vec![1.0]));
    
    graph.add_edge(a, b, 1.0);
    graph.add_edge(b, c, 1.0);
    graph.add_edge(c, a, 1.0);
    
    let ranks = graph.calculate_pagerank(0.85, 100);
    
    // All nodes should have equal rank in a cycle
    assert!((ranks[&a] - 0.333).abs() < 0.01);
    assert!((ranks[&b] - 0.333).abs() < 0.01);
    assert!((ranks[&c] - 0.333).abs() < 0.01);
}

#[test]
fn test_pagerank_hub_node() {
    let mut graph = ColumnGraph::new();
    let hub = graph.add_node(ColumnNode::new("hub", vec![1.0]));
    let nodes: Vec<_> = (0..5).map(|i| {
        graph.add_node(ColumnNode::new(&format!("node{}", i), vec![1.0]))
    }).collect();
    
    // All nodes point to hub
    for &node in &nodes {
        graph.add_edge(node, hub, 1.0);
    }
    
    let ranks = graph.calculate_pagerank(0.85, 100);
    
    // Hub should have highest rank
    assert!(ranks[&hub] > ranks[&nodes[0]] * 2.0);
}

#[test]
fn test_pagerank_disconnected_nodes() {
    let mut graph = ColumnGraph::new();
    let a = graph.add_node(ColumnNode::new("A", vec![1.0]));
    let b = graph.add_node(ColumnNode::new("B", vec![1.0]));
    let c = graph.add_node(ColumnNode::new("C", vec![1.0]));
    let d = graph.add_node(ColumnNode::new("D", vec![1.0]));
    
    // Create two disconnected components
    graph.add_edge(a, b, 1.0);
    graph.add_edge(c, d, 1.0);
    
    let ranks = graph.calculate_pagerank(0.85, 100);
    
    // All nodes should have non-zero rank
    assert!(ranks[&a] > 0.0);
    assert!(ranks[&b] > 0.0);
    assert!(ranks[&c] > 0.0);
    assert!(ranks[&d] > 0.0);
    
    // Sum of all ranks should be approximately 1.0
    let sum: f32 = ranks.values().sum();
    assert!((sum - 1.0).abs() < 0.01);
}

#[test]
fn test_pagerank_weighted_edges() {
    let mut graph = ColumnGraph::new();
    let a = graph.add_node(ColumnNode::new("A", vec![1.0]));
    let b = graph.add_node(ColumnNode::new("B", vec![1.0]));
    let c = graph.add_node(ColumnNode::new("C", vec![1.0]));
    
    // A strongly connected to B, weakly to C
    graph.add_edge(a, b, 0.9);
    graph.add_edge(a, c, 0.1);
    graph.add_edge(b, a, 0.5);
    graph.add_edge(c, a, 0.5);
    
    let ranks = graph.calculate_pagerank_weighted(0.85, 100);
    
    // B should have higher rank than C due to stronger connection from A
    assert!(ranks[&b] > ranks[&c]);
}

#[test]
fn test_pagerank_convergence() {
    let mut graph = ColumnGraph::new();
    let nodes: Vec<_> = (0..10).map(|i| {
        graph.add_node(ColumnNode::new(&format!("node{}", i), vec![1.0]))
    }).collect();
    
    // Create a more complex graph
    for i in 0..10 {
        for j in 0..10 {
            if i != j && (i + j) % 3 == 0 {
                graph.add_edge(nodes[i], nodes[j], 0.5);
            }
        }
    }
    
    // Test different iteration counts
    let _ranks_10 = graph.calculate_pagerank(0.85, 10);
    let ranks_50 = graph.calculate_pagerank(0.85, 50);
    let ranks_100 = graph.calculate_pagerank(0.85, 100);
    
    // Later iterations should converge
    let diff: f32 = nodes.iter()
        .map(|&node| (ranks_50[&node] - ranks_100[&node]).abs())
        .sum();
    
    assert!(diff < 0.001, "PageRank should converge");
}

#[test]
fn test_pagerank_single_node() {
    let mut graph = ColumnGraph::new();
    let a = graph.add_node(ColumnNode::new("A", vec![1.0]));
    
    let ranks = graph.calculate_pagerank(0.85, 10);
    
    assert_eq!(ranks.len(), 1);
    assert!((ranks[&a] - 1.0).abs() < 0.001);
}

#[test]
fn test_pagerank_self_loop() {
    let mut graph = ColumnGraph::new();
    let a = graph.add_node(ColumnNode::new("A", vec![1.0]));
    let b = graph.add_node(ColumnNode::new("B", vec![1.0]));
    
    graph.add_edge(a, a, 0.5); // Self loop
    graph.add_edge(a, b, 0.5);
    graph.add_edge(b, a, 1.0);
    
    let ranks = graph.calculate_pagerank(0.85, 100);
    
    // Should still converge with self loops
    assert!(ranks[&a] > 0.0);
    assert!(ranks[&b] > 0.0);
    let sum: f32 = ranks.values().sum();
    assert!((sum - 1.0).abs() < 0.01);
}

#[test]
fn test_pagerank_performance() {
    use std::time::Instant;
    
    let mut graph = ColumnGraph::new();
    let nodes: Vec<_> = (0..1000).map(|i| {
        graph.add_node(ColumnNode::new(&format!("col{}", i), vec![1.0]))
    }).collect();
    
    // Create edges
    for i in 0..1000 {
        for j in 0..10 {
            let target = (i + j + 1) % 1000;
            graph.add_edge(nodes[i], nodes[target], 0.5);
        }
    }
    
    let start = Instant::now();
    let ranks = graph.calculate_pagerank(0.85, 100);
    let elapsed = start.elapsed();
    
    println!("PageRank for 1000 nodes: {:?}", elapsed);
    assert_eq!(ranks.len(), 1000);
    assert!(elapsed.as_millis() < 100); // Should complete within 100ms
}