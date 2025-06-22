use datacloak_core::graph::{ColumnGraph, ColumnNode};

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

#[test]
fn test_get_neighbors() {
    let mut graph = ColumnGraph::new();
    let node1 = graph.add_node(ColumnNode::new("col1", vec![1.0]));
    let node2 = graph.add_node(ColumnNode::new("col2", vec![0.8]));
    let node3 = graph.add_node(ColumnNode::new("col3", vec![0.6]));
    
    graph.add_edge(node1, node2, 0.8);
    graph.add_edge(node1, node3, 0.6);
    
    let neighbors = graph.get_neighbors(node1);
    assert_eq!(neighbors.len(), 2);
    assert!(neighbors.contains(&node2));
    assert!(neighbors.contains(&node3));
}

#[test]
fn test_get_node_by_name() {
    let mut graph = ColumnGraph::new();
    let node_idx = graph.add_node(ColumnNode::new("target_column", vec![1.0, 2.0]));
    
    let found_node = graph.get_node_by_name("target_column");
    assert!(found_node.is_some());
    assert_eq!(found_node.unwrap(), node_idx);
    
    let not_found = graph.get_node_by_name("missing_column");
    assert!(not_found.is_none());
}

#[test]
fn test_graph_serialization() {
    let mut graph = ColumnGraph::new();
    let node1 = graph.add_node(ColumnNode::new("col1", vec![1.0, 0.0]));
    let node2 = graph.add_node(ColumnNode::new("col2", vec![0.0, 1.0]));
    graph.add_edge(node1, node2, 0.75);
    
    // Test serialization
    let serialized = graph.to_json().unwrap();
    let deserialized = ColumnGraph::from_json(&serialized).unwrap();
    
    assert_eq!(deserialized.node_count(), 2);
    assert_eq!(deserialized.edge_count(), 1);
    assert_eq!(deserialized.get_edge_weight(node1, node2), Some(0.75));
}

#[test]
fn test_remove_low_similarity_edges() {
    let mut graph = ColumnGraph::new();
    let node1 = graph.add_node(ColumnNode::new("col1", vec![1.0]));
    let node2 = graph.add_node(ColumnNode::new("col2", vec![0.8]));
    let node3 = graph.add_node(ColumnNode::new("col3", vec![0.3]));
    
    graph.add_edge(node1, node2, 0.8);
    graph.add_edge(node1, node3, 0.2); // Low similarity
    
    graph.remove_edges_below_threshold(0.5);
    
    assert_eq!(graph.edge_count(), 1);
    assert!(graph.get_edge_weight(node1, node2).is_some());
    assert!(graph.get_edge_weight(node1, node3).is_none());
}

#[test]
fn test_graph_metrics() {
    let mut graph = ColumnGraph::new();
    let node1 = graph.add_node(ColumnNode::new("col1", vec![1.0]));
    let node2 = graph.add_node(ColumnNode::new("col2", vec![0.8]));
    let node3 = graph.add_node(ColumnNode::new("col3", vec![0.6]));
    
    graph.add_edge(node1, node2, 0.8);
    graph.add_edge(node2, node3, 0.7);
    graph.add_edge(node1, node3, 0.6);
    
    let metrics = graph.get_metrics();
    assert_eq!(metrics.node_count, 3);
    assert_eq!(metrics.edge_count, 3);
    assert_eq!(metrics.density, 1.0); // Complete graph
    assert_eq!(metrics.avg_degree, 2.0);
}