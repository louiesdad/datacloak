use petgraph::graph::UnGraph;
use petgraph::graph::NodeIndex as PetNodeIndex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use anyhow::Result;

pub type NodeIndex = PetNodeIndex<u32>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnNode {
    pub name: String,
    pub embedding: Vec<f32>,
}

impl ColumnNode {
    pub fn new(name: &str, embedding: Vec<f32>) -> Self {
        Self {
            name: name.to_string(),
            embedding,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetrics {
    pub node_count: usize,
    pub edge_count: usize,
    pub density: f64,
    pub avg_degree: f64,
}

#[derive(Clone)]
pub struct ColumnGraph {
    graph: UnGraph<ColumnNode, f32>,
    name_to_node: HashMap<String, NodeIndex>,
}

impl ColumnGraph {
    pub fn new() -> Self {
        Self {
            graph: UnGraph::new_undirected(),
            name_to_node: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, node: ColumnNode) -> NodeIndex {
        let name = node.name.clone();
        let idx = self.graph.add_node(node);
        self.name_to_node.insert(name, idx);
        idx
    }

    pub fn add_edge(&mut self, a: NodeIndex, b: NodeIndex, weight: f32) {
        self.graph.add_edge(a, b, weight);
    }

    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    pub fn get_edge_weight(&self, a: NodeIndex, b: NodeIndex) -> Option<f32> {
        self.graph.find_edge(a, b)
            .and_then(|edge| self.graph.edge_weight(edge))
            .copied()
    }

    pub fn get_neighbors(&self, node: NodeIndex) -> Vec<NodeIndex> {
        self.graph.neighbors(node).collect()
    }

    pub fn get_node_by_name(&self, name: &str) -> Option<NodeIndex> {
        self.name_to_node.get(name).copied()
    }

    pub fn remove_edges_below_threshold(&mut self, threshold: f32) {
        let edges_to_remove: Vec<_> = self.graph.edge_indices()
            .filter_map(|edge| {
                let weight = self.graph.edge_weight(edge)?;
                if *weight < threshold {
                    Some(edge)
                } else {
                    None
                }
            })
            .collect();

        for edge in edges_to_remove {
            self.graph.remove_edge(edge);
        }
    }

    pub fn node_indices(&self) -> impl Iterator<Item = NodeIndex> + '_ {
        self.graph.node_indices()
    }
    
    pub fn edge_indices(&self) -> impl Iterator<Item = petgraph::graph::EdgeIndex<u32>> + '_ {
        self.graph.edge_indices()
    }
    
    pub fn get_node(&self, idx: NodeIndex) -> Option<&ColumnNode> {
        self.graph.node_weight(idx)
    }
    
    pub fn edge_endpoints(&self, edge: petgraph::graph::EdgeIndex<u32>) -> Option<(NodeIndex, NodeIndex)> {
        self.graph.edge_endpoints(edge)
    }
    
    pub fn edge_weight(&self, edge: petgraph::graph::EdgeIndex<u32>) -> Option<&f32> {
        self.graph.edge_weight(edge)
    }
    
    pub fn get_metrics(&self) -> GraphMetrics {
        let node_count = self.node_count();
        let edge_count = self.edge_count();
        
        let max_edges = if node_count > 1 {
            (node_count * (node_count - 1)) / 2
        } else {
            0
        };
        
        let density = if max_edges > 0 {
            edge_count as f64 / max_edges as f64
        } else {
            0.0
        };

        let total_degree: usize = self.graph.node_indices()
            .map(|node| self.graph.neighbors(node).count())
            .sum();
        
        let avg_degree = if node_count > 0 {
            total_degree as f64 / node_count as f64
        } else {
            0.0
        };

        GraphMetrics {
            node_count,
            edge_count,
            density,
            avg_degree,
        }
    }

    pub fn to_json(&self) -> Result<String> {
        #[derive(Serialize)]
        struct SerializedGraph {
            nodes: Vec<(usize, ColumnNode)>,
            edges: Vec<(usize, usize, f32)>,
        }

        let nodes: Vec<_> = self.graph.node_indices()
            .map(|idx| (idx.index(), self.graph[idx].clone()))
            .collect();

        let edges: Vec<_> = self.graph.edge_indices()
            .map(|edge| {
                let (a, b) = self.graph.edge_endpoints(edge).unwrap();
                let weight = *self.graph.edge_weight(edge).unwrap();
                (a.index(), b.index(), weight)
            })
            .collect();

        let serialized = SerializedGraph { nodes, edges };
        Ok(serde_json::to_string(&serialized)?)
    }

    pub fn from_json(json: &str) -> Result<Self> {
        #[derive(Deserialize)]
        struct SerializedGraph {
            nodes: Vec<(usize, ColumnNode)>,
            edges: Vec<(usize, usize, f32)>,
        }

        let data: SerializedGraph = serde_json::from_str(json)?;
        let mut graph = Self::new();
        
        let mut index_map = HashMap::new();
        
        for (old_idx, node) in data.nodes {
            let new_idx = graph.add_node(node);
            index_map.insert(old_idx, new_idx);
        }
        
        for (a_idx, b_idx, weight) in data.edges {
            if let (Some(&a), Some(&b)) = (index_map.get(&a_idx), index_map.get(&b_idx)) {
                graph.add_edge(a, b, weight);
            }
        }
        
        Ok(graph)
    }
}

impl Default for ColumnGraph {
    fn default() -> Self {
        Self::new()
    }
}