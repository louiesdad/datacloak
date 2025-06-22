use super::{ColumnGraph, NodeIndex};
use std::collections::HashMap;

pub struct PageRankCalculator {
    damping_factor: f32,
    tolerance: f32,
}

impl PageRankCalculator {
    pub fn new() -> Self {
        Self {
            damping_factor: 0.85,
            tolerance: 0.0001,
        }
    }
    
    pub fn with_damping_factor(mut self, damping_factor: f32) -> Self {
        self.damping_factor = damping_factor;
        self
    }
    
    pub fn with_tolerance(mut self, tolerance: f32) -> Self {
        self.tolerance = tolerance;
        self
    }
    
    pub fn calculate(&self, graph: &ColumnGraph, max_iterations: usize) -> HashMap<NodeIndex, f32> {
        let node_count = graph.node_count();
        if node_count == 0 {
            return HashMap::new();
        }
        
        // Initialize PageRank values
        let initial_rank = 1.0 / node_count as f32;
        let mut ranks: HashMap<NodeIndex, f32> = graph.node_indices()
            .map(|idx| (idx, initial_rank))
            .collect();
        
        let mut new_ranks = ranks.clone();
        
        // Power iteration
        for _iteration in 0..max_iterations {
            let mut converged = true;
            
            for node in graph.node_indices() {
                let mut rank = (1.0 - self.damping_factor) / node_count as f32;
                
                // Sum contributions from incoming edges
                for neighbor in graph.get_neighbors(node) {
                    let neighbor_out_degree = graph.get_neighbors(neighbor).len();
                    if neighbor_out_degree > 0 {
                        rank += self.damping_factor * ranks[&neighbor] / neighbor_out_degree as f32;
                    }
                }
                
                new_ranks.insert(node, rank);
                
                // Check convergence
                if (new_ranks[&node] - ranks[&node]).abs() > self.tolerance {
                    converged = false;
                }
            }
            
            // Swap ranks
            std::mem::swap(&mut ranks, &mut new_ranks);
            
            if converged {
                break;
            }
        }
        
        // Normalize to sum to 1
        let sum: f32 = ranks.values().sum();
        if sum > 0.0 {
            for rank in ranks.values_mut() {
                *rank /= sum;
            }
        }
        
        ranks
    }
    
    pub fn calculate_weighted(&self, graph: &ColumnGraph, max_iterations: usize) -> HashMap<NodeIndex, f32> {
        let node_count = graph.node_count();
        if node_count == 0 {
            return HashMap::new();
        }
        
        // Initialize PageRank values
        let initial_rank = 1.0 / node_count as f32;
        let mut ranks: HashMap<NodeIndex, f32> = graph.node_indices()
            .map(|idx| (idx, initial_rank))
            .collect();
        
        let mut new_ranks = ranks.clone();
        
        // Precompute outgoing weights for each node
        let mut out_weights: HashMap<NodeIndex, f32> = HashMap::new();
        for node in graph.node_indices() {
            let weight_sum: f32 = graph.edge_indices()
                .filter_map(|edge| {
                    let (a, _b) = graph.edge_endpoints(edge)?;
                    if a == node {
                        graph.edge_weight(edge).copied()
                    } else {
                        None
                    }
                })
                .sum();
            out_weights.insert(node, weight_sum);
        }
        
        // Power iteration
        for _iteration in 0..max_iterations {
            let mut converged = true;
            
            for node in graph.node_indices() {
                let mut rank = (1.0 - self.damping_factor) / node_count as f32;
                
                // Sum weighted contributions from incoming edges
                for edge in graph.edge_indices() {
                    if let Some((source, target)) = graph.edge_endpoints(edge) {
                        if target == node {
                            let edge_weight = graph.edge_weight(edge).copied().unwrap_or(0.0);
                            let source_out_weight = out_weights.get(&source).copied().unwrap_or(1.0);
                            
                            if source_out_weight > 0.0 {
                                rank += self.damping_factor * ranks[&source] * edge_weight / source_out_weight;
                            }
                        }
                    }
                }
                
                new_ranks.insert(node, rank);
                
                // Check convergence
                if (new_ranks[&node] - ranks[&node]).abs() > self.tolerance {
                    converged = false;
                }
            }
            
            // Swap ranks
            std::mem::swap(&mut ranks, &mut new_ranks);
            
            if converged {
                break;
            }
        }
        
        // Normalize to sum to 1
        let sum: f32 = ranks.values().sum();
        if sum > 0.0 {
            for rank in ranks.values_mut() {
                *rank /= sum;
            }
        }
        
        ranks
    }
}

impl ColumnGraph {
    /// Calculate PageRank for all nodes in the graph
    pub fn calculate_pagerank(&self, damping_factor: f32, max_iterations: usize) -> HashMap<NodeIndex, f32> {
        PageRankCalculator::new()
            .with_damping_factor(damping_factor)
            .calculate(self, max_iterations)
    }
    
    /// Calculate weighted PageRank considering edge weights
    pub fn calculate_pagerank_weighted(&self, damping_factor: f32, max_iterations: usize) -> HashMap<NodeIndex, f32> {
        PageRankCalculator::new()
            .with_damping_factor(damping_factor)
            .calculate_weighted(self, max_iterations)
    }
    
    /// Get nodes ranked by PageRank score
    pub fn get_top_nodes_by_pagerank(&self, k: usize, damping_factor: f32, max_iterations: usize) -> Vec<(NodeIndex, f32, String)> {
        let ranks = self.calculate_pagerank(damping_factor, max_iterations);
        let mut ranked_nodes: Vec<_> = ranks.into_iter()
            .filter_map(|(idx, rank)| {
                let name = self.get_node(idx)?.name.clone();
                Some((idx, rank, name))
            })
            .collect();
        
        ranked_nodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranked_nodes.truncate(k);
        ranked_nodes
    }
}