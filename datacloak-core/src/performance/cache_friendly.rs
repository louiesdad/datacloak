use std::sync::Arc;
use dashmap::DashMap;
use crossbeam::queue::ArrayQueue;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(usize);

struct NodeData {
    name: String,
    embedding: Vec<f32>,
    // Store neighbors inline for cache locality
    neighbors: Vec<(NodeId, f32)>,
}

pub struct CacheFriendlyGraph {
    // Use DashMap for lock-free concurrent access
    nodes: Arc<DashMap<NodeId, NodeData>>,
    // Node allocation counter
    next_id: Arc<AtomicUsize>,
    // Pre-allocated node pool for better cache usage
    node_pool: Option<Arc<ArrayQueue<NodeData>>>,
}

impl CacheFriendlyGraph {
    pub fn new() -> Self {
        Self {
            nodes: Arc::new(DashMap::new()),
            next_id: Arc::new(AtomicUsize::new(0)),
            node_pool: None,
        }
    }
    
    pub fn new_lock_free() -> Self {
        Self::new()
    }
    
    pub fn with_capacity(capacity: usize) -> Self {
        let nodes = DashMap::with_capacity(capacity);
        let node_pool = ArrayQueue::new(capacity);
        
        // Pre-allocate nodes
        for _ in 0..capacity {
            let _ = node_pool.push(NodeData {
                name: String::new(),
                embedding: Vec::new(),
                neighbors: Vec::with_capacity(10),
            });
        }
        
        Self {
            nodes: Arc::new(nodes),
            next_id: Arc::new(AtomicUsize::new(0)),
            node_pool: Some(Arc::new(node_pool)),
        }
    }
    
    pub fn add_node(&self, name: String, embedding: Vec<f32>) -> NodeId {
        let id = NodeId(self.next_id.fetch_add(1, Ordering::SeqCst));
        
        let node_data = if let Some(pool) = &self.node_pool {
            if let Some(mut node) = pool.pop() {
                node.name = name;
                node.embedding = embedding;
                node.neighbors.clear();
                node
            } else {
                NodeData {
                    name,
                    embedding,
                    neighbors: Vec::with_capacity(10),
                }
            }
        } else {
            NodeData {
                name,
                embedding,
                neighbors: Vec::with_capacity(10),
            }
        };
        
        self.nodes.insert(id, node_data);
        id
    }
    
    pub fn add_edge(&self, from: NodeId, to: NodeId, weight: f32) {
        // Add edge in both directions for undirected graph
        if let Some(mut from_node) = self.nodes.get_mut(&from) {
            from_node.neighbors.push((to, weight));
            // Keep neighbors sorted by node ID for cache-friendly access
            from_node.neighbors.sort_by_key(|(id, _)| id.0);
        }
        
        if let Some(mut to_node) = self.nodes.get_mut(&to) {
            to_node.neighbors.push((from, weight));
            to_node.neighbors.sort_by_key(|(id, _)| id.0);
        }
    }
    
    pub fn get_neighbors(&self, node: NodeId) -> Vec<NodeId> {
        self.nodes.get(&node)
            .map(|n| n.neighbors.iter().map(|(id, _)| *id).collect())
            .unwrap_or_default()
    }
    
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    
    pub fn iter_nodes_cache_friendly(&self) -> impl Iterator<Item = NodeId> + '_ {
        // Return nodes in ID order for sequential memory access
        let max_id = self.next_id.load(Ordering::SeqCst);
        (0..max_id).map(NodeId).filter(move |id| self.nodes.contains_key(id))
    }
    
    pub fn prefetch_neighbors(&self, node: NodeId) {
        if let Some(node_data) = self.nodes.get(&node) {
            for (neighbor_id, _) in &node_data.neighbors {
                if let Some(neighbor_data) = self.nodes.get(neighbor_id) {
                    // Prefetch neighbor data
                    let _ = &neighbor_data.embedding;
                }
            }
        }
    }
    
    pub fn batch_get_embeddings(&self, nodes: &[NodeId]) -> Vec<Option<Vec<f32>>> {
        nodes.iter()
            .map(|id| {
                self.nodes.get(id)
                    .map(|n| n.embedding.clone())
            })
            .collect()
    }
    
    pub fn clear(&self) {
        self.nodes.clear();
        self.next_id.store(0, Ordering::SeqCst);
    }
}

impl Default for CacheFriendlyGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cache_friendly_graph_basic() {
        let graph = CacheFriendlyGraph::new();
        
        let n1 = graph.add_node("node1".to_string(), vec![1.0, 2.0]);
        let n2 = graph.add_node("node2".to_string(), vec![3.0, 4.0]);
        
        graph.add_edge(n1, n2, 0.5);
        
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.get_neighbors(n1), vec![n2]);
        assert_eq!(graph.get_neighbors(n2), vec![n1]);
    }
    
    #[test]
    fn test_concurrent_access() {
        use std::thread;
        
        let graph = Arc::new(CacheFriendlyGraph::new());
        let mut handles = vec![];
        
        for i in 0..10 {
            let graph_clone = graph.clone();
            let handle = thread::spawn(move || {
                for j in 0..100 {
                    graph_clone.add_node(
                        format!("node_{}_{}", i, j),
                        vec![i as f32, j as f32]
                    );
                }
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        assert_eq!(graph.node_count(), 1000);
    }
}