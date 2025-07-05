use super::{ColumnGraph, NodeIndex};
use anyhow::Result;
use std::fs::File;
use std::io::Write;

impl ColumnGraph {
    /// Export the graph in DOT format for visualization
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("graph ColumnSimilarity {\n");
        dot.push_str("  node [shape=box];\n");

        // Add nodes
        for node_idx in self.node_indices() {
            let node = self.get_node(node_idx).unwrap();
            dot.push_str(&format!(
                "  {} [label=\"{}\"];\n",
                node_idx.index(),
                node.name
            ));
        }

        // Add edges
        for edge in self.edge_indices() {
            if let Some((a, b)) = self.edge_endpoints(edge) {
                let weight = *self.edge_weight(edge).unwrap();
                dot.push_str(&format!(
                    "  {} -- {} [label=\"{:.2}\", weight={:.2}];\n",
                    a.index(),
                    b.index(),
                    weight,
                    weight
                ));
            }
        }

        dot.push_str("}\n");
        dot
    }

    /// Save the graph visualization to a DOT file
    pub fn save_dot(&self, filename: &str) -> Result<()> {
        let mut file = File::create(filename)?;
        file.write_all(self.to_dot().as_bytes())?;
        Ok(())
    }

    /// Get the top K most connected nodes (by degree)
    pub fn top_nodes_by_degree(&self, k: usize) -> Vec<(NodeIndex, usize, String)> {
        let mut node_degrees: Vec<_> = self
            .node_indices()
            .map(|idx| {
                let degree = self.get_neighbors(idx).len();
                let name = self.get_node(idx).unwrap().name.clone();
                (idx, degree, name)
            })
            .collect();

        node_degrees.sort_by(|a, b| b.1.cmp(&a.1));
        node_degrees.truncate(k);
        node_degrees
    }

    /// Get the strongest edges in the graph
    pub fn top_edges(&self, k: usize) -> Vec<(String, String, f32)> {
        let mut edges: Vec<_> = self
            .edge_indices()
            .filter_map(|edge| {
                let (a, b) = self.edge_endpoints(edge)?;
                let weight = *self.edge_weight(edge)?;
                let name_a = self.get_node(a)?.name.clone();
                let name_b = self.get_node(b)?.name.clone();
                Some((name_a, name_b, weight))
            })
            .collect();

        edges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        edges.truncate(k);
        edges
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ColumnNode;

    #[test]
    fn test_dot_export() {
        let mut graph = ColumnGraph::new();
        let n1 = graph.add_node(ColumnNode::new("col1", vec![1.0]));
        let n2 = graph.add_node(ColumnNode::new("col2", vec![0.8]));
        graph.add_edge(n1, n2, 0.75);

        let dot = graph.to_dot();
        assert!(dot.contains("graph ColumnSimilarity"));
        assert!(dot.contains("col1"));
        assert!(dot.contains("col2"));
        assert!(dot.contains("0.75"));
    }

    #[test]
    fn test_top_nodes_by_degree() {
        let mut graph = ColumnGraph::new();
        let nodes: Vec<_> = (0..5)
            .map(|i| graph.add_node(ColumnNode::new(&format!("col{}", i), vec![i as f32])))
            .collect();

        // Make node 0 most connected
        for i in 1..5 {
            graph.add_edge(nodes[0], nodes[i], 0.8);
        }

        // Make node 1 second most connected
        graph.add_edge(nodes[1], nodes[2], 0.7);
        graph.add_edge(nodes[1], nodes[3], 0.6);

        let top = graph.top_nodes_by_degree(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].1, 4); // node 0 has degree 4
        assert_eq!(top[0].2, "col0");
        assert_eq!(top[1].1, 3); // node 1 has degree 3
        assert_eq!(top[1].2, "col1");
    }
}
