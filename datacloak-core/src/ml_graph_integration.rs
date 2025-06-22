use crate::ml_classifier::{Column, ColumnType, MLClassifier};
use crate::column_profiler::ColumnCandidate;
use crate::feature_extractor::FeatureExtractor;
use crate::graph::{ColumnGraph, ColumnNode};
use std::sync::Arc;

// Alias for integration tests
pub type MLGraphRanker = MLGraphIntegration;

pub struct MLGraphIntegration {
    classifier: Arc<MLClassifier>,
    feature_extractor: Arc<FeatureExtractor>,
}

impl MLGraphIntegration {
    pub fn new() -> Self {
        Self {
            classifier: Arc::new(MLClassifier::new()),
            feature_extractor: Arc::new(FeatureExtractor::new()),
        }
    }
    
    pub fn with_model(model_path: &str) -> Result<Self, String> {
        let classifier = MLClassifier::with_model(model_path)?;
        Ok(Self {
            classifier: Arc::new(classifier),
            feature_extractor: Arc::new(FeatureExtractor::new()),
        })
    }
    
    pub fn build_column_graph(&self, columns: &[Column]) -> (ColumnGraph, Vec<ColumnCandidate>) {
        let mut graph = ColumnGraph::new();
        let mut candidates = Vec::new();
        let mut node_indices = Vec::new();
        
        // First pass: Add nodes and classify columns
        for column in columns {
            // Extract embeddings for graph node
            let embedding = self.feature_extractor.extract_header_embedding(&column.name);
            let node = ColumnNode::new(&column.name, embedding);
            let node_idx = graph.add_node(node);
            node_indices.push(node_idx);
            
            // Classify column
            let prediction = self.classifier.predict(column);
            
            // Calculate relevance score
            let type_weight = match &prediction.column_type {
                ColumnType::TextLong => 1.0,
                ColumnType::TextShort => 0.8,
                ColumnType::Categorical => 0.6,
                _ => 0.1,
            };
            
            let final_score = prediction.confidence * type_weight;
            
            candidates.push(ColumnCandidate {
                name: column.name.clone(),
                column_type: prediction.column_type,
                confidence: prediction.confidence,
                final_score,
            });
        }
        
        // Second pass: Add edges based on semantic similarity
        for i in 0..columns.len() {
            for j in (i + 1)..columns.len() {
                let similarity = self.calculate_column_similarity(&columns[i], &columns[j]);
                if similarity > 0.3 {  // Threshold for edge creation
                    graph.add_edge(node_indices[i], node_indices[j], similarity);
                }
            }
        }
        
        // Remove weak edges to reduce noise
        graph.remove_edges_below_threshold(0.4);
        
        (graph, candidates)
    }
    
    fn calculate_column_similarity(&self, col1: &Column, col2: &Column) -> f32 {
        // Extract features for both columns
        let features1 = self.feature_extractor.extract_all_features(col1);
        let features2 = self.feature_extractor.extract_all_features(col2);
        
        // Simple cosine similarity
        let dot_product: f32 = features1.iter()
            .zip(features2.iter())
            .map(|(a, b)| a * b)
            .sum();
            
        let norm1: f32 = features1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = features2.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm1 * norm2 > 0.0 {
            dot_product / (norm1 * norm2)
        } else {
            0.0
        }
    }
    
    pub fn rank_columns_with_graph(
        &self,
        columns: &[Column],
        alpha: f32,  // Weight for ML score
    ) -> Vec<ColumnCandidate> {
        let (graph, mut candidates) = self.build_column_graph(columns);
        let beta = 1.0 - alpha;  // Weight for graph centrality
        
        // Calculate graph centrality scores
        let centrality_scores = self.calculate_centrality_scores(&graph);
        
        // Combine ML scores with graph centrality
        for candidate in candidates.iter_mut() {
            let ml_score = candidate.final_score;
            let graph_score = centrality_scores.get(&candidate.name).unwrap_or(&0.0);
            
            // Combined score
            candidate.final_score = alpha * ml_score + beta * graph_score;
        }
        
        // Sort by final combined score
        candidates.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).unwrap());
        
        candidates
    }
    
    fn calculate_centrality_scores(&self, graph: &ColumnGraph) -> std::collections::HashMap<String, f32> {
        use std::collections::HashMap;
        
        let mut scores = HashMap::new();
        
        // Simple degree centrality for now
        // TODO: Implement PageRank when available
        for node_idx in graph.node_indices() {
            if let Some(node) = graph.get_node(node_idx) {
                let neighbors = graph.get_neighbors(node_idx);
                
                // Weighted degree: sum of edge weights
                let weighted_degree: f32 = neighbors.iter()
                    .filter_map(|&neighbor| graph.get_edge_weight(node_idx, neighbor))
                    .sum();
                
                // Normalize by max possible degree
                let max_degree = (graph.node_count() - 1) as f32;
                let normalized_score = if max_degree > 0.0 {
                    weighted_degree / max_degree
                } else {
                    0.0
                };
                
                scores.insert(node.name.clone(), normalized_score);
            }
        }
        
        scores
    }
    
    pub async fn profile_columns(&self, columns: &[Column]) -> Result<Vec<ColumnCandidate>, String> {
        // Comprehensive column profiling with graph integration
        let candidates = self.rank_columns_with_graph(columns, 0.7);
        Ok(candidates)
    }
    
    pub async fn build_similarity_graph(&self, columns: &[Column]) -> Result<ColumnGraph, String> {
        let (graph, _) = self.build_column_graph(columns);
        Ok(graph)
    }
}

// Additional candidate structure for integration tests
#[derive(Debug, Clone)]
pub struct EnhancedColumnCandidate {
    pub name: String,
    pub column_type: ColumnType,
    pub confidence: f32,
    pub final_score: f32,
    pub ml_probability: f32,
}

impl From<ColumnCandidate> for EnhancedColumnCandidate {
    fn from(candidate: ColumnCandidate) -> Self {
        Self {
            name: candidate.name,
            column_type: candidate.column_type,
            confidence: candidate.confidence,
            final_score: candidate.final_score,
            ml_probability: candidate.confidence, // Use confidence as ML probability
        }
    }
}

// Add methods to ColumnGraph for integration tests
impl ColumnGraph {
    pub fn find_node(&self, name: &str) -> Option<crate::graph::NodeIndex> {
        self.get_node_by_name(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_column_graph() {
        let integration = MLGraphIntegration::new();
        
        let columns = vec![
            Column::new("description", vec!["Product description", "Another description"]),
            Column::new("comments", vec!["Great product", "Love it"]),
            Column::new("price", vec!["19.99", "29.99"]),
        ];
        
        let (graph, candidates) = integration.build_column_graph(&columns);
        
        assert_eq!(graph.node_count(), 3);
        assert_eq!(candidates.len(), 3);
    }
    
    #[test]
    fn test_column_similarity() {
        let integration = MLGraphIntegration::new();
        
        let col1 = Column::new("description", vec!["text", "more text"]);
        let col2 = Column::new("comments", vec!["text", "similar text"]);
        let col3 = Column::new("price", vec!["123", "456"]);
        
        let sim12 = integration.calculate_column_similarity(&col1, &col2);
        let sim13 = integration.calculate_column_similarity(&col1, &col3);
        
        // Text columns should be more similar to each other
        assert!(sim12 > sim13);
    }
    
    #[test]
    fn test_rank_columns_with_graph() {
        let integration = MLGraphIntegration::new();
        
        let columns = vec![
            Column::new("description", vec!["Long product description text", "Another paragraph"]),
            Column::new("comments", vec!["User comment", "Another comment"]),
            Column::new("id", vec!["123", "456"]),
            Column::new("price", vec!["19.99", "29.99"]),
        ];
        
        // Test with different alpha values
        let ranked_ml = integration.rank_columns_with_graph(&columns, 1.0); // Pure ML
        let _ranked_hybrid = integration.rank_columns_with_graph(&columns, 0.7); // Hybrid
        let _ranked_graph = integration.rank_columns_with_graph(&columns, 0.3); // More graph
        
        // Text columns should rank higher
        assert!(ranked_ml[0].final_score > 0.5);
        assert!(ranked_ml[0].column_type == ColumnType::TextLong || 
                ranked_ml[0].column_type == ColumnType::TextShort);
    }
    
    #[test]
    fn test_graph_centrality_calculation() {
        let integration = MLGraphIntegration::new();
        
        // Create columns that should form a connected component
        let columns = vec![
            Column::new("customer_name", vec!["John Doe", "Jane Smith"]),
            Column::new("customer_email", vec!["john@example.com", "jane@example.com"]),
            Column::new("customer_address", vec!["123 Main St", "456 Oak Ave"]),
            Column::new("product_id", vec!["P001", "P002"]),
            Column::new("order_total", vec!["99.99", "149.99"]),
        ];
        
        let (graph, _) = integration.build_column_graph(&columns);
        let centrality_scores = integration.calculate_centrality_scores(&graph);
        
        // Customer-related columns should have higher centrality
        // (they're more connected to each other)
        assert!(centrality_scores.len() == 5);
        
        // Verify all columns have scores
        for col in &columns {
            assert!(centrality_scores.contains_key(&col.name));
        }
    }
}