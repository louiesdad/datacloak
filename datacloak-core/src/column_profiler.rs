use crate::ml_classifier::{Column, ColumnType};
use crate::ml_graph_integration::MLGraphIntegration;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ColumnCandidate {
    pub name: String,
    pub column_type: ColumnType,
    pub confidence: f32,
    pub final_score: f32,
}

pub struct ColumnProfiler {
    graph_integration: Arc<MLGraphIntegration>,
}

impl ColumnProfiler {
    pub fn new() -> Self {
        Self {
            graph_integration: Arc::new(MLGraphIntegration::new()),
        }
    }

    pub fn with_model(model_path: &str) -> Result<Self, String> {
        let graph_integration = MLGraphIntegration::with_model(model_path)?;
        Ok(Self {
            graph_integration: Arc::new(graph_integration),
        })
    }

    pub async fn profile_file(&self, _file_path: &str) -> Result<Vec<ColumnCandidate>, String> {
        // For now, simple CSV reading - will be integrated with DataSource later

        // Mock implementation for testing
        let columns = vec![
            Column::new(
                "description",
                vec!["Product description text", "Another long description"],
            ),
            Column::new("price", vec!["19.99", "29.99", "39.99"]),
            Column::new(
                "comments",
                vec!["Great product!", "Loved it", "Would buy again"],
            ),
        ];

        // Use graph-based ranking with hybrid approach (70% ML, 30% graph)
        let candidates = self
            .graph_integration
            .rank_columns_with_graph(&columns, 0.7);

        Ok(candidates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_column_profiler() {
        let profiler = ColumnProfiler::new();
        let candidates = profiler.profile_file("test.csv").await.unwrap();

        assert!(!candidates.is_empty());
        // Text columns should rank higher
        assert!(candidates[0].final_score > 0.5);
    }
}
