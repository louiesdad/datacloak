use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use utoipa::ToSchema;

// Profile endpoint models
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ProfileRequest {
    pub file_id: Uuid,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ProfileResponse {
    pub candidates: Vec<ColumnCandidate>,
    pub total_columns: usize,
    pub profiling_time_ms: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
pub struct ColumnCandidate {
    pub name: String,
    pub index: usize,
    pub ml_score: f64,
    pub graph_score: f64,
    pub final_score: f64,
    pub features: ColumnFeatures,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
pub struct ColumnFeatures {
    pub text_length_avg: f64,
    pub text_length_std: f64,
    pub word_count_avg: f64,
    pub unique_ratio: f64,
    pub pattern_score: f64,
    pub entropy: f64,
}

// Analyze endpoint models
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct AnalyzeRequest {
    pub file_id: Uuid,
    pub selected_columns: Vec<String>,
    pub options: AnalysisOptions,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct AnalysisOptions {
    pub chain_type: ChainType,
    pub batch_size: Option<usize>,
    pub max_concurrent_requests: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ChainType {
    Sentiment,
    Entity,
    Classification,
    Custom(String),
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct AnalyzeResponse {
    pub run_id: Uuid,
    pub status: RunStatus,
    pub stream_url: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Running,
    Completed,
    Failed,
    Cancelled,
}

// Streaming result models
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct AnalysisResult {
    pub record_id: String,
    pub column: String,
    pub result: serde_json::Value,
    pub timestamp: DateTime<Utc>,
}

// ETA endpoint models
#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct EstimateRequest {
    pub file_id: Uuid,
    pub selected_columns: Vec<String>,
    pub chain_type: ChainType,
}

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct ETAResponse {
    pub estimated_seconds: u64,
    pub confidence_lower: u64,
    pub confidence_upper: u64,
    pub estimated_cost: f64,
    pub total_rows: usize,
    pub total_tokens_estimate: usize,
}

// Database models
#[derive(Debug, Serialize, Deserialize)]
pub struct AnalysisRun {
    pub run_id: Uuid,
    pub file_id: Uuid,
    pub selected_columns: Vec<String>,
    pub chain_type: ChainType,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub status: RunStatus,
    pub total_rows: Option<usize>,
    pub processed_rows: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AnalysisLog {
    pub id: Uuid,
    pub run_id: Uuid,
    pub record_id: String,
    pub column_name: String,
    pub result: serde_json::Value,
    pub latency_ms: i32,
    pub created_at: DateTime<Utc>,
}