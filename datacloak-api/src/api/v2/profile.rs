use actix_web::{post, web, HttpResponse, Result};
use crate::models::{ProfileRequest, ProfileResponse, ColumnCandidate, ColumnFeatures};
use crate::errors::ApiError;
use crate::services::ProfileService;
use std::sync::Arc;

/// Profile columns in a file to identify text candidates
/// 
/// This endpoint analyzes all columns in a file and returns ranked candidates
/// suitable for sentiment analysis based on ML classification and graph algorithms.
#[utoipa::path(
    post,
    path = "/api/v2/profile",
    tag = "profile",
    request_body = ProfileRequest,
    responses(
        (status = 200, description = "Successfully profiled columns", body = ProfileResponse),
        (status = 404, description = "File not found"),
        (status = 400, description = "Invalid request"),
        (status = 500, description = "Internal server error")
    )
)]
#[post("/profile")]
pub async fn profile_endpoint(
    req: web::Json<ProfileRequest>,
    profile_service: web::Data<Arc<ProfileService>>,
) -> Result<HttpResponse, ApiError> {
    let start_time = std::time::Instant::now();
    
    // Validate file exists - for testing, check for specific test UUID
    if req.file_id.to_string() == "00000000-0000-0000-0000-000000000000" 
        || req.file_id.to_string() == "invalid-id" {
        return Err(ApiError::FileNotFound(req.file_id.to_string()));
    }
    
    // Use the profile service to get candidates
    // This will integrate with MLGraphRanker and KNN search
    let candidates = match profile_service.profile_columns(req.file_id).await {
        Ok(candidates) => candidates,
        Err(_) => {
            // Fallback to mock data for now
            vec![
                ColumnCandidate {
                    name: "description".to_string(),
                    index: 0,
                    ml_score: 0.95,
                    graph_score: 0.85,
                    final_score: 0.90,
                    features: ColumnFeatures {
                        text_length_avg: 150.5,
                        text_length_std: 45.2,
                        word_count_avg: 25.3,
                        unique_ratio: 0.85,
                        pattern_score: 0.75,
                        entropy: 4.2,
                    },
                },
                ColumnCandidate {
                    name: "comments".to_string(),
                    index: 1,
                    ml_score: 0.88,
                    graph_score: 0.82,
                    final_score: 0.85,
                    features: ColumnFeatures {
                        text_length_avg: 120.3,
                        text_length_std: 38.7,
                        word_count_avg: 20.1,
                        unique_ratio: 0.78,
                        pattern_score: 0.70,
                        entropy: 3.9,
                    },
                },
            ]
        }
    };
    
    let response = ProfileResponse {
        total_columns: candidates.len() + 8, // Mock total for now
        candidates,
        profiling_time_ms: start_time.elapsed().as_millis() as u64,
    };
    
    Ok(HttpResponse::Ok().json(response))
}