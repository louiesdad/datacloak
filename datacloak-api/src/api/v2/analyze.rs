use actix_web::{post, web, HttpResponse, Result};
use crate::models::{AnalyzeRequest, AnalysisRun, RunStatus};
use crate::errors::ApiError;
use crate::streaming::{SseStream, StreamEvent, StreamManager};
use crate::services::WorkerCoordinator;
use uuid::Uuid;
use std::sync::Arc;
use tokio::task;

/// Analyze multiple columns for sentiment analysis
/// 
/// This endpoint starts a multi-field analysis job and returns a Server-Sent Events
/// stream for real-time results. Each result contains the analysis for a single
/// record/column combination.
#[utoipa::path(
    post,
    path = "/api/v2/analyze",
    tag = "analyze",
    request_body = AnalyzeRequest,
    responses(
        (status = 200, description = "Analysis started, streaming results", content_type = "text/event-stream"),
        (status = 404, description = "File not found"),
        (status = 400, description = "Invalid request"),
        (status = 500, description = "Internal server error")
    )
)]
#[post("/analyze")]
pub async fn analyze_endpoint(
    req: web::Json<AnalyzeRequest>,
    stream_manager: web::Data<Arc<StreamManager>>,
    worker_coordinator: web::Data<Arc<WorkerCoordinator>>,
) -> Result<HttpResponse, ApiError> {
    // Create SSE stream
    let (sse_stream, sender) = SseStream::new(1000);
    
    // Start analysis using worker coordinator
    let run_id = Uuid::new_v4();
    let columns = req.selected_columns.clone();
    let coordinator = worker_coordinator.clone();
    
    // Start workers for the analysis
    if let Err(e) = coordinator.start_processing(run_id, columns.clone()).await {
        return Err(ApiError::InternalError(format!("Failed to start workers: {}", e)));
    }
    
    task::spawn(async move {
        // Send initial progress
        let _ = sender.send(StreamEvent::Progress {
            processed: 0,
            total: 100, // Would get from file metadata
            percentage: 0.0,
            eta_seconds: Some(60),
        }).await;
        
        // Monitor worker progress and send updates
        let mut last_progress = 0.0;
        for _ in 0..30 { // Monitor for up to 30 iterations
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            
            // Get overall progress from coordinator
            if let Ok(progress) = coordinator.get_overall_progress(run_id).await {
                if progress.overall_percentage > last_progress {
                    let _ = sender.send(StreamEvent::Progress {
                        processed: progress.total_processed as u64,
                        total: (columns.len() * 100) as u64,
                        percentage: progress.overall_percentage as f32,
                        eta_seconds: Some(((100.0 - progress.overall_percentage) / 3.0) as u64),
                    }).await;
                    
                    last_progress = progress.overall_percentage;
                }
                
                // Check if complete
                if progress.overall_percentage >= 99.0 {
                    let _ = sender.send(StreamEvent::Complete {
                        total_processed: progress.total_processed as u64,
                        duration_ms: 15000,
                    }).await;
                    break;
                }
            }
            
            // Send mock results periodically
            if last_progress > 0.0 && last_progress as i32 % 20 == 0 {
                for column in &columns {
                    let event = StreamEvent::Result {
                        record_id: format!("rec_{}", (last_progress as i32) / 20),
                        column: column.clone(),
                        result: serde_json::json!({
                            "sentiment": "positive",
                            "confidence": 0.85 + (last_progress / 1000.0)
                        }),
                        sequence: (last_progress as i32 / 20) as u64,
                    };
                    
                    if sender.send(event).await.is_err() {
                        return; // Client disconnected
                    }
                }
            }
        }
    });
    
    Ok(HttpResponse::Ok()
        .content_type("text/event-stream")
        .streaming(sse_stream.into_sse()))
}