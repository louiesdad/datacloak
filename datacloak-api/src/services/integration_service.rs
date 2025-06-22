use crate::models::{AnalyzeRequest, ProfileRequest, EstimateRequest, ChainType, RunStatus};
use crate::services::{
    ProfileService, ETAService, WorkerCoordinator, CacheLayer, LRUCache
};
use crate::repositories::{
    AnalysisRunRepository, AnalysisLogRepository, CheckpointRepository, ProfileCacheRepository
};
use crate::streaming::{StreamManager, StreamEvent};
use uuid::Uuid;
use std::sync::Arc;
use anyhow::Result;
use tokio::sync::RwLock;
use std::collections::HashMap;

pub struct IntegrationService {
    // Core services
    profile_service: Arc<ProfileService>,
    eta_service: Arc<ETAService>,
    worker_coordinator: Arc<WorkerCoordinator>,
    stream_manager: Arc<StreamManager>,
    
    // Cache layer
    cache: Arc<dyn CacheLayer>,
    
    // Repositories
    analysis_run_repo: Option<AnalysisRunRepository>,
    analysis_log_repo: Option<AnalysisLogRepository>,
    checkpoint_repo: Option<CheckpointRepository>,
    
    // Orchestrator
    orchestrator: Arc<AnalysisOrchestrator>,
}

impl IntegrationService {
    pub fn new() -> Self {
        let profile_service = Arc::new(ProfileService::new());
        let eta_estimator = crate::services::ETAEstimator::new();
        let eta_service = Arc::new(ETAService::new(eta_estimator));
        let worker_coordinator = Arc::new(WorkerCoordinator::new(4));
        let stream_manager = Arc::new(StreamManager::new());
        let cache = Arc::new(LRUCache::new(100 * 1024 * 1024)); // 100MB cache
        
        let orchestrator = Arc::new(AnalysisOrchestrator::new(
            profile_service.clone(),
            eta_service.clone(),
            worker_coordinator.clone(),
            stream_manager.clone(),
        ));
        
        Self {
            profile_service,
            eta_service,
            worker_coordinator,
            stream_manager,
            cache,
            analysis_run_repo: None,
            analysis_log_repo: None,
            checkpoint_repo: None,
            orchestrator,
        }
    }
    
    pub fn with_repositories(
        mut self,
        analysis_run_repo: AnalysisRunRepository,
        analysis_log_repo: AnalysisLogRepository,
        checkpoint_repo: CheckpointRepository,
    ) -> Self {
        self.analysis_run_repo = Some(analysis_run_repo.clone());
        self.analysis_log_repo = Some(analysis_log_repo);
        self.checkpoint_repo = Some(checkpoint_repo.clone());
        
        // Update worker coordinator with repositories
        self.worker_coordinator = Arc::new(
            WorkerCoordinator::new(4).with_repositories(checkpoint_repo, analysis_run_repo)
        );
        
        self
    }
    
    pub async fn profile_columns(&self, request: ProfileRequest) -> Result<crate::models::ProfileResponse> {
        // Check cache first
        let cache_key = format!("profile:{}", request.file_id);
        if let Some(cached) = self.cache.get(&cache_key).await {
            if let Ok(response) = serde_json::from_value(cached) {
                return Ok(response);
            }
        }
        
        // Use orchestrator for complex profiling
        let result = self.orchestrator.profile_file(request.file_id).await?;
        
        // Cache the result for 1 hour
        let _ = self.cache.put(&cache_key, serde_json::to_value(&result)?, 3600).await;
        
        Ok(result)
    }
    
    pub async fn estimate_analysis(&self, request: EstimateRequest) -> Result<crate::models::ETAResponse> {
        // Use orchestrator for comprehensive estimation
        self.orchestrator.estimate_analysis(
            request.file_id,
            &request.selected_columns,
            request.chain_type,
        ).await
    }
    
    pub async fn start_analysis(&self, request: AnalyzeRequest) -> Result<Uuid> {
        // Use orchestrator to coordinate full analysis
        let run_id = Uuid::new_v4();
        
        self.orchestrator.start_analysis(
            run_id,
            request.file_id,
            request.selected_columns,
            request.options.chain_type,
        ).await?;
        
        Ok(run_id)
    }
    
    pub async fn get_health_status(&self) -> HealthStatus {
        let cache_stats = self.cache.stats().await;
        let active_streams = self.stream_manager.get_active_count().await;
        
        HealthStatus {
            service_status: "healthy".to_string(),
            cache_hit_rate: cache_stats.hit_rate,
            active_streams,
            worker_health: self.get_worker_health().await,
            memory_usage_mb: cache_stats.memory_usage_bytes / (1024 * 1024),
        }
    }
    
    async fn get_worker_health(&self) -> HashMap<usize, bool> {
        let mut health = HashMap::new();
        for worker_id in 0..4 {
            health.insert(worker_id, self.worker_coordinator.is_healthy(worker_id).await);
        }
        health
    }
}

// Analysis orchestrator that coordinates all components
pub struct AnalysisOrchestrator {
    profile_service: Arc<ProfileService>,
    eta_service: Arc<ETAService>,
    worker_coordinator: Arc<WorkerCoordinator>,
    stream_manager: Arc<StreamManager>,
    active_analyses: Arc<RwLock<HashMap<Uuid, AnalysisState>>>,
}

#[derive(Debug, Clone)]
struct AnalysisState {
    run_id: Uuid,
    file_id: Uuid,
    selected_columns: Vec<String>,
    chain_type: ChainType,
    status: RunStatus,
    started_at: chrono::DateTime<chrono::Utc>,
    estimated_completion: Option<chrono::DateTime<chrono::Utc>>,
}

impl AnalysisOrchestrator {
    pub fn new(
        profile_service: Arc<ProfileService>,
        eta_service: Arc<ETAService>,
        worker_coordinator: Arc<WorkerCoordinator>,
        stream_manager: Arc<StreamManager>,
    ) -> Self {
        Self {
            profile_service,
            eta_service,
            worker_coordinator,
            stream_manager,
            active_analyses: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn profile_file(&self, file_id: Uuid) -> Result<crate::models::ProfileResponse> {
        // This would integrate with Dev 1's MLGraphRanker
        // For now, delegate to profile service and wrap in ProfileResponse
        let candidates = self.profile_service.profile_columns(file_id).await.map_err(|e| anyhow::anyhow!(e))?;
        
        Ok(crate::models::ProfileResponse {
            candidates,
            total_columns: candidates.len(),
        })
    }
    
    pub async fn estimate_analysis(
        &self,
        file_id: Uuid,
        columns: &[String],
        chain_type: ChainType,
    ) -> Result<crate::models::ETAResponse> {
        // Mock file metadata - in real implementation would query file repository
        let total_rows = 10_000;
        
        self.eta_service.estimate(file_id, columns, chain_type, total_rows).await
    }
    
    pub async fn start_analysis(
        &self,
        run_id: Uuid,
        file_id: Uuid,
        columns: Vec<String>,
        chain_type: ChainType,
    ) -> Result<()> {
        // Record analysis state
        {
            let mut analyses = self.active_analyses.write().await;
            analyses.insert(run_id, AnalysisState {
                run_id,
                file_id,
                selected_columns: columns.clone(),
                chain_type: chain_type.clone(),
                status: RunStatus::Running,
                started_at: chrono::Utc::now(),
                estimated_completion: None, // Would calculate from ETA
            });
        }
        
        // Start workers through coordinator
        self.worker_coordinator.start_processing(run_id, columns).await?;
        
        // Set up monitoring task
        self.monitor_analysis(run_id).await;
        
        Ok(())
    }
    
    async fn monitor_analysis(&self, run_id: Uuid) {
        let coordinator = self.worker_coordinator.clone();
        let stream_manager = self.stream_manager.clone();
        let analyses = self.active_analyses.clone();
        
        tokio::spawn(async move {
            let mut last_progress = 0.0;
            
            for _ in 0..120 { // Monitor for up to 2 minutes
                tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
                
                // Get progress from coordinator
                if let Ok(progress) = coordinator.get_overall_progress(run_id).await {
                    if progress.overall_percentage > last_progress {
                        // Send progress update via stream
                        let event = StreamEvent::Progress {
                            processed: progress.total_processed as u64,
                            total: 1000, // Mock total
                            percentage: progress.overall_percentage as f32,
                            eta_seconds: Some(((100.0 - progress.overall_percentage) * 1.2) as u64),
                        };
                        
                        let _ = stream_manager.broadcast_to_run(&run_id, event).await;
                        last_progress = progress.overall_percentage;
                    }
                    
                    // Check if complete
                    if progress.overall_percentage >= 99.0 {
                        // Mark as completed
                        {
                            let mut analyses_write = analyses.write().await;
                            if let Some(analysis) = analyses_write.get_mut(&run_id) {
                                analysis.status = RunStatus::Completed;
                            }
                        }
                        
                        // Send completion event
                        let event = StreamEvent::Complete {
                            total_processed: progress.total_processed as u64,
                            duration_ms: 120_000,
                        };
                        
                        let _ = stream_manager.broadcast_to_run(&run_id, event).await;
                        break;
                    }
                }
            }
        });
    }
    
    pub async fn get_analysis_status(&self, run_id: &Uuid) -> Option<AnalysisState> {
        let analyses = self.active_analyses.read().await;
        analyses.get(run_id).cloned()
    }
    
    pub async fn cancel_analysis(&self, run_id: &Uuid) -> Result<()> {
        // Update status
        {
            let mut analyses = self.active_analyses.write().await;
            if let Some(analysis) = analyses.get_mut(run_id) {
                analysis.status = RunStatus::Cancelled;
            }
        }
        
        // Stop workers - would need worker coordinator method for this
        // For now, just remove from active analyses
        
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct AnalysisResult {
    pub run_id: Uuid,
    pub record_id: String,
    pub column: String,
    pub result: serde_json::Value,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
pub struct HealthStatus {
    pub service_status: String,
    pub cache_hit_rate: f64,
    pub active_streams: usize,
    pub worker_health: HashMap<usize, bool>,
    pub memory_usage_mb: usize,
}

// Additional integration utilities
pub mod utils {
    use super::*;
    
    pub async fn validate_file_exists(file_id: Uuid) -> Result<bool> {
        // In real implementation, would check file repository
        // For now, just check for test UUIDs
        Ok(file_id.to_string() != "00000000-0000-0000-0000-000000000000")
    }
    
    pub async fn get_file_metadata(file_id: Uuid) -> Result<FileMetadata> {
        // Mock file metadata
        Ok(FileMetadata {
            file_id,
            row_count: 10_000,
            column_count: 15,
            file_size_bytes: 5 * 1024 * 1024, // 5MB
            created_at: chrono::Utc::now(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct FileMetadata {
    pub file_id: Uuid,
    pub row_count: usize,
    pub column_count: usize,
    pub file_size_bytes: usize,
    pub created_at: chrono::DateTime<chrono::Utc>,
}