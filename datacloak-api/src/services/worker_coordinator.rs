use crate::models::{AnalysisRun, RunStatus};
use crate::repositories::{AnalysisRunRepository, CheckpointRepository, Checkpoint};
use crate::streaming::{StreamManager, StreamEvent};
use uuid::Uuid;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use tokio::task::JoinHandle;
use tokio::time::{Duration, interval};
use anyhow::Result;

#[derive(Debug, Clone)]
pub struct WorkerAssignment {
    pub worker_id: usize,
    pub columns: Vec<String>,
    pub estimated_load: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum WorkerStatus {
    Idle,
    Running,
    Failed(String),
    Recovering,
}

#[derive(Debug, Clone)]
pub struct WorkerInfo {
    pub worker_id: usize,
    pub status: WorkerStatus,
    pub assigned_columns: Vec<String>,
    pub current_progress: usize,
    pub last_heartbeat: chrono::DateTime<chrono::Utc>,
    pub error_count: usize,
}

#[derive(Debug, Clone)]
pub struct OverallProgress {
    pub overall_percentage: f64,
    pub worker_progress: HashMap<usize, usize>,
    pub total_processed: usize,
    pub estimated_completion: Option<chrono::DateTime<chrono::Utc>>,
}

#[derive(Clone)]
pub struct WorkerCoordinator {
    worker_count: usize,
    workers: Arc<RwLock<HashMap<usize, WorkerInfo>>>,
    active_runs: Arc<RwLock<HashMap<Uuid, Vec<WorkerAssignment>>>>,
    stream_manager: Arc<StreamManager>,
    checkpoint_repo: Option<CheckpointRepository>,
    analysis_repo: Option<AnalysisRunRepository>,
}

impl WorkerCoordinator {
    pub fn new(worker_count: usize) -> Self {
        let coordinator = Self {
            worker_count,
            workers: Arc::new(RwLock::new(HashMap::new())),
            active_runs: Arc::new(RwLock::new(HashMap::new())),
            stream_manager: Arc::new(StreamManager::new()),
            checkpoint_repo: None,
            analysis_repo: None,
        };
        
        // Initialize workers
        let workers_clone = coordinator.workers.clone();
        tokio::spawn(async move {
            let mut workers = workers_clone.write().await;
            for i in 0..worker_count {
                workers.insert(i, WorkerInfo {
                    worker_id: i,
                    status: WorkerStatus::Idle,
                    assigned_columns: Vec::new(),
                    current_progress: 0,
                    last_heartbeat: chrono::Utc::now(),
                    error_count: 0,
                });
            }
        });
        
        // Start health check task
        coordinator.start_health_monitor();
        coordinator
    }
    
    pub fn with_repositories(
        mut self,
        checkpoint_repo: CheckpointRepository,
        analysis_repo: AnalysisRunRepository,
    ) -> Self {
        self.checkpoint_repo = Some(checkpoint_repo);
        self.analysis_repo = Some(analysis_repo);
        self
    }
    
    pub fn assign_columns(&self, columns: &[String]) -> Vec<WorkerAssignment> {
        let mut assignments: Vec<WorkerAssignment> = Vec::new();
        let columns_per_worker = (columns.len() + self.worker_count - 1) / self.worker_count;
        
        for (worker_id, chunk) in columns.chunks(columns_per_worker).enumerate() {
            if worker_id >= self.worker_count {
                // Distribute remaining columns to existing workers
                let remaining_workers = self.worker_count;
                let worker_index = worker_id % remaining_workers;
                
                if let Some(existing_assignment) = assignments.get_mut(worker_index) {
                    existing_assignment.columns.extend(chunk.iter().cloned());
                    existing_assignment.estimated_load = existing_assignment.columns.len();
                }
            } else {
                assignments.push(WorkerAssignment {
                    worker_id,
                    columns: chunk.to_vec(),
                    estimated_load: chunk.len(),
                });
            }
        }
        
        assignments
    }
    
    pub async fn start_processing(&self, run_id: Uuid, columns: Vec<String>) -> Result<()> {
        let assignments = self.assign_columns(&columns);
        
        // Store assignments
        {
            let mut active_runs = self.active_runs.write().await;
            active_runs.insert(run_id, assignments.clone());
        }
        
        // Spawn workers
        for assignment in assignments {
            self.spawn_worker(assignment.worker_id, run_id, assignment.columns).await?;
        }
        
        Ok(())
    }
    
    pub async fn spawn_worker(
        &self,
        worker_id: usize,
        run_id: Uuid,
        columns: Vec<String>,
    ) -> Result<JoinHandle<()>> {
        // Update worker status
        {
            let mut workers = self.workers.write().await;
            if let Some(worker) = workers.get_mut(&worker_id) {
                worker.status = WorkerStatus::Running;
                worker.assigned_columns = columns.clone();
                worker.current_progress = 0;
                worker.last_heartbeat = chrono::Utc::now();
            }
        }
        
        // Create worker task
        let workers_clone = self.workers.clone();
        let stream_manager = self.stream_manager.clone();
        let checkpoint_repo = self.checkpoint_repo.clone();
        
        let handle = tokio::spawn(async move {
            // Simulate worker processing
            let total_work = 100; // Mock work units
            
            for progress in (0..=total_work).step_by(5) {
                // Check if worker should continue
                {
                    let workers = workers_clone.read().await;
                    if let Some(worker) = workers.get(&worker_id) {
                        if matches!(worker.status, WorkerStatus::Failed(_)) {
                            break;
                        }
                    }
                }
                
                // Update progress
                {
                    let mut workers = workers_clone.write().await;
                    if let Some(worker) = workers.get_mut(&worker_id) {
                        worker.current_progress = progress;
                        worker.last_heartbeat = chrono::Utc::now();
                    }
                }
                
                // Save checkpoint
                if let Some(repo) = &checkpoint_repo {
                    for column in &columns {
                        let checkpoint = Checkpoint {
                            worker_id: worker_id as i32,
                            run_id,
                            last_offset: (progress * 100) as i64,
                            last_record_id: format!("rec_{}", progress),
                        };
                        let _ = repo.save(&checkpoint).await;
                    }
                }
                
                // Send progress update
                let event = StreamEvent::Progress {
                    processed: progress as u64,
                    total: total_work as u64,
                    percentage: (progress as f32 / total_work as f32) * 100.0,
                    eta_seconds: Some(((total_work - progress) / 5) as u64),
                };
                let _ = stream_manager.broadcast_to_run(&run_id, event).await;
                
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            
            // Mark worker as idle when done
            {
                let mut workers = workers_clone.write().await;
                if let Some(worker) = workers.get_mut(&worker_id) {
                    worker.status = WorkerStatus::Idle;
                    worker.assigned_columns.clear();
                }
            }
        });
        
        Ok(handle)
    }
    
    pub async fn is_healthy(&self, worker_id: usize) -> bool {
        let workers = self.workers.read().await;
        if let Some(worker) = workers.get(&worker_id) {
            match worker.status {
                WorkerStatus::Failed(_) => false,
                _ => {
                    let elapsed = chrono::Utc::now() - worker.last_heartbeat;
                    elapsed < chrono::Duration::seconds(30) // 30 second timeout
                }
            }
        } else {
            false
        }
    }
    
    pub async fn mark_worker_failed(&self, worker_id: usize, error: &str) {
        let mut workers = self.workers.write().await;
        if let Some(worker) = workers.get_mut(&worker_id) {
            worker.status = WorkerStatus::Failed(error.to_string());
            worker.error_count += 1;
        }
    }
    
    pub async fn get_active_assignments(&self) -> Vec<WorkerAssignment> {
        let workers = self.workers.read().await;
        workers.values()
            .filter(|w| matches!(w.status, WorkerStatus::Running))
            .map(|w| WorkerAssignment {
                worker_id: w.worker_id,
                columns: w.assigned_columns.clone(),
                estimated_load: w.assigned_columns.len(),
            })
            .collect()
    }
    
    pub async fn update_worker_progress(&self, worker_id: usize, progress: usize) -> Result<()> {
        let mut workers = self.workers.write().await;
        if let Some(worker) = workers.get_mut(&worker_id) {
            worker.current_progress = progress;
            worker.last_heartbeat = chrono::Utc::now();
        }
        Ok(())
    }
    
    pub async fn get_overall_progress(&self, run_id: Uuid) -> Result<OverallProgress> {
        let workers = self.workers.read().await;
        let active_runs = self.active_runs.read().await;
        
        if let Some(assignments) = active_runs.get(&run_id) {
            let mut worker_progress = HashMap::new();
            let mut total_progress = 0;
            let mut total_workers = 0;
            
            for assignment in assignments {
                if let Some(worker) = workers.get(&assignment.worker_id) {
                    worker_progress.insert(assignment.worker_id, worker.current_progress);
                    total_progress += worker.current_progress;
                    total_workers += 1;
                }
            }
            
            let overall_percentage = if total_workers > 0 {
                (total_progress as f64 / (total_workers * 100) as f64) * 100.0
            } else {
                0.0
            };
            
            Ok(OverallProgress {
                overall_percentage,
                worker_progress,
                total_processed: total_progress,
                estimated_completion: None, // TODO: Calculate based on current rate
            })
        } else {
            Err(anyhow::anyhow!("Run not found"))
        }
    }
    
    pub async fn attempt_worker_recovery(&self, run_id: Uuid) -> Result<()> {
        let failed_workers: Vec<usize> = {
            let workers = self.workers.read().await;
            workers.values()
                .filter(|w| matches!(w.status, WorkerStatus::Failed(_)))
                .map(|w| w.worker_id)
                .collect()
        };
        
        for worker_id in failed_workers {
            // Attempt to redistribute work
            if let Some(assignments) = self.redistribute_work(run_id, worker_id).await? {
                // Try to restart the failed worker or reassign its work
                for assignment in assignments {
                    self.spawn_worker(assignment.worker_id, run_id, assignment.columns).await?;
                }
            }
        }
        
        Ok(())
    }
    
    async fn redistribute_work(&self, run_id: Uuid, failed_worker_id: usize) -> Result<Option<Vec<WorkerAssignment>>> {
        let mut active_runs = self.active_runs.write().await;
        
        if let Some(assignments) = active_runs.get_mut(&run_id) {
            // Find failed worker's columns
            let failed_columns: Vec<String> = assignments.iter()
                .find(|a| a.worker_id == failed_worker_id)
                .map(|a| a.columns.clone())
                .unwrap_or_default();
            
            if !failed_columns.is_empty() {
                // Remove failed worker
                assignments.retain(|a| a.worker_id != failed_worker_id);
                
                // Redistribute columns to remaining workers
                let remaining_worker_count = assignments.len();
                if remaining_worker_count > 0 {
                    let columns_per_worker = (failed_columns.len() + remaining_worker_count - 1) / remaining_worker_count;
                    
                    for (idx, chunk) in failed_columns.chunks(columns_per_worker).enumerate() {
                        if let Some(assignment) = assignments.get_mut(idx % remaining_worker_count) {
                            assignment.columns.extend(chunk.iter().cloned());
                            assignment.estimated_load = assignment.columns.len();
                        }
                    }
                    
                    return Ok(Some(assignments.clone()));
                }
            }
        }
        
        Ok(None)
    }
    
    fn start_health_monitor(&self) {
        let workers = self.workers.clone();
        let coordinator = self.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(10));
            
            loop {
                interval.tick().await;
                
                let worker_ids: Vec<usize> = {
                    let workers_read = workers.read().await;
                    workers_read.keys().cloned().collect()
                };
                
                for worker_id in worker_ids {
                    if !coordinator.is_healthy(worker_id).await {
                        // Mark worker as failed if not responding
                        coordinator.mark_worker_failed(worker_id, "Health check timeout").await;
                    }
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_worker_coordinator_basic() {
        let coordinator = WorkerCoordinator::new(3);
        
        let columns = vec!["col1".to_string(), "col2".to_string(), "col3".to_string()];
        let assignments = coordinator.assign_columns(&columns);
        
        assert_eq!(assignments.len(), 3);
        assert_eq!(assignments[0].columns.len(), 1);
    }
}