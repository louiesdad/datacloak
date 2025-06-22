use datacloak_api::services::{WorkerCoordinator, WorkerAssignment, WorkerStatus};
use datacloak_api::models::{AnalysisRun, ChainType, RunStatus};
use uuid::Uuid;
use tokio::time::Duration;

#[tokio::test]
async fn test_worker_assignment() {
    let coordinator = WorkerCoordinator::new(4); // 4 workers
    let columns = vec!["col1".to_string(), "col2".to_string(), "col3".to_string(), "col4".to_string(), "col5".to_string()];
    
    let assignments = coordinator.assign_columns(&columns);
    
    // Each worker gets columns (round-robin distribution)
    assert_eq!(assignments.len(), 4);
    assert_eq!(assignments[0].columns, vec!["col1", "col5"]); // Round-robin
    assert_eq!(assignments[1].columns, vec!["col2"]);
    assert_eq!(assignments[2].columns, vec!["col3"]);
    assert_eq!(assignments[3].columns, vec!["col4"]);
    
    // Verify all columns are assigned
    let all_assigned: Vec<String> = assignments.iter()
        .flat_map(|a| a.columns.clone())
        .collect();
    assert_eq!(all_assigned.len(), 5);
}

#[tokio::test]
async fn test_worker_health_monitoring() {
    let coordinator = WorkerCoordinator::new(2);
    let run_id = Uuid::new_v4();
    
    // Start workers
    let worker1 = coordinator.spawn_worker(0, run_id, vec!["col1".to_string()]).await.unwrap();
    let worker2 = coordinator.spawn_worker(1, run_id, vec!["col2".to_string()]).await.unwrap();
    
    // Check health
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(coordinator.is_healthy(0).await);
    assert!(coordinator.is_healthy(1).await);
    
    // Simulate worker failure
    coordinator.mark_worker_failed(0, "Worker crashed").await;
    assert!(!coordinator.is_healthy(0).await);
    assert!(coordinator.is_healthy(1).await);
}

#[tokio::test]
async fn test_work_redistribution() {
    let coordinator = WorkerCoordinator::new(3);
    let run_id = Uuid::new_v4();
    let columns = vec!["col1".to_string(), "col2".to_string(), "col3".to_string()];
    
    // Initial assignment
    coordinator.start_processing(run_id, columns).await.unwrap();
    
    // Worker 1 fails
    coordinator.mark_worker_failed(1, "Connection lost").await;
    
    // Work should be redistributed
    let active = coordinator.get_active_assignments().await;
    assert_eq!(active.len(), 2); // 2 workers remaining
    
    // col2 should be reassigned to another worker
    let all_columns: Vec<String> = active.iter()
        .flat_map(|a| a.columns.clone())
        .collect();
    assert!(all_columns.contains(&"col2".to_string()));
}

#[tokio::test]
async fn test_worker_progress_tracking() {
    let coordinator = WorkerCoordinator::new(2);
    let run_id = Uuid::new_v4();
    
    coordinator.start_processing(run_id, vec!["col1".to_string(), "col2".to_string()]).await.unwrap();
    
    // Update progress for workers
    coordinator.update_worker_progress(0, 50).await.unwrap();
    coordinator.update_worker_progress(1, 75).await.unwrap();
    
    let progress = coordinator.get_overall_progress(run_id).await.unwrap();
    assert!(progress.overall_percentage > 0.0);
    assert_eq!(progress.worker_progress.len(), 2);
    assert_eq!(progress.worker_progress[&0], 50);
    assert_eq!(progress.worker_progress[&1], 75);
}

#[tokio::test]
async fn test_worker_recovery() {
    let coordinator = WorkerCoordinator::new(3);
    let run_id = Uuid::new_v4();
    
    coordinator.start_processing(run_id, vec!["col1".to_string(), "col2".to_string()]).await.unwrap();
    
    // Simulate worker failure and recovery
    coordinator.mark_worker_failed(1, "Network error").await;
    tokio::time::sleep(Duration::from_millis(50)).await;
    
    // Worker should automatically recover
    coordinator.attempt_worker_recovery(run_id).await.unwrap();
    
    // Check that work is redistributed
    let active = coordinator.get_active_assignments().await;
    assert!(active.len() >= 2); // Should have recovered or redistributed
}

#[tokio::test]
async fn test_concurrent_worker_operations() {
    let coordinator = WorkerCoordinator::new(5);
    let run_id = Uuid::new_v4();
    
    // Start multiple workers concurrently
    let mut handles = vec![];
    for i in 0..5 {
        let coord = coordinator.clone();
        let handle = tokio::spawn(async move {
            coord.spawn_worker(i, run_id, vec![format!("col_{}", i)]).await
        });
        handles.push(handle);
    }
    
    // Wait for all workers to start
    for handle in handles {
        assert!(handle.await.unwrap().is_ok());
    }
    
    // All workers should be healthy
    for i in 0..5 {
        assert!(coordinator.is_healthy(i).await);
    }
    
    // Update progress concurrently
    let mut progress_handles = vec![];
    for i in 0..5 {
        let coord = coordinator.clone();
        let handle = tokio::spawn(async move {
            for progress in (0..100).step_by(10) {
                coord.update_worker_progress(i, progress).await.unwrap();
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        });
        progress_handles.push(handle);
    }
    
    // Wait for all progress updates
    for handle in progress_handles {
        handle.await.unwrap();
    }
    
    let final_progress = coordinator.get_overall_progress(run_id).await.unwrap();
    assert_eq!(final_progress.worker_progress.len(), 5);
}

#[tokio::test]
async fn test_load_balancing() {
    let coordinator = WorkerCoordinator::new(3);
    
    // Test with uneven column distribution
    let many_columns: Vec<String> = (0..10).map(|i| format!("col_{}", i)).collect();
    let assignments = coordinator.assign_columns(&many_columns);
    
    // Check that work is distributed reasonably evenly
    let worker_loads: Vec<usize> = assignments.iter().map(|a| a.columns.len()).collect();
    let max_load = *worker_loads.iter().max().unwrap();
    let min_load = *worker_loads.iter().min().unwrap();
    
    // Load difference should be at most 1
    assert!(max_load - min_load <= 1);
    
    // Total columns should be preserved
    let total_assigned: usize = worker_loads.iter().sum();
    assert_eq!(total_assigned, 10);
}