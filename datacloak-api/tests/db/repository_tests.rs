use datacloak_api::models::{AnalysisRun, AnalysisLog, RunStatus, ChainType};
use datacloak_api::repositories::{
    AnalysisRunRepository, AnalysisLogRepository, CheckpointRepository, ProfileCacheRepository
};
use sqlx::postgres::PgPoolOptions;
use uuid::Uuid;
use chrono::Utc;

async fn create_test_pool() -> sqlx::PgPool {
    // In real tests, this would connect to a test database
    // For now, we'll use an in-memory approach or test container
    PgPoolOptions::new()
        .max_connections(5)
        .connect(&std::env::var("TEST_DATABASE_URL").unwrap_or_else(|_| {
            "postgres://postgres:password@localhost/datacloak_test".to_string()
        }))
        .await
        .expect("Failed to create test pool")
}

#[tokio::test]
async fn test_analysis_run_stores_multiple_columns() {
    let pool = create_test_pool().await;
    let repo = AnalysisRunRepository::new(pool);
    
    // Create run with multiple columns
    let run = AnalysisRun {
        run_id: Uuid::new_v4(),
        file_id: Uuid::new_v4(),
        selected_columns: vec!["col1".to_string(), "col2".to_string()],
        chain_type: ChainType::Sentiment,
        started_at: Utc::now(),
        completed_at: None,
        status: RunStatus::Running,
        total_rows: Some(1000),
        processed_rows: Some(0),
    };
    
    repo.create(&run).await.unwrap();
    
    // Verify storage
    let retrieved = repo.get(run.run_id).await.unwrap();
    assert_eq!(retrieved.selected_columns.len(), 2);
    assert_eq!(retrieved.selected_columns[0], "col1");
}

#[tokio::test]
async fn test_analysis_logs_per_column() {
    let pool = create_test_pool().await;
    let repo = AnalysisLogRepository::new(pool.clone());
    let run_repo = AnalysisRunRepository::new(pool);
    
    // Create a run first
    let run_id = Uuid::new_v4();
    let run = AnalysisRun {
        run_id,
        file_id: Uuid::new_v4(),
        selected_columns: vec!["description".to_string(), "comments".to_string()],
        chain_type: ChainType::Sentiment,
        started_at: Utc::now(),
        completed_at: None,
        status: RunStatus::Running,
        total_rows: Some(100),
        processed_rows: Some(0),
    };
    run_repo.create(&run).await.unwrap();
    
    // Create logs for different columns
    let log1 = AnalysisLog {
        id: Uuid::new_v4(),
        run_id,
        record_id: "rec1".to_string(),
        column_name: "description".to_string(),
        result: serde_json::json!({"sentiment": "positive"}),
        latency_ms: 45,
        created_at: Utc::now(),
    };
    
    let mut log2 = log1.clone();
    log2.id = Uuid::new_v4();
    log2.column_name = "comments".to_string();
    
    repo.create(&log1).await.unwrap();
    repo.create(&log2).await.unwrap();
    
    // Query by column
    let desc_logs = repo.get_by_column(&run_id, "description").await.unwrap();
    assert_eq!(desc_logs.len(), 1);
    assert_eq!(desc_logs[0].column_name, "description");
}

#[tokio::test]
async fn test_checkpoint_persistence() {
    let pool = create_test_pool().await;
    let repo = CheckpointRepository::new(pool);
    
    // Save checkpoint
    let checkpoint = Checkpoint {
        worker_id: 1,
        run_id: Uuid::new_v4(),
        last_offset: 10000,
        last_record_id: "rec_10000".to_string(),
    };
    
    repo.save(&checkpoint).await.unwrap();
    
    // Update checkpoint
    let mut updated = checkpoint.clone();
    updated.last_offset = 20000;
    repo.save(&updated).await.unwrap();
    
    // Verify update
    let loaded = repo.load(1, &checkpoint.run_id).await.unwrap();
    assert_eq!(loaded.last_offset, 20000);
}

#[tokio::test]
async fn test_analysis_run_update_progress() {
    let pool = create_test_pool().await;
    let repo = AnalysisRunRepository::new(pool);
    
    // Create run
    let run = AnalysisRun {
        run_id: Uuid::new_v4(),
        file_id: Uuid::new_v4(),
        selected_columns: vec!["col1".to_string()],
        chain_type: ChainType::Sentiment,
        started_at: Utc::now(),
        completed_at: None,
        status: RunStatus::Running,
        total_rows: Some(1000),
        processed_rows: Some(0),
    };
    
    repo.create(&run).await.unwrap();
    
    // Update progress
    repo.update_progress(&run.run_id, 500).await.unwrap();
    
    let updated = repo.get(run.run_id).await.unwrap();
    assert_eq!(updated.processed_rows, Some(500));
    assert_eq!(updated.status, RunStatus::Running);
}

#[tokio::test]
async fn test_analysis_run_complete() {
    let pool = create_test_pool().await;
    let repo = AnalysisRunRepository::new(pool);
    
    // Create run
    let run = AnalysisRun {
        run_id: Uuid::new_v4(),
        file_id: Uuid::new_v4(),
        selected_columns: vec!["col1".to_string()],
        chain_type: ChainType::Sentiment,
        started_at: Utc::now(),
        completed_at: None,
        status: RunStatus::Running,
        total_rows: Some(1000),
        processed_rows: Some(0),
    };
    
    repo.create(&run).await.unwrap();
    
    // Complete the run
    repo.complete(&run.run_id).await.unwrap();
    
    let completed = repo.get(run.run_id).await.unwrap();
    assert_eq!(completed.status, RunStatus::Completed);
    assert!(completed.completed_at.is_some());
}

#[tokio::test]
async fn test_profile_cache_expiry() {
    let pool = create_test_pool().await;
    let repo = ProfileCacheRepository::new(pool);
    
    let file_id = Uuid::new_v4();
    let profile_result = serde_json::json!({
        "candidates": [
            {"name": "col1", "score": 0.9},
            {"name": "col2", "score": 0.8}
        ]
    });
    
    // Cache with 1 hour expiry
    repo.cache(&file_id, &profile_result, 3600).await.unwrap();
    
    // Should retrieve cached result
    let cached = repo.get(&file_id).await.unwrap();
    assert!(cached.is_some());
    assert_eq!(cached.unwrap(), profile_result);
    
    // Test cache invalidation
    repo.invalidate(&file_id).await.unwrap();
    let invalidated = repo.get(&file_id).await.unwrap();
    assert!(invalidated.is_none());
}

#[tokio::test]
async fn test_concurrent_checkpoint_updates() {
    let pool = create_test_pool().await;
    let repo = CheckpointRepository::new(pool);
    let run_id = Uuid::new_v4();
    
    // Simulate multiple workers updating checkpoints concurrently
    let mut handles = vec![];
    
    for worker_id in 0..5 {
        let repo_clone = repo.clone();
        let handle = tokio::spawn(async move {
            for offset in (0..10000).step_by(1000) {
                let checkpoint = Checkpoint {
                    worker_id,
                    run_id,
                    last_offset: offset,
                    last_record_id: format!("rec_{}", offset),
                };
                repo_clone.save(&checkpoint).await.unwrap();
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            }
        });
        handles.push(handle);
    }
    
    // Wait for all workers
    for handle in handles {
        handle.await.unwrap();
    }
    
    // Verify all checkpoints
    for worker_id in 0..5 {
        let checkpoint = repo.load(worker_id, &run_id).await.unwrap();
        assert_eq!(checkpoint.last_offset, 9000);
    }
}

#[derive(Debug, Clone)]
struct Checkpoint {
    worker_id: i32,
    run_id: Uuid,
    last_offset: i64,
    last_record_id: String,
}