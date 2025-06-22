# Developer 3: Backend Engineer Tasks

## Role Overview
Transform the service layer to support multi-field analysis, implement new API endpoints, update data models, and build the streaming infrastructure for real-time results.

## TDD Learning Requirements
Focus on these TDD concepts from the reference guide:
1. **Given-When-Then (BDD style)**: Express tests in business language
2. **Testing exceptions**: Test error conditions explicitly
3. **Extract Method refactoring**: Clean up complex logic
4. **One test at a time**: Resist writing multiple failing tests

## Sprint 1 Tasks (Days 1-5)

### Task 3.1: Service API Design and Implementation
**Story**: As a client, I need RESTful endpoints to profile columns and analyze multiple fields.

**TDD Tests First**:
```rust
#[tokio::test]
async fn test_profile_endpoint_returns_candidates() {
    // Given a test server and file
    let app = test::init_service(create_app()).await;
    let file_id = upload_test_file(&app, "test_data.csv").await;
    
    // When profiling columns
    let req = test::TestRequest::post()
        .uri("/api/v2/profile")
        .set_json(&json!({"file_id": file_id}))
        .to_request();
    let resp = test::call_service(&app, req).await;
    
    // Then receive ranked candidates
    assert_eq!(resp.status(), StatusCode::OK);
    let body: ProfileResponse = test::read_body_json(resp).await;
    assert!(body.candidates.len() > 0);
    assert!(body.candidates[0].final_score >= body.candidates[1].final_score);
}

#[tokio::test]
async fn test_profile_endpoint_validates_file() {
    let app = test::init_service(create_app()).await;
    
    // When profiling non-existent file
    let req = test::TestRequest::post()
        .uri("/api/v2/profile")
        .set_json(&json!({"file_id": "invalid-id"}))
        .to_request();
    let resp = test::call_service(&app, req).await;
    
    // Then return 404
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_analyze_multi_field_endpoint() {
    let app = test::init_service(create_app()).await;
    let file_id = upload_test_file(&app, "test_data.csv").await;
    
    // When analyzing multiple columns
    let req = test::TestRequest::post()
        .uri("/api/v2/analyze")
        .set_json(&json!({
            "file_id": file_id,
            "selected_columns": ["description", "comments"],
            "options": {"chain_type": "sentiment"}
        }))
        .to_request();
    let resp = test::call_service(&app, req).await;
    
    // Then receive streaming results
    assert_eq!(resp.status(), StatusCode::OK);
    assert!(resp.headers().get("content-type").unwrap().to_str().unwrap().contains("stream"));
}
```

**Implementation Steps**:
1. Create endpoint skeletons (return fake data)
2. Add request validation
3. Integrate with profiler service
4. Add streaming response support
5. Refactor for clean architecture

**Deliverables**:
- `api/v2/profile.rs` endpoint
- `api/v2/analyze.rs` endpoint
- OpenAPI specification
- Response time: <500ms for profile

### Task 3.2: Data Model Evolution
**Story**: As a system, I need updated data models to support multi-column analysis.

**TDD Database Tests**:
```rust
#[tokio::test]
async fn test_analysis_run_stores_multiple_columns() {
    let pool = create_test_db().await;
    let repo = AnalysisRunRepository::new(pool);
    
    // Create run with multiple columns
    let run = AnalysisRun {
        run_id: Uuid::new_v4(),
        file_id: Uuid::new_v4(),
        selected_columns: vec!["col1".to_string(), "col2".to_string()],
        chain_type: ChainType::Sentiment,
        started_at: Utc::now(),
        status: RunStatus::Running,
    };
    
    repo.create(&run).await.unwrap();
    
    // Verify storage
    let retrieved = repo.get(run.run_id).await.unwrap();
    assert_eq!(retrieved.selected_columns.len(), 2);
    assert_eq!(retrieved.selected_columns[0], "col1");
}

#[tokio::test]
async fn test_analysis_logs_per_column() {
    let pool = create_test_db().await;
    let repo = AnalysisLogRepository::new(pool);
    
    // Create logs for different columns
    let log1 = AnalysisLog {
        run_id: Uuid::new_v4(),
        record_id: "rec1".to_string(),
        column_name: "description".to_string(),
        result: json!({"sentiment": "positive"}),
        latency_ms: 45,
    };
    
    let log2 = log1.clone();
    log2.column_name = "comments".to_string();
    
    repo.create(&log1).await.unwrap();
    repo.create(&log2).await.unwrap();
    
    // Query by column
    let desc_logs = repo.get_by_column(&log1.run_id, "description").await.unwrap();
    assert_eq!(desc_logs.len(), 1);
    assert_eq!(desc_logs[0].column_name, "description");
}

#[tokio::test]
async fn test_checkpoint_persistence() {
    let pool = create_test_db().await;
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
    checkpoint.last_offset = 20000;
    repo.save(&checkpoint).await.unwrap();
    
    // Verify update
    let loaded = repo.load(1, &checkpoint.run_id).await.unwrap();
    assert_eq!(loaded.last_offset, 20000);
}
```

**Schema Migration**:
```sql
-- V2__multi_column_support.sql
ALTER TABLE analysis_runs 
    DROP COLUMN selected_column,
    ADD COLUMN selected_columns JSON NOT NULL DEFAULT '[]';

ALTER TABLE analysis_logs
    ADD COLUMN column_name VARCHAR(100) NOT NULL DEFAULT 'default',
    ADD INDEX idx_column_name (run_id, column_name);

CREATE TABLE analysis_checkpoints (
    worker_id INTEGER NOT NULL,
    run_id UUID NOT NULL,
    last_offset BIGINT NOT NULL,
    last_record_id VARCHAR(255),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (worker_id, run_id)
);
```

**Deliverables**:
- Database migrations
- Updated repository layer
- Query optimization
- Migration rollback plan

### Task 3.3: Streaming Response Infrastructure
**Story**: As a client, I need to receive analysis results as a stream for real-time updates.

**TDD Streaming Tests**:
```rust
#[tokio::test]
async fn test_streaming_response() {
    let (tx, rx) = mpsc::channel(10);
    
    // Simulate streaming results
    tokio::spawn(async move {
        for i in 0..5 {
            tx.send(AnalysisResult {
                record_id: format!("rec_{}", i),
                column: "description".to_string(),
                sentiment: "positive".to_string(),
            }).await.unwrap();
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    });
    
    // Convert to HTTP stream
    let stream = ReceiverStream::new(rx);
    let body = Body::wrap_stream(stream.map(|r| {
        Ok::<_, Error>(Bytes::from(serde_json::to_vec(&r).unwrap()))
    }));
    
    // Verify streaming
    let mut reader = body.into_data_stream();
    let mut count = 0;
    while let Some(chunk) = reader.next().await {
        assert!(chunk.is_ok());
        count += 1;
    }
    assert_eq!(count, 5);
}

#[tokio::test]
async fn test_stream_error_handling() {
    let (tx, rx) = mpsc::channel(10);
    
    // Simulate error in stream
    tokio::spawn(async move {
        tx.send(Ok(AnalysisResult::new())).await.unwrap();
        tx.send(Err(anyhow!("Processing error"))).await.unwrap();
    });
    
    let stream = ReceiverStream::new(rx);
    let results: Vec<_> = stream.collect().await;
    
    assert_eq!(results.len(), 2);
    assert!(results[0].is_ok());
    assert!(results[1].is_err());
}
```

**Implementation**:
1. Server-Sent Events (SSE) support
2. WebSocket alternative
3. Backpressure handling
4. Connection recovery
5. Progress updates

**Deliverables**:
- Streaming middleware
- Client SDK updates
- Error recovery logic
- Performance: <50ms latency

## Sprint 2 Tasks (Days 6-10)

### Task 3.4: ETA Estimation Service
**Story**: As a user, I need accurate time estimates before starting analysis.

**TDD First**:
```rust
#[tokio::test]
async fn test_eta_estimation_endpoint() {
    let app = test::init_service(create_app()).await;
    
    // Given a file and columns
    let req = test::TestRequest::post()
        .uri("/api/v2/estimate")
        .set_json(&json!({
            "file_id": "test-file",
            "selected_columns": ["col1", "col2"],
            "chain_type": "sentiment"
        }))
        .to_request();
    
    // When requesting estimate
    let resp = test::call_service(&app, req).await;
    
    // Then receive time estimate
    assert_eq!(resp.status(), StatusCode::OK);
    let eta: ETAResponse = test::read_body_json(resp).await;
    assert!(eta.estimated_seconds > 0);
    assert!(eta.confidence_upper > eta.estimated_seconds);
}

#[tokio::test]
async fn test_eta_calculation_accuracy() {
    let estimator = ETAEstimator::new();
    
    // Run sample analysis
    let sample_result = SampleMetric {
        rows: 1000,
        columns: 2,
        elapsed_ms: 5000,
        tokens_used: 2000,
    };
    
    // Estimate for full file
    let estimate = estimator.calculate_eta(
        100_000, // total rows
        2,       // columns
        &sample_result
    );
    
    // Linear scaling with some variance
    assert!(estimate.estimated_seconds >= 450); // At least 450s
    assert!(estimate.estimated_seconds <= 550); // At most 550s
}

#[tokio::test]
async fn test_eta_cost_calculation() {
    let estimator = ETAEstimator::new();
    
    let cost = estimator.calculate_cost(
        100_000,           // rows
        3,                 // columns
        ChainType::Sentiment,
    );
    
    // $0.10 per 1000 tokens, ~10 tokens per cell
    let expected = (100_000 * 3 * 10) / 1000 * 0.10;
    assert!((cost.estimated_cost - expected).abs() < 0.01);
}
```

**Implementation**:
1. Sample-based estimation
2. Historical data regression
3. Cost calculation
4. Confidence intervals
5. Progress extrapolation

**Deliverables**:
- ETA service component
- Historical metrics storage
- Cost calculator
- Accuracy: ±15% of actual

### Task 3.5: Worker Coordination System
**Story**: As a system, I need to coordinate multiple workers processing different columns.

**TDD Tests**:
```rust
#[tokio::test]
async fn test_worker_assignment() {
    let coordinator = WorkerCoordinator::new(4); // 4 workers
    let columns = vec!["col1", "col2", "col3", "col4", "col5"];
    
    let assignments = coordinator.assign_columns(columns);
    
    // Each worker gets columns
    assert_eq!(assignments.len(), 4);
    assert_eq!(assignments[0].columns, vec!["col1", "col5"]); // Round-robin
    assert_eq!(assignments[1].columns, vec!["col2"]);
}

#[tokio::test]
async fn test_worker_health_monitoring() {
    let coordinator = WorkerCoordinator::new(2);
    
    // Start workers
    let worker1 = coordinator.spawn_worker(0, vec!["col1"]);
    let worker2 = coordinator.spawn_worker(1, vec!["col2"]);
    
    // Check health
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert!(coordinator.is_healthy(0).await);
    assert!(coordinator.is_healthy(1).await);
    
    // Simulate worker failure
    worker1.abort();
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(!coordinator.is_healthy(0).await);
}

#[tokio::test]
async fn test_work_redistribution() {
    let coordinator = WorkerCoordinator::new(3);
    let columns = vec!["col1", "col2", "col3"];
    
    // Initial assignment
    coordinator.start_processing(columns).await;
    
    // Worker 1 fails
    coordinator.mark_failed(1).await;
    
    // Work redistributed
    let active = coordinator.get_active_assignments().await;
    assert_eq!(active.len(), 2); // 2 workers
    assert!(active.iter().any(|a| a.columns.contains(&"col2"))); // col2 reassigned
}
```

**Implementation**:
1. Worker lifecycle management
2. Health checks
3. Work redistribution
4. Progress aggregation
5. Failure recovery

**Deliverables**:
- Worker coordinator service
- Health monitoring
- Redistribution logic
- Recovery time: <30s

### Task 3.6: Cache Layer Implementation
**Story**: As a system, I need caching to improve performance for repeated analyses.

**TDD Cache Tests**:
```rust
#[tokio::test]
async fn test_profile_result_caching() {
    let cache = ProfileCache::new(100 * MB);
    let file_id = "test-file";
    
    // First call - miss
    let result = cache.get_or_compute(file_id, async {
        expensive_profile_operation().await
    }).await;
    assert_eq!(cache.stats().misses, 1);
    
    // Second call - hit
    let cached = cache.get_or_compute(file_id, async {
        panic!("Should not be called");
    }).await;
    assert_eq!(cache.stats().hits, 1);
    assert_eq!(result, cached);
}

#[tokio::test]
async fn test_cache_invalidation() {
    let cache = ProfileCache::new(100 * MB);
    
    // Cache result
    cache.put("file1", ProfileResult::new()).await;
    assert!(cache.get("file1").await.is_some());
    
    // File modified
    cache.invalidate_if_modified("file1", Utc::now()).await;
    assert!(cache.get("file1").await.is_none());
}

#[tokio::test]
async fn test_cache_memory_limits() {
    let cache = ProfileCache::new(1 * MB); // Small cache
    
    // Fill cache
    for i in 0..100 {
        let key = format!("file_{}", i);
        let large_result = ProfileResult::with_size(100 * KB);
        cache.put(&key, large_result).await;
    }
    
    // Verify memory limit respected
    assert!(cache.memory_usage() <= 1 * MB);
    assert!(cache.len() < 100); // Some evicted
}
```

**Implementation**:
1. LRU eviction policy
2. TTL support
3. Compression for large values
4. Distributed cache option
5. Cache warming

**Deliverables**:
- Cache abstraction layer
- Redis integration option
- Memory management
- Hit rate: >80% warm cache

## Sprint 3 Tasks (Days 11-15)

### Task 3.7: Integration Layer
**Story**: As a developer, I need clean integration between all service components.

**Integration Tests**:
```rust
#[tokio::test]
async fn test_full_analysis_workflow() {
    let app = create_test_app().await;
    
    // Upload file
    let file_id = upload_csv(&app, "test_data.csv").await;
    
    // Profile columns
    let candidates = profile_columns(&app, &file_id).await;
    assert!(candidates.len() >= 5);
    
    // Select top 3 text columns
    let selected = candidates.iter()
        .filter(|c| c.final_score > 0.7)
        .take(3)
        .map(|c| c.name.clone())
        .collect();
    
    // Get estimate
    let eta = estimate_runtime(&app, &file_id, &selected).await;
    assert!(eta.estimated_seconds > 0);
    
    // Start analysis
    let run_id = start_analysis(&app, &file_id, &selected).await;
    
    // Stream results
    let mut results = Vec::new();
    let mut stream = stream_results(&app, &run_id).await;
    
    while let Some(result) = stream.next().await {
        results.push(result?);
        
        // Check progress
        let progress = get_progress(&app, &run_id).await;
        assert!(progress.percentage <= 100);
    }
    
    assert!(!results.is_empty());
}

#[tokio::test]
async fn test_concurrent_analysis_runs() {
    let app = create_test_app().await;
    let file_id = upload_large_file(&app).await;
    
    // Start 5 concurrent analyses
    let mut handles = vec![];
    for i in 0..5 {
        let app = app.clone();
        let file = file_id.clone();
        
        let handle = tokio::spawn(async move {
            let columns = vec![format!("col_{}", i)];
            start_and_wait_analysis(&app, &file, &columns).await
        });
        handles.push(handle);
    }
    
    // All should complete
    for handle in handles {
        assert!(handle.await.unwrap().is_ok());
    }
}
```

### Task 3.8: Monitoring and Observability
**Story**: As an operator, I need comprehensive monitoring of the multi-field system.

**Monitoring Tests**:
```rust
#[tokio::test]
async fn test_metrics_collection() {
    let metrics = ServiceMetrics::new();
    let app = create_app_with_metrics(metrics.clone()).await;
    
    // Perform operations
    profile_columns(&app, "file1").await;
    start_analysis(&app, "file1", vec!["col1"]).await;
    
    // Verify metrics
    assert!(metrics.profile_requests.get() >= 1);
    assert!(metrics.analysis_runs.get() >= 1);
    assert!(metrics.response_time_ms.mean() > 0.0);
}

#[tokio::test]
async fn test_distributed_tracing() {
    let tracer = init_tracer("test-service");
    let app = create_app_with_tracing(tracer).await;
    
    // Make request with trace
    let resp = client.post("/api/v2/analyze")
        .header("x-trace-id", "test-trace-123")
        .send()
        .await;
    
    // Verify trace propagation
    let spans = tracer.get_spans("test-trace-123");
    assert!(spans.iter().any(|s| s.name == "profile_columns"));
    assert!(spans.iter().any(|s| s.name == "analyze_multi_field"));
}
```

**Metrics Implementation**:
```rust
pub struct ServiceMetrics {
    // API metrics
    pub profile_requests: Counter,
    pub analysis_runs: Counter,
    pub streaming_connections: Gauge,
    
    // Performance metrics
    pub response_time_ms: Histogram,
    pub stream_latency_ms: Histogram,
    pub columns_per_second: Gauge,
    
    // Resource metrics
    pub db_connections: Gauge,
    pub cache_size_mb: Gauge,
    pub worker_utilization: Gauge,
    
    // Business metrics
    pub columns_analyzed: Counter,
    pub tokens_consumed: Counter,
    pub analysis_cost_usd: Counter,
}
```

**Deliverables**:
- Prometheus metrics
- Distributed tracing
- Health check endpoints
- SLO dashboards

## Performance Requirements
- API response time: <500ms p99
- Streaming latency: <50ms per result
- Database queries: <100ms
- Cache hit rate: >80%
- Concurrent runs: 100+

## Testing Requirements
- Unit test coverage: 90%+
- Integration test coverage: 85%+
- Contract tests for API
- Load tests with 100 concurrent users
- Chaos engineering tests

## Dependencies
- actix-web 4.0 for HTTP
- tokio 1.35 for async
- sqlx for database
- redis for caching
- prometheus for metrics

## API Documentation
All endpoints must have:
- OpenAPI 3.0 specification
- Request/response examples
- Error code documentation
- Rate limit information
- Authentication details

## TDD Best Practices for Services
1. **Test the contract** - Focus on API behavior
2. **Mock external services** - Unit tests shouldn't need DB
3. **Test error paths** - Every endpoint should handle errors
4. **Test concurrency** - Services must handle parallel requests
5. **Test streaming** - Verify backpressure and disconnects

## Security Considerations
1. Input validation on all endpoints
2. Rate limiting per client
3. Authentication/authorization
4. SQL injection prevention
5. DoS protection

## Definition of Done
- [ ] All endpoints implemented with tests
- [ ] Database migrations tested
- [ ] API documentation complete
- [ ] Integration tests passing
- [ ] Load tests meeting SLAs
- [ ] Monitoring dashboards created
- [ ] Security review completed
- [ ] Performance benchmarks documented