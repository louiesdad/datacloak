use datacloak_api::services::monitoring::{
    ServiceMetrics, TracingService, HealthService, MonitoringService
};
use datacloak_api::api::v2::create_app;
use actix_web::{test, http::StatusCode};
use serde_json::Value;
use uuid::Uuid;
use std::collections::HashMap;

#[tokio::test]
async fn test_prometheus_metrics_endpoint() {
    let app = test::init_service(create_app()).await;
    
    // Call metrics endpoint
    let req = test::TestRequest::get()
        .uri("/monitoring/metrics")
        .to_request();
    let resp = test::call_service(&app, req).await;
    
    assert_eq!(resp.status(), StatusCode::OK);
    
    // Check content type
    let content_type = resp.headers().get("content-type").unwrap();
    assert!(content_type.to_str().unwrap().contains("text/plain"));
    
    // Check that response contains Prometheus metrics
    let body = test::read_body(resp).await;
    let metrics_text = std::str::from_utf8(&body).unwrap();
    
    // Should contain metric names
    assert!(metrics_text.contains("datacloak_api_requests_total"));
    assert!(metrics_text.contains("datacloak_cache_hit_rate"));
    assert!(metrics_text.contains("datacloak_memory_usage_bytes"));
}

#[tokio::test]
async fn test_health_check_endpoint() {
    let app = test::init_service(create_app()).await;
    
    let req = test::TestRequest::get()
        .uri("/monitoring/health")
        .to_request();
    let resp = test::call_service(&app, req).await;
    
    assert_eq!(resp.status(), StatusCode::OK);
    
    let health_response: Value = test::read_body_json(resp).await;
    
    // Check required fields
    assert!(health_response["status"].is_string());
    assert!(health_response["version"].is_string());
    assert!(health_response["uptime_seconds"].is_number());
    assert!(health_response["checks"].is_object());
    
    // Check that all components are checked
    let checks = &health_response["checks"];
    assert!(checks["database"].is_object());
    assert!(checks["cache"].is_object());
    assert!(checks["workers"].is_object());
}

#[tokio::test]
async fn test_readiness_probe() {
    let app = test::init_service(create_app()).await;
    
    let req = test::TestRequest::get()
        .uri("/monitoring/ready")
        .to_request();
    let resp = test::call_service(&app, req).await;
    
    // Should be ready in test environment
    assert_eq!(resp.status(), StatusCode::OK);
    
    let ready_response: Value = test::read_body_json(resp).await;
    assert_eq!(ready_response["status"], "ready");
}

#[tokio::test]
async fn test_liveness_probe() {
    let app = test::init_service(create_app()).await;
    
    let req = test::TestRequest::get()
        .uri("/monitoring/live")
        .to_request();
    let resp = test::call_service(&app, req).await;
    
    assert_eq!(resp.status(), StatusCode::OK);
    
    let live_response: Value = test::read_body_json(resp).await;
    assert_eq!(live_response["status"], "alive");
    assert!(live_response["timestamp"].is_string());
}

#[tokio::test]
async fn test_slo_dashboard() {
    let app = test::init_service(create_app()).await;
    
    let req = test::TestRequest::get()
        .uri("/monitoring/slo")
        .to_request();
    let resp = test::call_service(&app, req).await;
    
    assert_eq!(resp.status(), StatusCode::OK);
    
    let slo_response: Value = test::read_body_json(resp).await;
    
    // Check SLO compliance structure
    assert!(slo_response["slo_compliance"].is_object());
    assert!(slo_response["thresholds"].is_object());
    assert!(slo_response["timestamp"].is_string());
    
    let compliance = &slo_response["slo_compliance"];
    assert!(compliance["error_rate_compliant"].is_boolean());
    assert!(compliance["latency_compliant"].is_boolean());
    assert!(compliance["availability_compliant"].is_boolean());
    assert!(compliance["overall_score"].is_number());
}

#[tokio::test]
async fn test_traces_endpoint() {
    let app = test::init_service(create_app()).await;
    
    let req = test::TestRequest::get()
        .uri("/monitoring/traces")
        .to_request();
    let resp = test::call_service(&app, req).await;
    
    assert_eq!(resp.status(), StatusCode::OK);
    
    let traces_response: Value = test::read_body_json(resp).await;
    assert!(traces_response["active_traces"].is_number());
    assert!(traces_response["timestamp"].is_string());
}

#[tokio::test]
async fn test_service_metrics_recording() {
    let metrics = ServiceMetrics::new().unwrap();
    
    // Test API request recording
    metrics.record_api_request("/api/v2/profile", "POST", 0.125, 200);
    metrics.record_api_request("/api/v2/analyze", "POST", 0.050, 400);
    
    // Test cache metrics
    metrics.update_cache_metrics(0.85);
    
    // Test resource metrics
    metrics.update_resource_metrics(1024 * 1024 * 512, 5, 3);
    
    // Test analysis completion
    let file_id = Uuid::new_v4();
    let columns = vec!["col1".to_string(), "col2".to_string()];
    metrics.record_analysis_complete(file_id, &columns, 120.5, true);
    metrics.record_analysis_complete(file_id, &columns, 45.0, false);
    
    // Test worker metrics
    metrics.update_worker_metrics(75.0);
    
    // Test stream count
    metrics.update_stream_count(8);
    
    // Render metrics and verify they contain expected data
    let metrics_text = metrics.render_metrics().unwrap();
    
    // API metrics
    assert!(metrics_text.contains("datacloak_api_requests_total"));
    assert!(metrics_text.contains("datacloak_api_errors_total"));
    
    // Performance metrics
    assert!(metrics_text.contains("datacloak_cache_hit_rate"));
    assert!(metrics_text.contains("datacloak_active_streams"));
    
    // Resource metrics
    assert!(metrics_text.contains("datacloak_memory_usage_bytes"));
    
    // Business metrics
    assert!(metrics_text.contains("datacloak_files_analyzed_total"));
    assert!(metrics_text.contains("datacloak_successful_analyses_total"));
    assert!(metrics_text.contains("datacloak_failed_analyses_total"));
}

#[tokio::test]
async fn test_distributed_tracing() {
    let tracing_service = TracingService::new();
    
    // Start a trace
    let mut metadata = HashMap::new();
    metadata.insert("user_id".to_string(), "test_user".to_string());
    metadata.insert("file_id".to_string(), Uuid::new_v4().to_string());
    
    let trace_id = tracing_service.start_trace("profile_analysis", metadata).await;
    
    // Add metadata to trace
    tracing_service.add_trace_metadata(
        trace_id,
        "selected_columns".to_string(),
        "3".to_string()
    ).await;
    
    // Start a span
    let mut span_metadata = HashMap::new();
    span_metadata.insert("worker_id".to_string(), "1".to_string());
    
    let span_id = tracing_service.start_span(
        trace_id,
        "column_processing",
        span_metadata
    ).await;
    
    assert!(span_id.is_some());
    
    // Check active trace count
    let active_count = tracing_service.get_active_trace_count().await;
    assert!(active_count >= 1);
    
    // Finish the trace
    tracing_service.finish_trace(trace_id, true, None).await;
    
    // Active count should decrease
    let final_count = tracing_service.get_active_trace_count().await;
    assert!(final_count < active_count);
}

#[tokio::test]
async fn test_health_service_components() {
    let health_service = HealthService::new("0.1.0".to_string());
    
    let health_status = health_service.check_health().await;
    
    // Basic health check structure
    assert_eq!(health_status.version, "0.1.0");
    assert!(health_status.uptime_seconds >= 0);
    
    // All components should be checked
    assert!(health_status.checks.contains_key("database"));
    assert!(health_status.checks.contains_key("cache"));
    assert!(health_status.checks.contains_key("workers"));
    
    // In test environment, all should be healthy
    for (_, component_health) in &health_status.checks {
        assert_eq!(component_health.status, "healthy");
        assert!(component_health.response_time_ms.is_some());
    }
}

#[tokio::test]
async fn test_monitoring_service_integration() {
    let monitoring = MonitoringService::new("0.1.0".to_string()).unwrap();
    
    // Test metrics integration
    monitoring.metrics.record_api_request("/test", "GET", 0.1, 200);
    let metrics_output = monitoring.metrics.render_metrics().unwrap();
    assert!(metrics_output.contains("datacloak_api_requests_total"));
    
    // Test tracing integration
    let trace_id = monitoring.tracing.start_trace(
        "test_operation",
        HashMap::new()
    ).await;
    assert!(monitoring.tracing.get_active_trace_count().await > 0);
    
    // Test health integration
    let health = monitoring.health.check_health().await;
    assert_eq!(health.version, "0.1.0");
    
    // Test SLO integration
    let slo_status = monitoring.slo_dashboard.check_slo_compliance(&monitoring.metrics);
    assert!(slo_status.overall_score >= 0.0 && slo_status.overall_score <= 1.0);
    
    // Clean up
    monitoring.tracing.finish_trace(trace_id, true, None).await;
}

#[tokio::test]
async fn test_metrics_with_api_workflow() {
    let app = test::init_service(create_app()).await;
    
    // Make some API calls to generate metrics
    let file_id = Uuid::new_v4();
    
    // Profile request
    let profile_req = test::TestRequest::post()
        .uri("/api/v2/profile")
        .set_json(&serde_json::json!({"file_id": file_id}))
        .to_request();
    let _ = test::call_service(&app, profile_req).await;
    
    // Estimate request
    let estimate_req = test::TestRequest::post()
        .uri("/api/v2/estimate")
        .set_json(&serde_json::json!({
            "file_id": file_id,
            "selected_columns": ["col1"],
            "chain_type": "sentiment"
        }))
        .to_request();
    let _ = test::call_service(&app, estimate_req).await;
    
    // Check metrics endpoint
    let metrics_req = test::TestRequest::get()
        .uri("/monitoring/metrics")
        .to_request();
    let metrics_resp = test::call_service(&app, metrics_req).await;
    
    assert_eq!(metrics_resp.status(), StatusCode::OK);
    
    let metrics_body = test::read_body(metrics_resp).await;
    let metrics_text = std::str::from_utf8(&metrics_body).unwrap();
    
    // Should show some API activity
    assert!(metrics_text.contains("datacloak_api_requests_total"));
}

#[tokio::test]
async fn test_error_scenarios() {
    let app = test::init_service(create_app()).await;
    
    // Test with invalid endpoint that would generate error metrics
    let invalid_req = test::TestRequest::post()
        .uri("/api/v2/nonexistent")
        .to_request();
    let invalid_resp = test::call_service(&app, invalid_req).await;
    
    // Should return 404
    assert_eq!(invalid_resp.status(), StatusCode::NOT_FOUND);
    
    // Health should still be good despite errors
    let health_req = test::TestRequest::get()
        .uri("/monitoring/health")
        .to_request();
    let health_resp = test::call_service(&app, health_req).await;
    
    assert_eq!(health_resp.status(), StatusCode::OK);
    
    let health_status: Value = test::read_body_json(health_resp).await;
    // Service should still be healthy despite 404 errors
    assert!(health_status["status"].as_str().unwrap() == "healthy" || 
            health_status["status"].as_str().unwrap() == "degraded");
}

#[tokio::test]
async fn test_concurrent_monitoring_access() {
    let app = std::sync::Arc::new(test::init_service(create_app()).await);
    let mut handles = vec![];
    
    // Spawn multiple tasks accessing monitoring endpoints
    for _ in 0..5 {
        let app_clone = app.clone();
        let handle = tokio::spawn(async move {
            // Metrics
            let metrics_req = test::TestRequest::get()
                .uri("/monitoring/metrics")
                .to_request();
            let metrics_resp = test::call_service(&app_clone, metrics_req).await;
            assert_eq!(metrics_resp.status(), StatusCode::OK);
            
            // Health
            let health_req = test::TestRequest::get()
                .uri("/monitoring/health")
                .to_request();
            let health_resp = test::call_service(&app_clone, health_req).await;
            assert_eq!(health_resp.status(), StatusCode::OK);
        });
        handles.push(handle);
    }
    
    // Wait for all tasks
    for handle in handles {
        handle.await.unwrap();
    }
}