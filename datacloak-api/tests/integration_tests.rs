use datacloak_api::api::v2::create_app;
use datacloak_api::services::{IntegrationService, AnalysisOrchestrator};
use datacloak_api::models::{ChainType, ProfileResponse, AnalyzeRequest};
use actix_web::{test, http::StatusCode};
use serde_json::json;
use uuid::Uuid;

#[tokio::test]
async fn test_full_analysis_workflow() {
    let app = test::init_service(create_app()).await;
    
    // Step 1: Upload file (simulated with test UUID)
    let file_id = Uuid::new_v4();
    
    // Step 2: Profile columns
    let profile_req = test::TestRequest::post()
        .uri("/api/v2/profile")
        .set_json(&json!({"file_id": file_id}))
        .to_request();
    let profile_resp = test::call_service(&app, profile_req).await;
    
    assert_eq!(profile_resp.status(), StatusCode::OK);
    let profile_result: ProfileResponse = test::read_body_json(profile_resp).await;
    assert!(profile_result.candidates.len() >= 2);
    
    // Step 3: Select top 3 text columns
    let selected_columns: Vec<String> = profile_result.candidates
        .iter()
        .filter(|c| c.final_score > 0.7)
        .take(3)
        .map(|c| c.name.clone())
        .collect();
    
    // Step 4: Get estimate
    let estimate_req = test::TestRequest::post()
        .uri("/api/v2/estimate")
        .set_json(&json!({
            "file_id": file_id,
            "selected_columns": selected_columns,
            "chain_type": "sentiment"
        }))
        .to_request();
    let estimate_resp = test::call_service(&app, estimate_req).await;
    
    assert_eq!(estimate_resp.status(), StatusCode::OK);
    let eta = test::read_body_json::<datacloak_api::models::ETAResponse>(estimate_resp).await;
    assert!(eta.estimated_seconds > 0);
    
    // Step 5: Start analysis (streaming)
    let analyze_req = test::TestRequest::post()
        .uri("/api/v2/analyze")
        .set_json(&json!({
            "file_id": file_id,
            "selected_columns": selected_columns,
            "options": {"chain_type": "sentiment"}
        }))
        .to_request();
    let analyze_resp = test::call_service(&app, analyze_req).await;
    
    // Should return streaming response
    assert_eq!(analyze_resp.status(), StatusCode::OK);
    assert!(analyze_resp.headers().get("content-type").unwrap().to_str().unwrap().contains("event-stream"));
}

#[tokio::test]
async fn test_concurrent_analysis_runs() {
    let app = test::init_service(create_app()).await;
    
    // Start 3 concurrent analyses
    let mut handles = vec![];
    for i in 0..3 {
        let app_clone = app.clone();
        let handle = tokio::spawn(async move {
            let file_id = Uuid::new_v4();
            let columns = vec![format!("col_{}", i)];
            
            let req = test::TestRequest::post()
                .uri("/api/v2/analyze")
                .set_json(&json!({
                    "file_id": file_id,
                    "selected_columns": columns,
                    "options": {"chain_type": "sentiment"}
                }))
                .to_request();
            
            test::call_service(&app_clone, req).await
        });
        handles.push(handle);
    }
    
    // All should complete successfully
    for handle in handles {
        let resp = handle.await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }
}

#[tokio::test]
async fn test_error_handling_and_recovery() {
    let app = test::init_service(create_app()).await;
    
    // Test with invalid file ID
    let invalid_req = test::TestRequest::post()
        .uri("/api/v2/profile")
        .set_json(&json!({"file_id": "00000000-0000-0000-0000-000000000000"}))
        .to_request();
    let invalid_resp = test::call_service(&app, invalid_req).await;
    
    assert_eq!(invalid_resp.status(), StatusCode::NOT_FOUND);
    
    // Test with malformed request
    let malformed_req = test::TestRequest::post()
        .uri("/api/v2/analyze")
        .set_json(&json!({"invalid": "data"}))
        .to_request();
    let malformed_resp = test::call_service(&app, malformed_req).await;
    
    assert_eq!(malformed_resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_caching_integration() {
    let app = test::init_service(create_app()).await;
    let file_id = Uuid::new_v4();
    
    // First profile request
    let req1 = test::TestRequest::post()
        .uri("/api/v2/profile")
        .set_json(&json!({"file_id": file_id}))
        .to_request();
    let resp1 = test::call_service(&app, req1).await;
    assert_eq!(resp1.status(), StatusCode::OK);
    
    // Second profile request (should be faster due to caching)
    let req2 = test::TestRequest::post()
        .uri("/api/v2/profile")
        .set_json(&json!({"file_id": file_id}))
        .to_request();
    let resp2 = test::call_service(&app, req2).await;
    assert_eq!(resp2.status(), StatusCode::OK);
    
    // Results should be identical
    let result1: ProfileResponse = test::read_body_json(resp1).await;
    let result2: ProfileResponse = test::read_body_json(resp2).await;
    assert_eq!(result1.candidates.len(), result2.candidates.len());
}

#[tokio::test]
async fn test_performance_requirements() {
    let app = test::init_service(create_app()).await;
    let file_id = Uuid::new_v4();
    
    // Test API response time <500ms
    let start = std::time::Instant::now();
    let req = test::TestRequest::post()
        .uri("/api/v2/profile")
        .set_json(&json!({"file_id": file_id}))
        .to_request();
    let resp = test::call_service(&app, req).await;
    let elapsed = start.elapsed();
    
    assert_eq!(resp.status(), StatusCode::OK);
    assert!(elapsed.as_millis() < 500, "Response time {} ms exceeds 500ms requirement", elapsed.as_millis());
    
    // Test estimate endpoint performance
    let start = std::time::Instant::now();
    let req = test::TestRequest::post()
        .uri("/api/v2/estimate")
        .set_json(&json!({
            "file_id": file_id,
            "selected_columns": ["col1", "col2"],
            "chain_type": "sentiment"
        }))
        .to_request();
    let resp = test::call_service(&app, req).await;
    let elapsed = start.elapsed();
    
    assert_eq!(resp.status(), StatusCode::OK);
    assert!(elapsed.as_millis() < 500, "Estimate response time {} ms exceeds 500ms requirement", elapsed.as_millis());
}