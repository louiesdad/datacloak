use actix_web::{test, http::StatusCode};
use datacloak_api::api::v2::create_app;
use datacloak_api::models::ProfileResponse;
use serde_json::json;
use uuid::Uuid;

#[tokio::test]
async fn test_profile_endpoint_returns_candidates() {
    // Given a test server and file
    let app = test::init_service(create_app()).await;
    let file_id = Uuid::new_v4();
    
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
        .set_json(&json!({"file_id": "00000000-0000-0000-0000-000000000000"}))
        .to_request();
    let resp = test::call_service(&app, req).await;
    
    // Then return 404
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_analyze_multi_field_endpoint() {
    let app = test::init_service(create_app()).await;
    let file_id = Uuid::new_v4();
    
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