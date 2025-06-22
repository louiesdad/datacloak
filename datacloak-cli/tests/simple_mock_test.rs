use datacloak_cli::mock_llm::{MockLlmServerBuilder, ResponseTemplates};
use std::sync::Arc;

#[test]
fn test_response_templates() {
    let templates = ResponseTemplates::new();
    
    // Test sentiment response
    let response = templates.sentiment_response("This is amazing!");
    assert!(response.contains("positive"));
    assert!(response.contains("confidence"));
    
    // Test churn response  
    let response = templates.churn_response("I want to cancel my subscription");
    assert!(response.contains("high_risk"));
}

#[test]
fn test_mock_server_builder() {
    let server = MockLlmServerBuilder::new()
        .with_scenario("sentiment-analysis")
        .with_error_rate(0.1)
        .with_latency(50, 100)
        .with_rate_limit(20)
        .build();
    
    // Just verify it builds successfully
    let _server = Arc::new(server);
}