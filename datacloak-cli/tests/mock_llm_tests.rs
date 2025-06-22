use datacloak_cli::mock_llm::*;
use tokio;
use reqwest;
use serde_json;
use std::time::{Duration, Instant};
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::test]
async fn test_mock_llm_multi_field_support() {
    // Start mock server
    let port = 3002;
    let server_handle = tokio::spawn(async move {
        start_mock_server(port, Some("multi-field".to_string())).await.unwrap();
    });
    
    // Wait for server to start
    tokio::time::sleep(Duration::from_millis(500)).await;
    
    // Send multi-field request
    let client = reqwest::Client::new();
    let request = ChatRequest {
        model: "gpt-4".to_string(),
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: "Analyze sentiment in multiple fields".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: r#"Field: description
Text: Great product!
Field: review
Text: Could be better"#.to_string(),
            },
        ],
        temperature: None,
        max_tokens: None,
    };
    
    let resp = client.post(&format!("http://localhost:{}/v1/chat/completions", port))
        .header("authorization", "Bearer test-key")
        .json(&request)
        .send()
        .await
        .unwrap();
    
    assert_eq!(resp.status(), 200);
    let result: ChatResponse = resp.json().await.unwrap();
    assert!(result.choices[0].message.content.contains("positive"));
    
    // Clean up
    server_handle.abort();
}

#[test]
fn test_mock_response_templates() {
    let templates = ResponseTemplates::new();
    
    // Test sentiment templates
    let positive_text = "This is amazing!";
    let response = templates.sentiment_response(positive_text);
    assert!(response.contains("positive"));
    assert!(response.contains("confidence"));
    
    // Test churn prediction
    let churn_text = "I'm canceling my subscription";
    let response = templates.churn_response(churn_text);
    assert!(response.contains("high_risk"));
}

#[tokio::test]
async fn test_mock_server_rate_limiting() {
    // Start mock server with low rate limit
    let port = 3003;
    let server_handle = tokio::spawn(async move {
        let server = Arc::new(MockLlmServerBuilder::new()
            .with_rate_limit(10) // 10 req/s
            .with_scenario("generic")
            .build());
        server.start(port).await.unwrap();
    });
    
    // Wait for server to start
    tokio::time::sleep(Duration::from_millis(500)).await;
    
    let client = reqwest::Client::new();
    
    // Send 15 requests rapidly
    let mut handles = vec![];
    let success_count = Arc::new(Mutex::new(0));
    let rate_limited_count = Arc::new(Mutex::new(0));
    
    for _ in 0..15 {
        let client = client.clone();
        let success = success_count.clone();
        let rate_limited = rate_limited_count.clone();
        
        let handle = tokio::spawn(async move {
            let request = ChatRequest {
                model: "gpt-4".to_string(),
                messages: vec![ChatMessage {
                    role: "user".to_string(),
                    content: "test".to_string(),
                }],
                temperature: None,
                max_tokens: None,
            };
            
            let resp = client.post("http://localhost:3003/v1/chat/completions")
                .header("authorization", "Bearer test-key")
                .json(&request)
                .send()
                .await
                .unwrap();
            
            match resp.status().as_u16() {
                200 => *success.lock().await += 1,
                429 => *rate_limited.lock().await += 1,
                _ => panic!("Unexpected response status: {}", resp.status()),
            }
        });
        handles.push(handle);
    }
    
    // Wait for all requests to complete
    for handle in handles {
        handle.await.unwrap();
    }
    
    let success = *success_count.lock().await;
    let rate_limited = *rate_limited_count.lock().await;
    
    assert_eq!(success, 10);
    assert_eq!(rate_limited, 5);
    
    // Clean up
    server_handle.abort();
}

#[tokio::test]
async fn test_mock_server_latency_simulation() {
    // Start mock server with specific latency
    let port = 3004;
    let server_handle = tokio::spawn(async move {
        let server = Arc::new(MockLlmServerBuilder::new()
            .with_latency(100, 150) // 100-150ms
            .build());
        server.start(port).await.unwrap();
    });
    
    // Wait for server to start
    tokio::time::sleep(Duration::from_millis(500)).await;
    
    let client = reqwest::Client::new();
    let request = ChatRequest {
        model: "gpt-4".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "test".to_string(),
        }],
        temperature: None,
        max_tokens: None,
    };
    
    // Measure request time
    let start = Instant::now();
    let resp = client.post(&format!("http://localhost:{}/v1/chat/completions", port))
        .header("authorization", "Bearer test-key")
        .json(&request)
        .send()
        .await
        .unwrap();
    let elapsed = start.elapsed();
    
    assert_eq!(resp.status(), 200);
    // Should take at least 100ms
    assert!(elapsed.as_millis() >= 100);
    // Should take less than 200ms (150ms max + overhead)
    assert!(elapsed.as_millis() < 200);
    
    // Clean up
    server_handle.abort();
}

#[tokio::test]
async fn test_mock_server_error_injection() {
    // Start mock server with high error rate
    let port = 3005;
    let server_handle = tokio::spawn(async move {
        let server = Arc::new(MockLlmServerBuilder::new()
            .with_error_rate(0.5) // 50% error rate
            .build());
        server.start(port).await.unwrap();
    });
    
    // Wait for server to start
    tokio::time::sleep(Duration::from_millis(500)).await;
    
    let client = reqwest::Client::new();
    let mut error_count = 0;
    let mut success_count = 0;
    
    // Send 20 requests
    for _ in 0..20 {
        let request = ChatRequest {
            model: "gpt-4".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "test".to_string(),
            }],
            temperature: None,
            max_tokens: None,
        };
        
        let resp = client.post(&format!("http://localhost:{}/v1/chat/completions", port))
            .header("authorization", "Bearer test-key")
            .json(&request)
            .send()
            .await
            .unwrap();
        
        match resp.status().as_u16() {
            200 => success_count += 1,
            500 => error_count += 1,
            _ => panic!("Unexpected status: {}", resp.status()),
        }
    }
    
    // With 50% error rate, we should see some of each
    assert!(error_count > 5);
    assert!(success_count > 5);
    assert_eq!(error_count + success_count, 20);
    
    // Clean up
    server_handle.abort();
}

#[tokio::test]
async fn test_multi_field_sentiment_analysis() {
    let port = 3006;
    let server_handle = tokio::spawn(async move {
        start_mock_server(port, Some("sentiment-analysis".to_string())).await.unwrap();
    });
    
    // Wait for server to start
    tokio::time::sleep(Duration::from_millis(500)).await;
    
    let client = reqwest::Client::new();
    let request = ChatRequest {
        model: "gpt-4".to_string(),
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: "Analyze sentiment for each field separately".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: r#"{
                    "record_id": "123",
                    "fields": {
                        "customer_feedback": "The product is excellent!",
                        "support_ticket": "Having issues with login",
                        "review": "Would recommend to friends"
                    }
                }"#.to_string(),
            },
        ],
        temperature: None,
        max_tokens: None,
    };
    
    let resp = client.post(&format!("http://localhost:{}/v1/chat/completions", port))
        .header("authorization", "Bearer test-key")
        .json(&request)
        .send()
        .await
        .unwrap();
    
    assert_eq!(resp.status(), 200);
    let result: ChatResponse = resp.json().await.unwrap();
    let content = &result.choices[0].message.content;
    
    // Parse response as JSON
    let analysis: serde_json::Value = serde_json::from_str(content).unwrap();
    
    // Check that we have sentiment for each field
    assert!(analysis["fields"]["customer_feedback"]["sentiment"].is_string());
    assert!(analysis["fields"]["support_ticket"]["sentiment"].is_string());
    assert!(analysis["fields"]["review"]["sentiment"].is_string());
    
    // Clean up
    server_handle.abort();
}