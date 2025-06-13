use data_obfuscator::llm_client::{LlmClient, LlmError};
use mockito::Server;
use std::time::{Duration, Instant};
use tokio::time::timeout;

#[tokio::test]
async fn test_rate_limiting_basic() {
    let mut server = Server::new_async().await;
    
    let _m = server.mock("POST", "/")
        .match_header("authorization", "Bearer test-key")
        .match_header("content-type", "application/json")
        .with_status(200)
        .with_body(r#"{ "choices": [ { "message": { "content": "response" } } ] }"#)
        .expect(3)  // Expect exactly 3 calls
        .create_async()
        .await;

    // Create client with 3 requests per second limit
    let client = LlmClient::with_rate_limit(server.url(), "test-key".to_string(), 3);
    
    let start = Instant::now();
    
    // Make 3 rapid requests - should work but be rate limited
    let mut results = Vec::new();
    for i in 0..3 {
        let result = client.chat(&format!("test message {}", i)).await;
        results.push(result);
    }
    
    let elapsed = start.elapsed();
    
    // All requests should succeed
    for result in results {
        assert!(result.is_ok(), "Request should succeed");
        assert_eq!(result.unwrap(), "response");
    }
    
    // Should take at least 2/3 seconds due to rate limiting (3 req/s = ~333ms between requests)
    assert!(elapsed >= Duration::from_millis(600), 
           "Rate limiting should cause delay, took: {:?}", elapsed);
}

#[tokio::test]
async fn test_rate_limiting_with_retry_after_header() {
    let mut server = Server::new_async().await;
    
    let _m1 = server.mock("POST", "/")
        .match_header("authorization", "Bearer test-key")
        .with_status(429)  // Rate limited
        .with_header("retry-after", "2")  // Wait 2 seconds
        .expect(1)
        .create_async()
        .await;
    
    let _m2 = server.mock("POST", "/")
        .match_header("authorization", "Bearer test-key") 
        .with_status(200)
        .with_body(r#"{ "choices": [ { "message": { "content": "success after retry" } } ] }"#)
        .expect(1)
        .create_async()
        .await;

    let client = LlmClient::with_rate_limit(server.url(), "test-key".to_string(), 10);
    
    let start = Instant::now();
    
    // First request should get 429 with Retry-After
    let result1 = client.chat("test message").await;
    assert!(matches!(result1, Err(LlmError::RateLimitExceeded)));
    
    let retry_elapsed = start.elapsed();
    
    // Should have slept for ~2 seconds due to Retry-After header
    assert!(retry_elapsed >= Duration::from_secs(2), 
           "Should respect Retry-After header, took: {:?}", retry_elapsed);
    
    // Second request should succeed
    let result2 = client.chat("test message 2").await;
    assert!(result2.is_ok());
    assert_eq!(result2.unwrap(), "success after retry");
}

#[tokio::test]
async fn test_rate_limiting_default_constructor() {
    let mut server = Server::new_async().await;
    
    let _m = server.mock("POST", "/")
        .match_header("authorization", "Bearer test-key")
        .with_status(200)
        .with_body(r#"{ "choices": [ { "message": { "content": "response" } } ] }"#)
        .expect_at_least(1)
        .create_async()
        .await;

    // Test default constructor sets 3 req/s limit
    let client = LlmClient::new(server.url(), "test-key".to_string());
    
    let start = Instant::now();
    
    // Make 2 requests to test rate limiting
    let result1 = client.chat("test 1").await;
    let result2 = client.chat("test 2").await;
    
    let elapsed = start.elapsed();
    
    assert!(result1.is_ok());
    assert!(result2.is_ok());
    
    // Should take at least 333ms due to 3 req/s limit
    assert!(elapsed >= Duration::from_millis(300), 
           "Default rate limiting should cause delay, took: {:?}", elapsed);
}

#[tokio::test]
async fn test_concurrent_rate_limiting() {
    let mut server = Server::new_async().await;
    
    let _m = server.mock("POST", "/")
        .match_header("authorization", "Bearer test-key")
        .with_status(200)
        .with_body(r#"{ "choices": [ { "message": { "content": "concurrent response" } } ] }"#)
        .expect(5)
        .create_async()
        .await;

    let client = LlmClient::with_rate_limit(server.url(), "test-key".to_string(), 2);
    
    let start = Instant::now();
    
    // Launch 5 concurrent requests
    let mut handles = Vec::new();
    for i in 0..5 {
        let client_clone = LlmClient::with_rate_limit(server.url(), "test-key".to_string(), 2);
        let handle = tokio::spawn(async move {
            client_clone.chat(&format!("concurrent test {}", i)).await
        });
        handles.push(handle);
    }
    
    // Wait for all to complete
    let mut results = Vec::new();
    for handle in handles {
        let result = handle.await.unwrap();
        results.push(result);
    }
    
    let elapsed = start.elapsed();
    
    // All should succeed
    for result in results {
        assert!(result.is_ok(), "Concurrent request should succeed");
    }
    
    // Should take at least 2 seconds for 5 requests at 2 req/s
    assert!(elapsed >= Duration::from_secs(2), 
           "Concurrent rate limiting should cause significant delay, took: {:?}", elapsed);
}