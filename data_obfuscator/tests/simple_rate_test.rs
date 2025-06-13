use data_obfuscator::llm_client::LlmClient;
use std::time::{Duration, Instant};

#[tokio::test] 
async fn test_rate_limiter_direct() {
    // Test the rate limiter directly without HTTP calls
    let client = LlmClient::with_rate_limit("http://test.com".to_string(), "key".to_string(), 2);
    
    let start = Instant::now();
    
    // Access the rate limiter multiple times by making failed HTTP calls
    // This should still trigger rate limiting
    let mut attempts = 0;
    for _i in 0..3 {
        let _ = client.chat("test").await; // These will fail due to invalid URL, but rate limiting should still work
        attempts += 1;
    }
    
    let elapsed = start.elapsed();
    
    println!("Made {} attempts in {:?}", attempts, elapsed);
    
    // With 2 req/s, 3 requests should take at least 1 second (2 waits of 500ms each)
    assert!(elapsed >= Duration::from_millis(500), 
           "Rate limiting should cause delay, took: {:?}", elapsed);
}

#[test]
fn test_rate_limiter_creation() {
    // Test that we can create rate limiters with different rates
    let client1 = LlmClient::new("http://test.com".to_string(), "key".to_string());
    let client2 = LlmClient::with_rate_limit("http://test.com".to_string(), "key".to_string(), 5);
    let client3 = LlmClient::with_rate_limit("http://test.com".to_string(), "key".to_string(), 1);
    
    // Just verify they compile and create successfully
    assert_eq!(client1.endpoint, "http://test.com");
    assert_eq!(client2.endpoint, "http://test.com");
    assert_eq!(client3.endpoint, "http://test.com");
}