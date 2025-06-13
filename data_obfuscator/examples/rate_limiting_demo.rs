use data_obfuscator::llm_client::LlmClient;
use std::time::{Duration, Instant};

#[tokio::main]
async fn main() {
    println!("Rate Limiting Demo");
    println!("=================");
    
    // Create client with 2 requests per second limit
    let client = LlmClient::with_rate_limit(
        "https://httpbin.org/delay/0.1".to_string(),  // Fast endpoint for testing
        "dummy-key".to_string(), 
        2  // 2 requests per second
    );
    
    println!("Created client with 2 req/s rate limit");
    
    let start = Instant::now();
    
    // Make 4 requests - should be spread out over time due to rate limiting
    for i in 0..4 {
        let request_start = Instant::now();
        
        // This will fail (httpbin doesn't return ChatGPT format), but rate limiting will still work
        let result = client.chat(&format!("Request {}", i + 1)).await;
        
        let request_elapsed = request_start.elapsed();
        let total_elapsed = start.elapsed();
        
        println!(
            "Request {}: {:?} after {:?} total (took {:?})", 
            i + 1, 
            result.is_ok(),
            total_elapsed,
            request_elapsed
        );
    }
    
    let total_elapsed = start.elapsed();
    println!("\nTotal time for 4 requests: {:?}", total_elapsed);
    println!("Expected minimum time with 2 req/s: ~1.5 seconds");
    
    if total_elapsed >= Duration::from_millis(1500) {
        println!("✅ Rate limiting is working correctly!");
    } else {
        println!("❌ Rate limiting may not be working as expected");
    }
}