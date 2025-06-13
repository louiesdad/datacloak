# Rate Limiting Implementation

This document describes the robust rate limiting implementation added to the LLM client using the Governor crate.

## Features Implemented

### 4.1 ✅ Governor Dependency Added
```toml
governor = "0.10.0"
```

### 4.2 ✅ Governor Rate Limiter Integration
- **Default**: 3 requests per second
- **Configurable**: Custom rate limits via `LlmClient::with_rate_limit()`
- **Algorithm**: Token bucket with `NotKeyed`, `InMemoryState`, `QuantaClock`

#### Usage Examples

```rust
// Default: 3 req/s
let client = LlmClient::new(endpoint, api_key);

// Custom rate limit: 5 req/s  
let client = LlmClient::with_rate_limit(endpoint, api_key, 5);
```

### 4.3 ✅ Retry-After Header Support
- **Detection**: Automatically detects HTTP 429 (Too Many Requests) responses
- **Parsing**: Supports Retry-After header in seconds format
- **Backoff**: Uses `tokio::time::sleep()` to respect server-requested delays
- **Fallback**: Defaults to 60 seconds if header format is unrecognized

#### Implementation Details

```rust
if response.status().as_u16() == 429 {
    if let Some(retry_after) = response.headers().get("retry-after") {
        if let Ok(retry_after_str) = retry_after.to_str() {
            let sleep_duration = if let Ok(seconds) = retry_after_str.parse::<u64>() {
                Duration::from_secs(seconds)
            } else {
                Duration::from_secs(60) // Default fallback
            };
            
            tokio::time::sleep(sleep_duration).await;
            return Err(LlmError::RateLimitExceeded);
        }
    }
}
```

## Rate Limiting Flow

1. **Pre-request**: `self.rate_limiter.until_ready().await` blocks until quota available
2. **Request**: HTTP request proceeds normally
3. **429 Detection**: Check response status for rate limiting
4. **Retry-After**: Parse and respect server's requested delay
5. **Error**: Return `LlmError::RateLimitExceeded` for proper error handling

## Performance Characteristics

- **Overhead**: Minimal (~microseconds) when quota available
- **Fairness**: Token bucket ensures smooth request distribution
- **Memory**: In-memory state, suitable for single-process applications
- **Accuracy**: Quanta clock provides high-precision timing

## Error Handling

```rust
#[derive(Debug, Error)]
pub enum LlmError {
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("invalid response")]
    InvalidResponse,
    #[error("rate limit exceeded")]  // New error type
    RateLimitExceeded,
}
```

## Testing

Rate limiting functionality is verified through:
- Unit tests with mock servers
- Integration tests with real timing verification
- Example demonstrating proper spacing between requests

Run the demo:
```bash
cargo run --example rate_limiting_demo
```

## Production Considerations

- **Rate Limits**: Adjust based on API provider limits (OpenAI: 3500 RPM for GPT-4)
- **Monitoring**: Track `RateLimitExceeded` errors in metrics
- **Scaling**: Consider distributed rate limiting for multi-process deployments
- **Fallback**: Implement exponential backoff for sustained rate limiting

## Configuration Recommendations

| API Provider | Recommended Rate | Notes |
|--------------|------------------|-------|
| OpenAI GPT-4 | 3 req/s | Conservative limit under 3500 RPM |
| OpenAI GPT-3.5 | 5 req/s | Higher limits available |
| Local Models | 10+ req/s | Depends on hardware |

## Security Benefits

- **DoS Protection**: Prevents accidental API quota exhaustion
- **Cost Control**: Limits unexpected API billing spikes  
- **Stability**: Ensures predictable application performance
- **Compliance**: Respects API provider terms of service