# DataCloak Security Documentation

This document outlines the security features and protections implemented in DataCloak to ensure safe processing of sensitive data.

## Overview

DataCloak implements defense-in-depth security principles:
- **Input Validation**: ReDoS-protected pattern matching
- **Rate Limiting**: Governor-based API protection
- **Secure Dependencies**: Validator and Luhn libraries for critical validations
- **Performance Bounds**: Guaranteed processing times
- **Comprehensive Testing**: Security benchmarks and vulnerability testing

## ReDoS (Regular Expression Denial of Service) Protection

### Background
Regular expressions can be vulnerable to catastrophic backtracking when processing malicious input, leading to CPU exhaustion and denial of service attacks.

### Protection Measures

#### 1. Secure Email Validation
```rust
// Instead of vulnerable regex
❌ regex: r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b"

// Uses validator library (RFC 5321 compliant)
✅ use validator::ValidateEmail;
   ValidateEmail::validate_email(email)
```

**Benefits:**
- RFC 5321 compliant email validation
- Built-in length limits (320 characters max)
- No regex backtracking vulnerabilities
- Validates against proper email format standards

#### 2. Credit Card Validation
```rust
// Instead of complex regex patterns
❌ regex: r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b"

// Uses Luhn algorithm
✅ use luhn;
   luhn::valid(&card_number)
```

**Benefits:**
- Mathematical validation (Luhn algorithm)
- No regex processing required
- Industry-standard credit card validation
- Detects typos and invalid numbers

#### 3. Pre-compiled Regex Patterns
```rust
// Pre-compile patterns at startup
use once_cell::sync::Lazy;
static SSN_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b\d{3}-\d{2}-\d{4}\b").unwrap()
});
```

**Benefits:**
- Compilation happens once at startup
- No runtime regex compilation overhead
- Bounded patterns with fixed structure
- Input length limits prevent excessive processing

#### 4. RegexSet for Efficiency
```rust
// Use RegexSet for multiple patterns
let regex_set = RegexSet::new(&patterns)?;
let matches: Vec<usize> = regex_set.matches(text).into_iter().collect();
```

**Benefits:**
- Single pass through input text
- Efficient multiple pattern matching
- Reduced CPU usage for pattern detection
- Better cache locality

### Security Benchmarks

Run ReDoS protection tests:
```bash
# Test patterns against OWASP ReDoS corpus
cargo bench regex_redos

# Verify all patterns complete under 1ms
cargo bench regex_redos -- --test

# Test with attack strings
cargo test test_email_validation_performance_various_long_strings
```

**Sample Attack Strings Tested:**
- Long email domains: `a@${"a".repeat(50000)}`
- Complex patterns: `${"a+b".repeat(10000)}@domain.com`
- Backtracking triggers: `${"a".repeat(30000)}X@domain.com`

### Production Monitoring

Monitor regex performance in production:
```rust
use std::time::Instant;

let start = Instant::now();
let is_valid = ValidateEmail::validate_email(email);
let duration = start.elapsed();

if duration.as_millis() >= 1 {
    log::warn!("Slow email validation: {}ms for {}", duration.as_millis(), email.len());
}
```

## Rate Limiting Protection

### Governor Implementation

DataCloak uses the Governor crate for robust rate limiting:

```rust
use governor::{Quota, RateLimiter, state::{NotKeyed, InMemoryState}, clock::QuantaClock};

// Default: 3 requests per second
let quota = Quota::per_second(3);
let rate_limiter = RateLimiter::direct(quota);

// Wait for permission before making request
rate_limiter.until_ready().await;
```

### Features

1. **Token Bucket Algorithm**
   - Allows initial burst of requests
   - Smooth rate limiting over time
   - Configurable quota and burst size

2. **Retry-After Header Support**
   ```rust
   if response.status().as_u16() == 429 {
       if let Some(retry_after) = response.headers().get("retry-after") {
           let sleep_duration = parse_retry_after(retry_after);
           tokio::time::sleep(sleep_duration).await;
       }
   }
   ```

3. **Configurable Limits**
   ```rust
   // Custom rate limit
   let client = LlmClient::with_rate_limit(endpoint, api_key, 5); // 5 req/s
   
   // Default rate limit
   let client = LlmClient::new(endpoint, api_key); // 3 req/s
   ```

### Testing Rate Limits

```bash
# Test concurrent requests respect rate limits
cargo test concurrent_rate_limiting_test

# Test sequential timing
cargo test rate_limiting_sequential_timing

# Manual testing with mock server
cargo run --example rate_limiting_demo
```

## Input Validation Security

### Length Limits
```rust
// Prevent excessively long inputs
if text.len() > 100_000 {
    return result; // Skip processing for very long texts
}

// Email length validation (handled by validator)
// Credit card validation (handled by luhn)
// SSN pattern: fixed 11-character format
```

### Pattern Complexity Bounds
```rust
// Limit regex matching iterations
let matches_found: Vec<String> = regex.find_iter(&result)
    .take(1000) // Maximum 1000 matches per pattern
    .map(|mat| mat.as_str().to_string())
    .collect();
```

### Timeout Protection
```rust
// Built into validator library
// All email validations complete < 1ms
// Credit card Luhn validation is O(n) where n = card length
```

## Streaming Security

### Memory-Bounded Processing

DataCloak processes files in configurable chunks to prevent memory exhaustion:

```rust
pub struct StreamConfig {
    pub chunk_size: usize, // Default: 256KB
}

// Memory usage = chunk_size + pattern buffers
// Maximum memory: chunk_size + ~1MB for patterns
```

### Chunk Size Security Considerations

| Chunk Size | Memory Usage | Security Implications |
|------------|--------------|----------------------|
| 8KB | Low (8KB) | Suitable for memory-constrained environments |
| 256KB | Medium (256KB) | Balanced performance and memory usage |
| 1MB | High (1MB) | Best performance, requires adequate memory |
| 4MB+ | Very High | May cause memory pressure, not recommended |

## Production Security Recommendations

### 1. Pattern Validation
```bash
# Regularly test patterns for ReDoS vulnerabilities
cargo bench regex_redos -- --test

# Update dependencies for security patches
cargo audit
cargo update
```

### 2. Rate Limiting Configuration
```rust
// Production rate limiting
let client = LlmClient::with_rate_limit(
    endpoint,
    api_key,
    3, // Conservative rate for production
);
```

### 3. Input Sanitization
```bash
# Validate input files before processing
file_size=$(stat -c%s "$input_file")
if [ $file_size -gt 1073741824 ]; then  # 1GB limit
    echo "File too large for processing"
    exit 1
fi
```

### 4. Monitoring and Alerting
```rust
// Monitor processing times
let start = Instant::now();
let result = obfuscator.obfuscate_text(text);
let duration = start.elapsed();

if duration > Duration::from_millis(100) {
    log::warn!("Slow obfuscation: {}ms for {} chars", 
               duration.as_millis(), text.len());
}
```

### 5. Environment Configuration
```bash
# Secure environment variables
export OPENAI_API_KEY="sk-your-secure-key"
export RUST_LOG="warn"  # Minimize log verbosity in production

# Resource limits
ulimit -m 1048576  # 1GB memory limit
ulimit -t 60       # 60 second CPU time limit
```

## Security Testing

### Automated Security Tests

1. **ReDoS Protection Tests**
   ```bash
   cargo test test_email_validation_performance_various_long_strings
   cargo bench regex_redos
   ```

2. **Rate Limiting Tests**
   ```bash
   cargo test concurrent_rate_limiting_test
   cargo test rate_limiting_sequential_timing
   ```

3. **Input Validation Tests**
   ```bash
   cargo test test_invalid_email_not_obfuscated
   cargo test test_invalid_credit_card_not_obfuscated
   ```

### Manual Security Testing

1. **ReDoS Attack Simulation**
   ```bash
   # Test with malicious regex patterns
   echo "a@${"a".repeat(100000)}" | ./target/release/data_obfuscator
   ```

2. **Rate Limit Testing**
   ```bash
   # Concurrent request testing
   for i in {1..10}; do
     curl -X POST localhost:3001/api/chat & 
   done
   wait
   ```

3. **Memory Exhaustion Testing**
   ```bash
   # Large file processing
   dd if=/dev/urandom of=large_test.txt bs=1M count=1024
   ./target/release/data_obfuscator --document-path large_test.txt --chunk-size 8192
   ```

## Vulnerability Disclosure

If you discover a security vulnerability in DataCloak:

1. **DO NOT** open a public GitHub issue
2. Email security findings to: [security@datacloak.io]
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will respond within 48 hours and provide a timeline for fixes.

## Security Updates

- Monitor [GitHub Security Advisories](https://github.com/advisories)
- Subscribe to Rust security announcements
- Regularly update dependencies with `cargo update`
- Review `cargo audit` output for known vulnerabilities

## Compliance Considerations

DataCloak's security features support compliance with:
- **GDPR**: Data minimization, privacy by design
- **HIPAA**: PHI protection through obfuscation
- **PCI DSS**: Credit card data protection
- **SOX**: Data integrity and audit trails

## Security Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Input Data    │────▶│  Pattern Scanner │────▶│  Secure Validators│
│                 │     │  (ReDoS-Protected)│     │  (Validator/Luhn) │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                 │
                                 ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Obfuscated    │◀────│   Obfuscator     │◀────│  Rate Limiter   │
│     Output      │     │  (Streaming)     │     │  (Governor)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

This security-first architecture ensures that DataCloak can safely process sensitive data at scale while maintaining high performance and reliability.