# Data Obfuscator

This project provides a high-performance Rust service that obfuscates sensitive text before it is sent to an LLM and then restores the original values in the response.

## Features

- **Secure Pattern Detection**: Uses validator library for emails and Luhn algorithm for credit cards to prevent ReDoS attacks
- **High-Performance Streaming**: Configurable chunk sizes (8KB to 4MB) for processing large files efficiently
- **Rate Limiting**: Built-in Governor rate limiter (default 3 req/s) with Retry-After header support
- **Production-Ready**: Comprehensive error handling, retry logic, and monitoring

## Building

```bash
cargo build --release
```

Or build the Docker image:

```bash
docker build -t data_obfuscator .
```

## Running

The binary accepts several CLI flags:

```bash
./target/release/data_obfuscator \
    --rules config/obfuscation_rules.json \
    --llm-endpoint https://api.openai.com/v1/chat/completions \
    --api-key $OPENAI_API_KEY \
    --document-path input.txt \
    --chunk-size 262144  # 256KB chunks for streaming
```

With Docker:

```bash
docker run --rm -v $(pwd)/data:/data data_obfuscator \
    --rules /app/config/obfuscation_rules.json \
    --llm-endpoint https://api.openai.com/v1/chat/completions \
    --api-key $OPENAI_API_KEY \
    --document-path /data/input.txt
```

## Performance Optimizations

### Streaming Configuration

The service supports configurable chunk sizes for optimal performance:

```bash
# Small files or low memory (8KB chunks)
--chunk-size 8192

# Default balanced performance (256KB chunks)
--chunk-size 262144

# Large files with abundant memory (1MB chunks)
--chunk-size 1048576
```

See [BENCHMARK.md](BENCHMARK.md) for detailed performance analysis.

### Rate Limiting

The LLM client includes automatic rate limiting:
- Default: 3 requests per second
- Honors Retry-After headers from API responses
- Configurable via code (see [RATE_LIMITING.md](RATE_LIMITING.md))

## Security Features

### ReDoS Protection

All regex patterns are protected against Regular Expression Denial of Service (ReDoS) attacks:

- Email validation uses the `validator` crate (RFC 5321 compliant)
- Credit card validation uses Luhn algorithm
- Other patterns use pre-compiled RegexSet for efficiency
- Input length limits prevent catastrophic backtracking

Run security benchmarks:
```bash
cargo bench regex_redos
```

### Pattern Detection

Built-in secure patterns:
- **Email**: validator::validate_email (ReDoS-safe)
- **Credit Card**: luhn::valid (algorithm-based)
- **SSN**: Length-limited regex pattern
- **Phone**: Pre-compiled regex with bounds

## Testing

### Unit Tests
```bash
cargo test
```

### Integration Tests
```bash
# Test concurrent rate limiting
cargo test concurrent_rate_limiting_test

# Test streaming with different chunk sizes
cargo test streaming_chunk_test
```

### Security Benchmarks
```bash
# ReDoS vulnerability testing
cargo bench regex_redos -- --test

# Streaming performance
cargo bench streaming_benchmark
```

### Performance Testing
```bash
# Create a 1GB test file
dd if=/dev/urandom of=test_1gb.txt bs=1M count=1024

# Test with different chunk sizes
time ./target/release/data_obfuscator --document-path test_1gb.txt --chunk-size 8192
time ./target/release/data_obfuscator --document-path test_1gb.txt --chunk-size 1048576
```

## Configuration

### Obfuscation Rules (config/obfuscation_rules.json)
```json
{
  "rules": [
    {
      "pattern": "\\b[\\w.%+-]+@[\\w.-]+\\.[A-Za-z]{2,}\\b",
      "label": "EMAIL"
    },
    {
      "pattern": "\\b\\d{3}-\\d{2}-\\d{4}\\b",
      "label": "SSN"
    }
  ]
}
```

### Environment Variables
- `OPENAI_API_KEY`: API key for LLM service
- `RUST_LOG`: Logging level (debug, info, warn, error)

## Architecture

The service uses a modular architecture:

1. **Pattern Detection**: `secure_obfuscator.rs`
   - Validator-based email detection
   - Luhn algorithm for credit cards
   - Pre-compiled regex patterns

2. **Streaming**: `obfuscator.rs`
   - Configurable chunk-based processing
   - Memory-efficient for large files
   - Line-aware chunking

3. **LLM Client**: `llm_client.rs`
   - Governor rate limiting
   - Retry-After header support
   - Automatic retry logic

4. **Metrics**: `metrics.rs`
   - Prometheus-compatible metrics
   - Request duration histograms
   - Obfuscation counters

## Monitoring

The service exposes Prometheus metrics:
- `request_duration_seconds`: Total request processing time
- `obfuscation_duration_seconds`: Time spent obfuscating
- `llm_duration_seconds`: LLM API call duration
- `request_count`: Total requests processed

## Production Deployment

### Recommended Settings
```bash
# Production configuration
./target/release/data_obfuscator \
    --rules /etc/datacloak/rules.json \
    --llm-endpoint https://api.openai.com/v1/chat/completions \
    --api-key $OPENAI_API_KEY \
    --chunk-size 262144 \
    --debug-obfuscated-path /var/log/datacloak/obfuscated.log
```

### Docker Compose
```yaml
version: '3.8'
services:
  data-obfuscator:
    image: data_obfuscator:latest
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - RUST_LOG=info
    volumes:
      - ./config:/app/config
      - ./data:/data
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '2.0'
```

## Troubleshooting

### High Memory Usage
- Reduce chunk size: `--chunk-size 8192`
- Process files in batches
- Monitor with metrics endpoint

### Rate Limiting Errors
- Check current rate: 3 req/s default
- Monitor Retry-After headers
- Implement exponential backoff

### ReDoS Vulnerabilities
- Run `cargo bench regex_redos` regularly
- Keep validator and luhn crates updated
- Monitor pattern matching performance

The service will print the de-obfuscated LLM response to stdout.
