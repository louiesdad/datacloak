<p align="center">
  <img src="../logo.png" alt="DataCloak Logo" width="150"/>
</p>

# DataCloak Core Library

High-performance Rust library for large-scale data obfuscation with automatic PII detection and LLM-based churn prediction.

## Features

- **Automatic PII Detection**: Scans data samples to detect and recommend PII patterns
- **High-Performance Obfuscation**: SIMD-accelerated regex matching with concurrent processing
- **Streaming Processing**: Handles 20+ GB datasets with memory-efficient streaming
- **Batch LLM Integration**: Efficient batch processing with rate limiting and retries
- **Flexible Data Sources**: PostgreSQL, CSV, Parquet, JSON support
- **Secure Caching**: Encrypted storage for obfuscation mappings
- **Production Ready**: Comprehensive error handling, logging, and metrics

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Examples](#examples)
- [Running Tests](#running-tests)
- [Benchmarks](#benchmarks)
- [API Documentation](#api-documentation)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Contributing](#contributing)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
datacloak-core = "0.1.0"
```

Or clone and build from source:

```bash
git clone https://github.com/yourusername/datacloak.git
cd datacloak/datacloak-core
cargo build --release
```

## Quick Start

```rust
use datacloak_core::{DataCloak, DataCloakConfig, DataSource, PatternSet};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure DataCloak
    let mut config = DataCloakConfig::default();
    config.llm_config.api_key = std::env::var("OPENAI_API_KEY")?;
    
    let datacloak = DataCloak::new(config);
    
    // Detect PII patterns in your data
    let source = DataSource::csv("customer_data.csv".into());
    let detection = datacloak.detect_patterns(source).await?;
    
    println!("Detected patterns: {:?}", detection.detected_patterns);
    
    // Set patterns and analyze churn
    let patterns = PatternSet::default_pii().to_vec();
    datacloak.set_patterns(patterns)?;
    
    let results = datacloak.analyze_churn(
        DataSource::csv("customer_data.csv".into()),
        patterns,
        1000, // batch size
    ).await?;
    
    println!("Churn predictions: {:?}", results.predictions);
    
    Ok(())
}
```

## Examples

The library includes several examples demonstrating different use cases:

### Basic Usage Example

Shows pattern detection, obfuscation, and churn analysis:

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key"

# Run the basic example
cargo run --example basic_usage
```

### Running All Examples

```bash
# List all available examples
cargo run --example

# Run a specific example
cargo run --example basic_usage
cargo run --example streaming_example  # (if available)
```

### Creating Your Own Example

Create a new file in `examples/` directory:

```rust
// examples/my_example.rs
use datacloak_core::{DataCloak, DataCloakConfig};

fn main() {
    // Your example code here
}
```

## Running Tests

The library includes comprehensive unit and integration tests:

### Run All Tests

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run tests in parallel
cargo test -- --test-threads=4
```

### Run Specific Tests

```bash
# Run only unit tests
cargo test --lib

# Run only integration tests
cargo test --test integration_tests

# Run a specific test by name
cargo test test_pattern_detection

# Run tests matching a pattern
cargo test pattern
```

### Test Coverage

Generate test coverage report (requires cargo-tarpaulin):

```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Generate coverage report
cargo tarpaulin --out Html
```

## Benchmarks

The library includes performance benchmarks for critical operations:

### Available Benchmarks

1. **Obfuscation Benchmark** (`benches/obfuscation.rs`)
   - Measures obfuscation performance on small text batches

2. **Comprehensive Benchmark** (`benches/obfuscation_bench.rs`)
   - Tests batch processing performance
   - Nested JSON obfuscation
   - Cache operations
   - Pattern detection

### Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run a specific benchmark
cargo bench obfuscation

# Run benchmarks without optimization (for debugging)
cargo bench --no-run

# Run with specific features
cargo bench --features "simd"
```

### Benchmark Output

Benchmarks will output performance metrics:

```text
obfuscate_small_texts   time:   [1.2345 ms 1.2456 ms 1.2567 ms]
                        change: [-2.1234% -1.0123% +0.1234%] (p = 0.12 > 0.05)
                        No change in performance detected.
```

### Creating Custom Benchmarks

Add a new benchmark in `benches/` directory:

```rust
// benches/my_benchmark.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use datacloak_core::DataCloak;

fn bench_my_operation(c: &mut Criterion) {
    c.bench_function("my_operation", |b| {
        b.iter(|| {
            // Your benchmark code here
        })
    });
}

criterion_group!(benches, bench_my_operation);
criterion_main!(benches);
```

## API Documentation

Generate and view the API documentation:

```bash
# Generate documentation
cargo doc

# Generate and open in browser
cargo doc --open

# Include private items
cargo doc --document-private-items

# Generate documentation for all dependencies
cargo doc --no-deps
```

## Architecture

```text
datacloak-core/
├── src/
│   ├── lib.rs          # Main library interface
│   ├── detector.rs     # Automatic PII pattern detection
│   ├── obfuscator.rs   # High-performance obfuscation engine
│   ├── streaming.rs    # Chunk-based streaming processor
│   ├── llm_batch.rs    # Batch LLM client with retries
│   ├── cache.rs        # Secure token mapping cache
│   ├── data_source.rs  # Data source abstractions
│   ├── patterns.rs     # Pattern definitions and types
│   └── errors.rs       # Error types
├── tests/
│   └── integration_tests.rs  # Integration test suite
├── benches/
│   ├── obfuscation.rs       # Basic obfuscation benchmarks
│   └── obfuscation_bench.rs # Comprehensive benchmarks
├── examples/
│   └── basic_usage.rs       # Basic usage example
└── config/
    └── default_rules.json   # Default obfuscation rules
```

## Configuration

### DataCloakConfig

```rust
DataCloakConfig {
    // Processing settings
    batch_size: 1000,              // Records per batch
    max_concurrency: 4,            // Concurrent operations
    
    // Cache settings
    cache_ttl: Duration::from_secs(3600),
    cache_encryption_key: "your-32-byte-key",
    
    // LLM settings
    llm_config: LlmBatchConfig {
        endpoint: "https://api.openai.com/v1/chat/completions",
        api_key: "your-api-key",
        model: "gpt-3.5-turbo",
        temperature: 0.3,
        max_retries: 3,
        timeout: Duration::from_secs(30),
        rate_limit: Some(10), // requests per second
    },
    
    // Pattern detection
    detection_confidence_threshold: 0.8
}
```

### Environment Variables

```bash
# Required for LLM integration
export OPENAI_API_KEY="your-api-key"

# Optional configuration
export DATACLOAK_CACHE_DIR="/tmp/datacloak"
export DATACLOAK_LOG_LEVEL="info"
export DATACLOAK_MAX_CONCURRENCY="8"
```

## Performance Tips

1. **Batch Size**: Adjust `batch_size` based on your memory constraints
2. **Concurrency**: Set `max_concurrency` to number of CPU cores
3. **Streaming**: Use streaming for files larger than available RAM
4. **Caching**: Enable caching for repeated obfuscation operations
5. **Pattern Compilation**: Compile patterns once and reuse

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use streaming
2. **Slow Performance**: Check regex complexity and enable SIMD
3. **LLM Timeouts**: Increase timeout or reduce batch size
4. **Pattern Misses**: Lower confidence threshold or add custom patterns

### Debug Logging

Enable debug logging:

```bash
RUST_LOG=datacloak_core=debug cargo run
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Run clippy and fix warnings
6. Submit a pull request

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/datacloak.git
cd datacloak/datacloak-core

# Install development tools
rustup component add clippy rustfmt

# Run checks
cargo fmt --check
cargo clippy -- -D warnings
cargo test
cargo bench
```

## License

MIT License - see LICENSE file for details
