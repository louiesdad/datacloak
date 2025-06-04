# DataCloak

<p align="center">
  <img src="logo.jpg" alt="DataCloak Logo" width="150"/>
</p>

<p align="center">
  <strong>High-performance data obfuscation library with automatic PII detection and LLM-based analytics</strong>
</p>


## Overview

DataCloak is a Rust-based library designed for large-scale data obfuscation with automatic PII detection and LLM-powered analytics. It's built to handle 20+ GB datasets efficiently while maintaining data privacy through intelligent obfuscation and de-obfuscation workflows.

### Key Features

- 🔍 **Automatic PII Detection**: ML-powered pattern detection with confidence scoring
- ⚡ **High Performance**: SIMD-accelerated regex matching, parallel processing
- 📊 **Large-Scale Processing**: Stream 20+ GB datasets with minimal memory footprint
- 🤖 **LLM Integration**: Batch processing for churn prediction and analytics
- 🔐 **Secure Caching**: Encrypted token mappings with persistence
- 🌐 **Multi-Language Support**: FFI bindings for Python, gRPC for remote calls
- 📈 **Production Ready**: Comprehensive error handling, retry logic, and monitoring

## Architecture

```
datacloak/
├── datacloak-core/      # Core Rust library
├── datacloak-ffi/       # C FFI bindings
├── datacloak-grpc/      # gRPC service wrapper
├── datacloak-py/        # Python bindings
└── datacloak-web/       # Web service API
```

## Quick Start

### Rust Library

Add to your `Cargo.toml`:

```toml
[dependencies]
datacloak = "0.1.0"
```

Basic usage:

```rust
use datacloak::{DataCloak, DataCloakConfig, DataSource, PatternSet};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure DataCloak
    let mut config = DataCloakConfig::default();
    config.llm_config.api_key = std::env::var("OPENAI_API_KEY")?;
    
    let datacloak = DataCloak::new(config);
    
    // Detect patterns automatically
    let source = DataSource::csv("customers.csv".into());
    let detection = datacloak.detect_patterns(source.clone()).await?;
    
    // Run churn analysis
    let patterns = detection.recommended_patterns();
    let results = datacloak.analyze_churn(source, patterns, Some(1000)).await?;
    
    println!("Average churn probability: {:.2}%", 
        results.average_churn_probability * 100.0);
    
    Ok(())
}
```

### Python Library

Install via pip:

```bash
pip install datacloak
```

Usage:

```python
import datacloak

# Initialize
dc = datacloak.DataCloak(
    llm_endpoint="https://api.openai.com/v1/chat/completions",
    api_key=os.environ["OPENAI_API_KEY"]
)

# Detect patterns
patterns = dc.detect_patterns(
    source=datacloak.CSV("customers.csv")
)

# Run analysis
results = dc.analyze_churn(
    source=datacloak.CSV("customers.csv"),
    patterns=patterns.recommended,
    batch_size=1000
)

print(f"High risk customers: {results.high_risk_customers}")
```

### Web API

Start the server:

```bash
docker run -p 8080:8080 -e OPENAI_API_KEY=$OPENAI_API_KEY datacloak/web
```

Make requests:

```bash
curl -X POST http://localhost:8080/api/v1/detect \
  -H "Content-Type: application/json" \
  -d '{
    "source": {
      "type": "csv",
      "path": "/data/customers.csv"
    }
  }'
```

## Performance

DataCloak is designed for high-performance data processing:

| Dataset Size | Processing Time | Memory Usage |
|-------------|-----------------|--------------|
| 1 GB        | ~30 seconds     | 200 MB       |
| 10 GB       | ~5 minutes      | 500 MB       |
| 20 GB       | ~10 minutes     | 800 MB       |

*Benchmarked on AWS c5.4xlarge instance*

## Supported Data Sources

- **PostgreSQL**: Cursor-based streaming for large queries
- **CSV**: Memory-mapped file processing
- **Parquet**: Columnar data format support
- **JSON**: In-memory processing for smaller datasets

## Pattern Types

Built-in patterns with regex and ML-based detection:

- Email addresses
- Phone numbers
- Social Security Numbers (SSN)
- Credit card numbers
- IP addresses
- Dates of birth
- Medical record numbers
- Driver's license numbers
- Bank account numbers
- Passport numbers
- Names (ML-based)
- Addresses (ML-based)
- Custom patterns

## Security

- **Encryption**: AES-256 for cache persistence
- **API Security**: Token-based authentication
- **Data Privacy**: All PII remains obfuscated during LLM processing
- **Audit Logging**: Comprehensive activity tracking

## Development

### Prerequisites

- Rust 1.70+ (for core library)
- Python 3.8+ (for Python bindings)
- Docker (for containerized deployment)

### Building from Source

```bash
# Clone repository
git clone https://github.com/yourusername/datacloak.git
cd datacloak

# Build core library
cd datacloak-core
cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo bench
```

### Running Tests

```bash
# Unit tests
cargo test

# Integration tests
cargo test --test '*'

# Benchmarks
cargo bench

# Coverage
cargo tarpaulin --out Html
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Documentation

- [API Documentation](https://docs.rs/datacloak)
- [Architecture Guide](docs/architecture.md)
- [Performance Tuning](docs/performance.md)
- [Security Best Practices](docs/security.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with ❤️ by the DataCloak team
- Powered by Rust 🦀
- Special thanks to all contributors

## Support

- 📧 Email: support@datacloak.io
- 💬 Discord: [Join our community](https://discord.gg/datacloak)
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/datacloak/issues)
