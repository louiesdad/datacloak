# DataCloak Client Integration Guide

## Overview

DataCloak is a high-performance data analysis system that provides multi-field sentiment analysis with automatic column discovery, PII detection, and secure data obfuscation. This guide will help you integrate DataCloak into your infrastructure.

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [API Integration](#api-integration)
4. [CLI Usage](#cli-usage)
5. [Monitoring & Observability](#monitoring--observability)
6. [Security Configuration](#security-configuration)
7. [Performance Tuning](#performance-tuning)
8. [Troubleshooting](#troubleshooting)

## Installation

### Option 1: Docker (Recommended)

```bash
# Pull the official DataCloak image
docker pull datacloak/datacloak:latest

# Run DataCloak API server
docker run -d \
  --name datacloak-api \
  -p 8080:8080 \
  -e DATABASE_URL="postgres://user:pass@host:5432/datacloak" \
  -e REDIS_URL="redis://host:6379" \
  -e LLM_API_KEY="your-openai-api-key" \
  -e DATACLOAK_CACHE_KEY="your-32-byte-hex-key" \
  -v /path/to/data:/data \
  datacloak/datacloak:latest

# Run DataCloak CLI
docker run --rm -it \
  -v /path/to/data:/data \
  -e LLM_API_KEY="your-openai-api-key" \
  datacloak/datacloak-cli:latest \
  analyze --file /data/customer_data.csv --auto-discover
```

### Option 2: Binary Installation

```bash
# Download latest release
curl -L https://github.com/datacloak/releases/latest/download/datacloak-linux-amd64.tar.gz | tar xz

# Install CLI
sudo mv datacloak /usr/local/bin/
sudo chmod +x /usr/local/bin/datacloak

# Install API server
sudo mv datacloak-api /usr/local/bin/
sudo chmod +x /usr/local/bin/datacloak-api
```

### Option 3: Build from Source

```bash
# Clone repository
git clone https://github.com/datacloak/datacloak.git
cd datacloak

# Build all components
cargo build --release --all

# Binaries will be in target/release/
```

## Configuration

### Environment Variables

Create a `.env` file or set these environment variables:

```bash
# Required
DATABASE_URL=postgres://user:password@localhost:5432/datacloak
LLM_API_KEY=sk-your-openai-api-key
DATACLOAK_CACHE_KEY=0123456789abcdef0123456789abcdef  # 32-byte hex key

# Optional
REDIS_URL=redis://localhost:6379
LOG_LEVEL=info
PORT=8080
WORKER_COUNT=4
MAX_BATCH_SIZE=1000
RATE_LIMIT_PER_SECOND=3
MEMORY_LIMIT_MB=1024
```

### Database Setup

```sql
-- Run migrations
psql -U postgres -d datacloak < migrations/V1__initial_schema.sql
psql -U postgres -d datacloak < migrations/V2__multi_column_support.sql
```

### Configuration File

Create `datacloak.yaml` for advanced configuration:

```yaml
server:
  host: 0.0.0.0
  port: 8080
  workers: 4

llm:
  provider: openai
  model: gpt-4
  api_key: ${LLM_API_KEY}
  rate_limit: 3
  timeout_seconds: 30
  max_retries: 3

processing:
  batch_size: 1000
  max_concurrent_analyses: 10
  chunk_size_mb: 100
  stream_buffer_size: 10000

cache:
  type: redis
  url: ${REDIS_URL}
  ttl_seconds: 3600
  max_size_mb: 1024

monitoring:
  metrics_port: 9090
  trace_sample_rate: 0.1
  log_level: info
  log_format: json
```

## API Integration

### Authentication

Include your API key in the Authorization header:

```bash
Authorization: Bearer your-api-key
```

### 1. Profile Columns (Auto-Discovery)

```bash
POST /api/v2/profile
Content-Type: application/json

{
  "file_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Response:**
```json
{
  "candidates": [
    {
      "column": "customer_feedback",
      "ml_prob": 0.95,
      "graph_score": 0.88,
      "final_score": 0.92,
      "null_pct": 0.02,
      "mean_len": 125.5,
      "predicted_type": "TextLong"
    },
    {
      "column": "order_id",
      "ml_prob": 0.15,
      "graph_score": 0.10,
      "final_score": 0.13,
      "null_pct": 0.0,
      "mean_len": 12.0,
      "predicted_type": "Identifier"
    }
  ],
  "total_columns": 15,
  "recommended": 3
}
```

### 2. Estimate Analysis Time/Cost

```bash
POST /api/v2/estimate
Content-Type: application/json

{
  "file_id": "550e8400-e29b-41d4-a716-446655440000",
  "selected_columns": ["customer_feedback", "support_notes", "reviews"],
  "chain_type": "sentiment"
}
```

**Response:**
```json
{
  "estimated_seconds": 300,
  "confidence_lower": 250,
  "confidence_upper": 350,
  "estimated_cost": 2.50,
  "total_rows": 50000,
  "total_tokens_estimate": 250000
}
```

### 3. Start Multi-Field Analysis (Streaming)

```bash
POST /api/v2/analyze
Content-Type: application/json
Accept: text/event-stream

{
  "file_id": "550e8400-e29b-41d4-a716-446655440000",
  "selected_columns": ["customer_feedback", "support_notes"],
  "options": {
    "chain_type": "sentiment",
    "include_confidence": true,
    "batch_size": 100
  }
}
```

**Server-Sent Events Response:**
```
event: progress
data: {"processed":0,"total":50000,"percentage":0.0,"eta_seconds":300}

event: result
data: {"record_id":"rec_1","column":"customer_feedback","result":{"sentiment":"positive","confidence":0.92},"sequence":1}

event: result
data: {"record_id":"rec_1","column":"support_notes","result":{"sentiment":"neutral","confidence":0.78},"sequence":2}

event: progress
data: {"processed":100,"total":50000,"percentage":0.2,"eta_seconds":298}

event: complete
data: {"total_processed":50000,"duration_ms":295000}
```

### Client SDK Example (Python)

```python
import requests
import sseclient
import json

class DataCloakClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def profile_columns(self, file_id):
        """Profile columns to find text-heavy candidates"""
        response = requests.post(
            f'{self.base_url}/api/v2/profile',
            headers=self.headers,
            json={'file_id': file_id}
        )
        return response.json()
    
    def estimate_analysis(self, file_id, columns):
        """Get time and cost estimates"""
        response = requests.post(
            f'{self.base_url}/api/v2/estimate',
            headers=self.headers,
            json={
                'file_id': file_id,
                'selected_columns': columns,
                'chain_type': 'sentiment'
            }
        )
        return response.json()
    
    def analyze_streaming(self, file_id, columns, callback):
        """Start analysis with streaming results"""
        response = requests.post(
            f'{self.base_url}/api/v2/analyze',
            headers={**self.headers, 'Accept': 'text/event-stream'},
            json={
                'file_id': file_id,
                'selected_columns': columns,
                'options': {'chain_type': 'sentiment'}
            },
            stream=True
        )
        
        client = sseclient.SSEClient(response)
        for event in client.events():
            data = json.loads(event.data)
            callback(event.event, data)

# Usage example
client = DataCloakClient('http://localhost:8080', 'your-api-key')

# 1. Profile columns
profile = client.profile_columns('file-uuid')
text_columns = [c['column'] for c in profile['candidates'] if c['final_score'] > 0.7]

# 2. Get estimate
estimate = client.estimate_analysis('file-uuid', text_columns)
print(f"Estimated time: {estimate['estimated_seconds']}s, Cost: ${estimate['estimated_cost']}")

# 3. Start analysis
def handle_event(event_type, data):
    if event_type == 'result':
        print(f"Record {data['record_id']}, Column {data['column']}: {data['result']['sentiment']}")
    elif event_type == 'progress':
        print(f"Progress: {data['percentage']:.1f}%")

client.analyze_streaming('file-uuid', text_columns, handle_event)
```

## CLI Usage

### Basic Commands

```bash
# Profile columns in a CSV file
datacloak profile --file customer_data.csv --output json

# Analyze specific columns
datacloak analyze --file customer_data.csv \
  --columns "feedback,comments,reviews" \
  --output csv > results.csv

# Auto-discover and analyze text columns
datacloak analyze --file customer_data.csv \
  --auto-discover \
  --threshold 0.75 \
  --output stream

# Dry-run to see what would be processed
datacloak analyze --file large_dataset.csv \
  --auto-discover \
  --dry-run

# Detect PII patterns
datacloak detect --patterns email,ssn,phone \
  --data-source customer_data.csv

# Obfuscate sensitive data
datacloak obfuscate --patterns patterns.json \
  --data-source input.csv \
  --output obfuscated.csv
```

### Advanced Usage

```bash
# Use mock LLM for testing
datacloak analyze --file test.csv \
  --columns text_field \
  --mock-llm

# Run predefined test scenario
datacloak test-scenario --scenario customer-churn

# Custom API endpoint
datacloak --api-url http://datacloak-api:8080 \
  analyze --file data.csv --auto-discover
```

## Monitoring & Observability

### Metrics Endpoint

DataCloak exposes Prometheus metrics at `/monitoring/metrics`:

```bash
# Scrape metrics
curl http://localhost:8080/monitoring/metrics
```

Key metrics:
- `datacloak_columns_profiled_total` - Total columns profiled
- `datacloak_analysis_runs_total` - Total analysis runs
- `datacloak_rows_processed_total` - Total rows processed
- `datacloak_llm_api_errors_total` - LLM API errors
- `datacloak_streaming_latency_seconds` - Streaming response latency
- `datacloak_memory_usage_bytes` - Current memory usage

### Health Checks

```bash
# Overall health
GET /monitoring/health

# Response
{
  "status": "healthy",
  "version": "1.0.0",
  "components": {
    "database": {
      "status": "healthy",
      "latency_ms": 2
    },
    "cache": {
      "status": "healthy",
      "hit_rate": 0.85
    },
    "workers": {
      "status": "healthy",
      "active": 2,
      "total": 4
    }
  }
}

# Kubernetes probes
GET /monitoring/ready   # Readiness probe
GET /monitoring/live    # Liveness probe
```

### Distributed Tracing

DataCloak supports OpenTelemetry tracing. Configure your trace collector:

```yaml
# datacloak.yaml
tracing:
  enabled: true
  endpoint: http://jaeger:14268/api/traces
  service_name: datacloak
  sample_rate: 0.1
```

View traces at `/monitoring/traces` or in your tracing backend.

### Structured Logging

All logs are in JSON format with trace correlation:

```json
{
  "timestamp": "2024-01-15T10:30:45Z",
  "level": "info",
  "message": "Analysis started",
  "trace_id": "1234567890abcdef",
  "span_id": "abcdef1234",
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "columns": ["feedback", "comments"],
  "estimated_time": 300
}
```

Configure log aggregation to your preferred backend (ELK, Splunk, etc).

### SLO Monitoring

Monitor service level objectives at `/monitoring/slo`:

```json
{
  "error_rate": {
    "current": 0.005,
    "threshold": 0.01,
    "compliant": true
  },
  "latency_p95": {
    "current_ms": 450,
    "threshold_ms": 500,
    "compliant": true
  },
  "availability": {
    "current": 0.999,
    "threshold": 0.999,
    "compliant": true
  }
}
```

## Security Configuration

### Rate Limiting

Configure rate limits per endpoint:

```yaml
rate_limits:
  profile: 10  # requests per second
  analyze: 3
  estimate: 20
  default: 5
```

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 3
X-RateLimit-Remaining: 2
X-RateLimit-Reset: 1673789045
Retry-After: 60  # When rate limited
```

### Encryption

All cached data is encrypted with AES-256-GCM:

```bash
# Generate a secure key
openssl rand -hex 32

# Set in environment
export DATACLOAK_CACHE_KEY=your-generated-key
```

### API Key Management

```bash
# Generate API keys
datacloak-admin create-key --name "Production App" --scopes "read,write"

# Revoke keys
datacloak-admin revoke-key --key-id "key_123456"

# List active keys
datacloak-admin list-keys
```

## Performance Tuning

### Memory Configuration

```yaml
performance:
  max_memory_mb: 2048
  chunk_size_mb: 100
  stream_buffer_size: 10000
  cache_size_mb: 512
```

### Concurrency Settings

```yaml
concurrency:
  worker_count: 8  # CPU cores
  max_parallel_analyses: 20
  db_pool_size: 20
  redis_pool_size: 10
```

### Large File Processing

For files over 20GB:

```yaml
large_files:
  enable_streaming: true
  chunk_size_mb: 200
  max_chunks_in_memory: 5
  use_memory_mapping: true
```

## Troubleshooting

### Common Issues

**1. High Memory Usage**
```bash
# Check memory metrics
curl http://localhost:8080/monitoring/metrics | grep memory

# Adjust chunk size
export CHUNK_SIZE_MB=50
```

**2. Slow Performance**
```bash
# Enable performance profiling
export RUST_LOG=datacloak=debug,performance=trace

# Check slow queries
SELECT query, mean_time 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;
```

**3. LLM API Errors**
```bash
# Check rate limiting
curl http://localhost:8080/monitoring/metrics | grep llm_api_errors

# Increase retry configuration
export LLM_MAX_RETRIES=5
export LLM_BACKOFF_MS=1000
```

### Debug Mode

Enable detailed debugging:

```bash
# Set debug logging
export RUST_LOG=debug
export RUST_BACKTRACE=1

# Enable trace sampling
export TRACE_SAMPLE_RATE=1.0

# Run with verbose output
datacloak --verbose analyze --file test.csv --debug
```

### Support

- Documentation: https://docs.datacloak.io
- API Reference: http://localhost:8080/swagger-ui/
- GitHub Issues: https://github.com/datacloak/datacloak/issues
- Email: support@datacloak.io

## Example Integration Workflow

```python
# Complete integration example
import datacloak

# Initialize client
client = datacloak.Client(
    api_url="http://datacloak-api:8080",
    api_key=os.getenv("DATACLOAK_API_KEY")
)

# Upload file
file_id = client.upload_file("customer_data.csv")

# Profile columns
profile = client.profile_columns(file_id)
print(f"Found {profile['recommended']} text columns out of {profile['total_columns']}")

# Select high-scoring columns
text_columns = [
    c['column'] 
    for c in profile['candidates'] 
    if c['final_score'] > 0.7
]

# Get estimate
estimate = client.estimate(file_id, text_columns)
print(f"Analysis will take ~{estimate['estimated_seconds']}s and cost ~${estimate['estimated_cost']}")

# Start analysis with progress tracking
analysis = client.analyze(
    file_id=file_id,
    columns=text_columns,
    stream=True
)

for event in analysis.stream():
    if event.type == 'progress':
        print(f"Progress: {event.data['percentage']:.1f}%")
    elif event.type == 'result':
        # Process each result
        record_id = event.data['record_id']
        sentiment = event.data['result']['sentiment']
        confidence = event.data['result']['confidence']
        
        # Store in your database
        db.save_sentiment(record_id, sentiment, confidence)

print("Analysis complete!")
```

## License

DataCloak is licensed under the MIT License. See LICENSE file for details.