# DataCloak API Service

RESTful API service for DataCloak multi-field sentiment analysis.

## Overview

This service provides v2 API endpoints for:
- Column profiling and text candidate identification
- Multi-field sentiment analysis with streaming results
- Runtime and cost estimation

## API Endpoints

### Profile Columns
```
POST /api/v2/profile
```
Analyzes all columns in a file and returns ranked candidates suitable for sentiment analysis.

**Request:**
```json
{
  "file_id": "uuid"
}
```

**Response:**
```json
{
  "candidates": [
    {
      "name": "description",
      "index": 0,
      "ml_score": 0.95,
      "graph_score": 0.85,
      "final_score": 0.90,
      "features": {
        "text_length_avg": 150.5,
        "text_length_std": 45.2,
        "word_count_avg": 25.3,
        "unique_ratio": 0.85,
        "pattern_score": 0.75,
        "entropy": 4.2
      }
    }
  ],
  "total_columns": 10,
  "profiling_time_ms": 250
}
```

### Analyze Multiple Fields
```
POST /api/v2/analyze
```
Starts a multi-field analysis job and returns a Server-Sent Events stream.

**Request:**
```json
{
  "file_id": "uuid",
  "selected_columns": ["description", "comments"],
  "options": {
    "chain_type": "sentiment",
    "batch_size": 100,
    "max_concurrent_requests": 10
  }
}
```

**Response:** Server-Sent Events stream
```
data: {"record_id": "rec_1", "column": "description", "result": {"sentiment": "positive", "confidence": 0.85}, "timestamp": "2024-01-01T00:00:00Z"}

data: {"record_id": "rec_1", "column": "comments", "result": {"sentiment": "neutral", "confidence": 0.75}, "timestamp": "2024-01-01T00:00:01Z"}
```

### Estimate Runtime
```
POST /api/v2/estimate
```
Provides estimated runtime, cost, and token usage for analysis.

**Request:**
```json
{
  "file_id": "uuid",
  "selected_columns": ["col1", "col2"],
  "chain_type": "sentiment"
}
```

**Response:**
```json
{
  "estimated_seconds": 300,
  "confidence_lower": 250,
  "confidence_upper": 350,
  "estimated_cost": 0.25,
  "total_rows": 10000,
  "total_tokens_estimate": 50000
}
```

## API Documentation

Interactive API documentation is available at:
- Swagger UI: http://localhost:8080/swagger-ui/

## Running the Service

```bash
cd datacloak-api
cargo run
```

The service will start on http://localhost:8080

## Testing

```bash
cd datacloak-api
cargo test
```

## Performance Requirements

- API response time: <500ms p99
- Streaming latency: <50ms per result
- Database queries: <100ms
- Cache hit rate: >80%
- Concurrent runs: 100+

## Current Status

This is a skeleton implementation with:
- ✅ Basic endpoint structure
- ✅ Request/response models
- ✅ OpenAPI documentation
- ✅ Test framework
- ❌ Database integration (TODO)
- ❌ Actual profiling logic (TODO)
- ❌ SSE streaming implementation (TODO)
- ❌ Worker coordination (TODO)
- ❌ Cache layer (TODO)