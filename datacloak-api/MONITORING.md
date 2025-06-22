# DataCloak API Monitoring and Observability

This document describes the comprehensive monitoring and observability features implemented in the DataCloak API service.

## Overview

The monitoring system provides:
- **Prometheus Metrics**: Comprehensive metrics collection for APIs, performance, resources, and business KPIs
- **Distributed Tracing**: Request tracing across service components
- **Health Checks**: Kubernetes-ready health and readiness probes
- **SLO Monitoring**: Service Level Objective tracking and compliance
- **Structured Logging**: JSON-formatted logs with correlation IDs

## Endpoints

### Metrics
- `GET /monitoring/metrics` - Prometheus metrics in text format
- Content-Type: `text/plain; version=0.0.4; charset=utf-8`

### Health Checks
- `GET /monitoring/health` - Overall service health status
- `GET /monitoring/ready` - Kubernetes readiness probe
- `GET /monitoring/live` - Kubernetes liveness probe

### Observability
- `GET /monitoring/slo` - SLO compliance dashboard
- `GET /monitoring/traces` - Active traces information

## Metrics Collected

### API Metrics
```
datacloak_api_requests_total - Total number of API requests
datacloak_api_request_duration_seconds - Duration of API requests
datacloak_api_errors_total - Total number of API errors
```

### Performance Metrics
```
datacloak_cache_hit_rate - Current cache hit rate (0-1)
datacloak_active_streams - Number of active SSE streams
datacloak_worker_utilization - Worker utilization percentage (0-100)
datacloak_analysis_duration_seconds - Duration of analysis operations
```

### Resource Metrics
```
datacloak_memory_usage_bytes - Current memory usage in bytes
datacloak_active_connections - Number of active database connections
datacloak_queue_depth - Current analysis queue depth
```

### Business Metrics
```
datacloak_files_analyzed_total - Total number of files analyzed
datacloak_columns_processed_total - Total number of columns processed
datacloak_successful_analyses_total - Total number of successful analyses
datacloak_failed_analyses_total - Total number of failed analyses
```

## Distributed Tracing

### Request Tracing
Every API request automatically generates a trace with:
- Unique trace ID
- Request method and path
- Request/response correlation
- Duration and outcome tracking

### Custom Spans
Services can create custom spans for internal operations:
```rust
let span_id = tracing_service.start_span(
    parent_trace_id,
    "database_query",
    metadata
).await;
```

### Trace Propagation
Request IDs are automatically:
- Generated for incoming requests
- Extracted from `x-request-id` header if present
- Propagated through internal service calls
- Included in response headers

## Health Checks

### Health Status Response
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "uptime_seconds": 3600,
  "checks": {
    "database": {
      "status": "healthy",
      "message": "Database connection pool healthy",
      "response_time_ms": 5
    },
    "cache": {
      "status": "healthy",
      "message": "Cache layer responding",
      "response_time_ms": 2
    },
    "workers": {
      "status": "healthy",
      "message": "4/4 workers healthy",
      "response_time_ms": 1
    }
  }
}
```

### Readiness Criteria
Service is ready when:
- Database connection is healthy
- Cache layer is responding
- Critical configuration is loaded

### Liveness Criteria
Service is alive when:
- HTTP server is responding
- Main event loop is not blocked

## SLO Monitoring

### Default SLOs
- **Error Rate**: < 1% (99% success rate)
- **Latency P95**: < 500ms for API endpoints
- **Availability**: > 99.9% uptime

### SLO Response
```json
{
  "slo_compliance": {
    "error_rate_compliant": true,
    "latency_compliant": true,
    "availability_compliant": true,
    "overall_score": 1.0
  },
  "thresholds": {
    "error_rate": 0.01,
    "latency_p95_ms": 500.0,
    "availability": 0.999
  }
}
```

## Structured Logging

### Log Format
All logs are emitted in JSON format with:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "message": "API request completed",
  "trace_id": "550e8400-e29b-41d4-a716-446655440000",
  "endpoint": "/api/v2/profile",
  "method": "POST",
  "duration_ms": 125.5,
  "status_code": 200
}
```

### Log Levels
- `ERROR`: Service errors, failed operations
- `WARN`: Degraded performance, recoverable issues
- `INFO`: Normal operations, request completion
- `DEBUG`: Detailed internal operations
- `TRACE`: Very detailed debugging information

## Deployment Integration

### Kubernetes Configuration
```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: datacloak-api
    image: datacloak-api:latest
    ports:
    - containerPort: 8080
    livenessProbe:
      httpGet:
        path: /monitoring/live
        port: 8080
      initialDelaySeconds: 30
      periodSeconds: 10
    readinessProbe:
      httpGet:
        path: /monitoring/ready
        port: 8080
      initialDelaySeconds: 5
      periodSeconds: 5
```

### Prometheus Scraping
```yaml
apiVersion: v1
kind: ServiceMonitor
metadata:
  name: datacloak-api
spec:
  selector:
    matchLabels:
      app: datacloak-api
  endpoints:
  - port: http
    path: /monitoring/metrics
    interval: 30s
```

### Grafana Dashboard
Key dashboard panels:
- API request rate and latency
- Error rate trending
- Cache hit rate
- Worker utilization
- Analysis throughput
- SLO compliance status

## Development

### Adding Custom Metrics
```rust
// In your service
let custom_counter = Counter::with_opts(
    CounterOpts::new("custom_operations_total", "Custom operations")
)?;
registry.register(Box::new(custom_counter.clone()))?;

// Usage
custom_counter.inc();
```

### Adding Custom Health Checks
```rust
impl HealthService {
    async fn check_my_component(&self) -> ComponentHealth {
        // Your health check logic
        ComponentHealth {
            status: "healthy".to_string(),
            message: Some("Component is responding".to_string()),
            response_time_ms: Some(response_time),
        }
    }
}
```

### Custom Tracing
```rust
// Start operation trace
let trace_id = monitoring.tracing.start_trace(
    "my_operation",
    metadata
).await;

// Add contextual data
monitoring.tracing.add_trace_metadata(
    trace_id,
    "key".to_string(),
    "value".to_string()
).await;

// Finish trace
monitoring.tracing.finish_trace(
    trace_id,
    success,
    error_message
).await;
```

## Environment Variables

```bash
# Logging configuration
RUST_LOG=info,datacloak_api=debug

# Metrics configuration
METRICS_ENABLED=true
METRICS_NAMESPACE=datacloak

# Tracing configuration
TRACING_ENABLED=true
TRACE_SAMPLE_RATE=0.1

# Health check intervals
HEALTH_CHECK_INTERVAL=30
HEALTH_CHECK_TIMEOUT=5
```

## Troubleshooting

### High Error Rate
1. Check `/monitoring/health` for component status
2. Review error logs for patterns
3. Check database and cache connectivity
4. Verify worker coordinator status

### High Latency
1. Check cache hit rate metrics
2. Monitor database connection pool
3. Review worker utilization
4. Check for resource constraints

### Failed Health Checks
1. Check individual component health in `/monitoring/health`
2. Verify database connectivity
3. Test cache layer operations
4. Check worker coordinator status

### Missing Metrics
1. Verify Prometheus scraping configuration
2. Check `/monitoring/metrics` endpoint directly
3. Review middleware configuration
4. Check service registration

## Security Considerations

- Metrics endpoints do not expose sensitive data
- Health checks provide operational status only
- Trace IDs are UUIDs with no business logic correlation
- Log sanitization prevents credential exposure
- Access control should be implemented at infrastructure level