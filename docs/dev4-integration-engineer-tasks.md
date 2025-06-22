# Developer 4: Integration Engineer Tasks

## Role Overview
Update the CLI for multi-field support, build comprehensive testing framework, implement monitoring/observability, and ensure smooth integration of all components.

## TDD Learning Requirements
Master these TDD concepts from the reference guide:
1. **Testing edge cases**: Boundary conditions and error scenarios
2. **Test organization**: Arrange-Act-Assert and Given-When-Then
3. **Refactoring patterns**: Remove duplication while keeping tests green
4. **Property-based testing**: Generate test cases automatically

## Sprint 1 Tasks (Days 1-5)

### Task 4.1: CLI Multi-Field Support
**Story**: As a user, I need the CLI to support analyzing multiple fields with auto-discovery.

**TDD CLI Tests First**:
```rust
#[test]
fn test_cli_profile_command() {
    let mut cmd = Command::cargo_bin("datacloak").unwrap();
    cmd.arg("profile")
       .arg("--file").arg("test_data.csv")
       .arg("--output").arg("json");
    
    let output = cmd.output().unwrap();
    assert!(output.status.success());
    
    let result: ProfileOutput = serde_json::from_slice(&output.stdout).unwrap();
    assert!(result.candidates.len() > 0);
    assert!(result.candidates[0].score >= result.candidates[1].score);
}

#[test]
fn test_cli_multi_field_analyze() {
    let mut cmd = Command::cargo_bin("datacloak").unwrap();
    cmd.arg("analyze")
       .arg("--file").arg("test_data.csv")
       .arg("--columns").arg("description,comments,feedback")
       .arg("--output").arg("csv");
    
    let output = cmd.output().unwrap();
    assert!(output.status.success());
    
    // Verify CSV has all columns
    let csv_output = String::from_utf8(output.stdout).unwrap();
    assert!(csv_output.contains("description_sentiment"));
    assert!(csv_output.contains("comments_sentiment"));
    assert!(csv_output.contains("feedback_sentiment"));
}

#[test]
fn test_cli_auto_discovery_mode() {
    let mut cmd = Command::cargo_bin("datacloak").unwrap();
    cmd.arg("analyze")
       .arg("--file").arg("test_data.csv")
       .arg("--auto-discover")
       .arg("--threshold").arg("0.7");
    
    let output = cmd.output().unwrap();
    assert!(output.status.success());
    
    // Should analyze high-scoring columns automatically
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("Auto-discovered"));
    assert!(stderr.contains("columns for analysis"));
}

#[test]
fn test_cli_dry_run_with_estimate() {
    let mut cmd = Command::cargo_bin("datacloak").unwrap();
    cmd.arg("analyze")
       .arg("--file").arg("large_file.csv")
       .arg("--columns").arg("col1,col2,col3")
       .arg("--dry-run");
    
    let output = cmd.output().unwrap();
    assert!(output.status.success());
    
    let result: DryRunOutput = serde_json::from_slice(&output.stdout).unwrap();
    assert!(result.estimated_time_seconds > 0);
    assert!(result.estimated_cost_usd > 0.0);
    assert_eq!(result.selected_columns.len(), 3);
}
```

**CLI Implementation Steps**:
1. Add `profile` subcommand
2. Update `analyze` for multiple columns
3. Add auto-discovery flag
4. Implement dry-run with estimates
5. Add progress bar for streaming

**Updated CLI Interface**:
```rust
#[derive(Parser)]
#[command(name = "datacloak")]
#[command(about = "Multi-field sentiment analysis with auto-discovery")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Profile columns to find text-heavy candidates
    Profile {
        #[arg(short, long)]
        file: PathBuf,
        
        #[arg(short, long, default_value = "json")]
        output: OutputFormat,
        
        #[arg(long)]
        ml_only: bool,
        
        #[arg(long)]
        graph_only: bool,
    },
    
    /// Analyze sentiment in multiple fields
    Analyze {
        #[arg(short, long)]
        file: PathBuf,
        
        #[arg(short, long, value_delimiter = ',')]
        columns: Option<Vec<String>>,
        
        #[arg(long)]
        auto_discover: bool,
        
        #[arg(long, default_value = "0.7")]
        threshold: f32,
        
        #[arg(long)]
        dry_run: bool,
        
        #[arg(short, long, default_value = "json")]
        output: OutputFormat,
    },
}
```

**Deliverables**:
- Updated CLI with new commands
- Auto-discovery integration
- Progress reporting
- Help documentation

### Task 4.2: Integration Testing Framework
**Story**: As a developer, I need comprehensive integration tests for the multi-field system.

**Test Framework Design**:
```rust
#[tokio::test]
async fn test_scenario_customer_churn_analysis() {
    let scenario = TestScenario::load("customer_churn");
    let harness = TestHarness::new().await;
    
    // Upload test data
    let file_id = harness.upload_file(scenario.input_file).await;
    
    // Profile columns
    let profile = harness.profile_columns(file_id).await;
    assert!(profile.find_column("customer_feedback").score > 0.8);
    assert!(profile.find_column("support_tickets").score > 0.7);
    
    // Analyze with auto-discovery
    let analysis = harness.analyze_auto(file_id, 0.7).await;
    
    // Verify results match expected
    let results = analysis.collect_results().await;
    assert_eq!(results.len(), scenario.expected_rows);
    
    // Check sentiment distribution
    let sentiment_dist = results.sentiment_distribution("customer_feedback");
    assert!(sentiment_dist.positive > 0.3);
    assert!(sentiment_dist.negative < 0.4);
    
    // Verify performance
    assert!(analysis.elapsed_seconds < scenario.max_duration_seconds);
}

#[test]
fn test_property_based_column_detection() {
    proptest!(|(
        columns: Vec<ColumnSpec>,
        rows: u32 in 1000..100000u32,
    )| {
        let file = generate_csv_from_specs(&columns, rows);
        let profiler = ColumnProfiler::new();
        
        let candidates = profiler.profile_file(&file).unwrap();
        
        // Properties to verify
        let text_columns = columns.iter()
            .filter(|c| c.column_type == ColumnType::TextLong)
            .count();
        
        let discovered_text = candidates.iter()
            .filter(|c| c.final_score > 0.7)
            .count();
        
        // Should discover most text columns
        prop_assert!(discovered_text >= text_columns * 8 / 10);
        
        // Should not have false positives for numeric
        let numeric_false_positives = candidates.iter()
            .filter(|c| {
                c.final_score > 0.7 && 
                columns.iter().any(|spec| 
                    spec.name == c.name && spec.column_type == ColumnType::Numeric
                )
            })
            .count();
        prop_assert_eq!(numeric_false_positives, 0);
    });
}

#[tokio::test]
async fn test_large_file_processing() {
    let harness = TestHarness::new().await;
    
    // Generate 20GB file
    let large_file = generate_large_csv(20 * GB, 1000); // 1000 columns
    let file_id = harness.upload_file(large_file).await;
    
    // Profile should complete quickly
    let start = Instant::now();
    let profile = harness.profile_columns(file_id).await;
    assert!(start.elapsed().as_secs() < 10);
    
    // Select top 10 columns
    let selected = profile.top_n(10);
    
    // Analyze with monitoring
    let monitor = MemoryMonitor::new();
    let analysis = harness.analyze_columns(file_id, selected).await;
    
    // Stream results with memory tracking
    while let Some(batch) = analysis.next_batch().await {
        assert!(monitor.current_usage() < 1 * GB);
        assert!(batch.len() > 0);
    }
    
    // Verify completion
    assert_eq!(analysis.status(), AnalysisStatus::Completed);
}
```

**Test Scenarios**:
1. Customer churn with feedback columns
2. Financial fraud with transaction descriptions  
3. Medical records with clinical notes
4. Social media with posts and comments
5. E-commerce with product reviews

**Deliverables**:
- Test harness framework
- Scenario definitions
- Property-based tests
- Performance benchmarks

### Task 4.3: Mock LLM Server Enhancement
**Story**: As a developer, I need an enhanced mock LLM server for testing multi-field scenarios.

**TDD Mock Server Tests**:
```rust
#[tokio::test]
async fn test_mock_llm_multi_field_support() {
    let mock_server = MockLLMServer::new()
        .with_latency(50, 10) // 50ms ± 10ms
        .with_error_rate(0.01); // 1% errors
    
    let addr = mock_server.start().await;
    
    // Send multi-field request
    let client = Client::new();
    let request = LLMRequest {
        model: "gpt-4",
        messages: vec![
            Message::system("Analyze sentiment"),
            Message::user("Field: description\nText: Great product!"),
            Message::user("Field: review\nText: Could be better"),
        ],
    };
    
    let resp = client.post(&format!("http://{}/v1/completions", addr))
        .json(&request)
        .send()
        .await
        .unwrap();
    
    assert_eq!(resp.status(), 200);
    let result: LLMResponse = resp.json().await.unwrap();
    assert!(result.choices[0].text.contains("positive"));
}

#[test]
fn test_mock_response_templates() {
    let templates = ResponseTemplates::new();
    
    // Test sentiment templates
    let positive_text = "This is amazing!";
    let response = templates.sentiment_response(positive_text);
    assert!(response.contains("positive"));
    assert!(response.contains("confidence"));
    
    // Test churn prediction
    let churn_text = "I'm canceling my subscription";
    let response = templates.churn_response(churn_text);
    assert!(response.contains("high_risk"));
}

#[tokio::test]
async fn test_mock_server_rate_limiting() {
    let mock_server = MockLLMServer::new()
        .with_rate_limit(10); // 10 req/s
    
    let addr = mock_server.start().await;
    let client = Client::new();
    
    // Send 15 requests rapidly
    let mut handles = vec![];
    for _ in 0..15 {
        let client = client.clone();
        let addr = addr.clone();
        
        let handle = tokio::spawn(async move {
            client.post(&format!("http://{}/v1/completions", addr))
                .json(&dummy_request())
                .send()
                .await
        });
        handles.push(handle);
    }
    
    // Collect results
    let mut success = 0;
    let mut rate_limited = 0;
    
    for handle in handles {
        match handle.await.unwrap() {
            Ok(resp) if resp.status() == 200 => success += 1,
            Ok(resp) if resp.status() == 429 => rate_limited += 1,
            _ => panic!("Unexpected response"),
        }
    }
    
    assert_eq!(success, 10);
    assert_eq!(rate_limited, 5);
}
```

**Mock Server Features**:
1. Multi-field request handling
2. Configurable latency simulation
3. Error injection
4. Rate limiting
5. Response templates

**Deliverables**:
- Enhanced mock LLM server
- Response template system
- Latency simulation
- Load testing support

## Sprint 2 Tasks (Days 6-10)

### Task 4.4: Monitoring Infrastructure
**Story**: As an operator, I need comprehensive monitoring for the multi-field system.

**Monitoring Tests**:
```rust
#[tokio::test]
async fn test_prometheus_metrics_export() {
    let metrics = SystemMetrics::new();
    let exporter = PrometheusExporter::new(metrics.clone());
    
    // Simulate operations
    metrics.columns_profiled.inc_by(100);
    metrics.analysis_runs.inc();
    metrics.ml_inference_time.observe(0.5);
    
    // Export metrics
    let output = exporter.export();
    
    assert!(output.contains("datacloak_columns_profiled_total 100"));
    assert!(output.contains("datacloak_analysis_runs_total 1"));
    assert!(output.contains("datacloak_ml_inference_seconds"));
}

#[test]
fn test_structured_logging() {
    let (sink, logs) = test_sink();
    let logger = Logger::new(sink);
    
    logger.info("Starting analysis", json!({
        "run_id": "123",
        "columns": ["col1", "col2"],
        "estimated_time": 300,
    }));
    
    let log = logs.pop().unwrap();
    assert_eq!(log.level, Level::Info);
    assert_eq!(log.message, "Starting analysis");
    assert_eq!(log.fields["run_id"], "123");
    assert_eq!(log.fields["columns"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn test_distributed_tracing() {
    let tracer = init_tracer("datacloak");
    
    let root_span = tracer.start("analyze_request");
    
    // Child spans
    let profile_span = tracer.start_child("profile_columns", &root_span);
    thread::sleep(Duration::from_millis(50));
    profile_span.end();
    
    let ml_span = tracer.start_child("ml_inference", &root_span);
    thread::sleep(Duration::from_millis(100));
    ml_span.end();
    
    root_span.end();
    
    // Verify trace
    let trace = tracer.get_trace(root_span.trace_id());
    assert_eq!(trace.spans.len(), 3);
    assert!(trace.duration_ms() >= 150);
}
```

**Monitoring Components**:
```rust
pub struct SystemMetrics {
    // Profiling metrics
    pub columns_profiled: Counter,
    pub ml_inference_time: Histogram,
    pub graph_ranking_time: Histogram,
    pub profile_cache_hits: Counter,
    
    // Analysis metrics
    pub analysis_runs: Counter,
    pub columns_per_run: Histogram,
    pub rows_processed: Counter,
    pub streaming_latency: Histogram,
    
    // Resource metrics
    pub memory_usage_bytes: Gauge,
    pub cpu_usage_percent: Gauge,
    pub goroutines: Gauge,
    pub open_connections: Gauge,
    
    // Error metrics
    pub llm_errors: Counter,
    pub profiling_errors: Counter,
    pub stream_disconnects: Counter,
}

pub struct AlertRules {
    rules: Vec<Rule>,
}

impl AlertRules {
    pub fn default() -> Self {
        Self {
            rules: vec![
                Rule::new("High memory usage")
                    .when("memory_usage_bytes > 8GB")
                    .for_duration("5m")
                    .severity(Severity::Warning),
                
                Rule::new("LLM error rate")
                    .when("rate(llm_errors[5m]) > 0.1")
                    .severity(Severity::Critical),
                
                Rule::new("Slow profiling")
                    .when("histogram_quantile(0.99, profile_time) > 10s")
                    .severity(Severity::Warning),
            ],
        }
    }
}
```

**Deliverables**:
- Prometheus metrics
- Grafana dashboards
- Alert rules
- Tracing integration

### Task 4.5: End-to-End Testing Suite
**Story**: As a team, we need automated E2E tests covering all user workflows.

**E2E Test Suite**:
```rust
#[tokio::test]
async fn test_e2e_auto_discovery_workflow() {
    let env = TestEnvironment::new().await;
    
    // Start all services
    env.start_services(&[
        "datacloak-api",
        "mock-llm",
        "postgres",
        "redis",
    ]).await;
    
    // User uploads file via CLI
    let output = env.run_cli(&[
        "datacloak", "upload", 
        "--file", "testdata/customer_data.csv"
    ]).await;
    let file_id = extract_file_id(&output);
    
    // User profiles columns
    let output = env.run_cli(&[
        "datacloak", "profile",
        "--file-id", &file_id,
        "--output", "json"
    ]).await;
    let profile: ProfileOutput = serde_json::from_str(&output).unwrap();
    
    // Verify ML and graph scores
    let feedback_col = profile.find_column("customer_feedback").unwrap();
    assert!(feedback_col.ml_probability > 0.9);
    assert!(feedback_col.graph_score > 0.8);
    
    // User runs analysis with auto-discovery
    let output = env.run_cli(&[
        "datacloak", "analyze",
        "--file-id", &file_id,
        "--auto-discover",
        "--threshold", "0.75",
        "--output", "stream"
    ]).await;
    
    // Verify streaming output
    let mut line_count = 0;
    for line in output.lines() {
        if let Ok(result) = serde_json::from_str::<AnalysisResult>(line) {
            line_count += 1;
            assert!(!result.record_id.is_empty());
            assert!(["positive", "negative", "neutral"].contains(&result.sentiment.as_str()));
        }
    }
    assert!(line_count > 100);
    
    // Check metrics
    let metrics = env.get_metrics().await;
    assert!(metrics["datacloak_columns_analyzed"] >= 2.0);
}

#[tokio::test] 
async fn test_e2e_failure_recovery() {
    let env = TestEnvironment::new().await;
    env.start_services_all().await;
    
    // Start large analysis
    let run_id = env.start_analysis(
        "large_file.csv",
        vec!["col1", "col2", "col3"]
    ).await;
    
    // Wait for progress
    env.wait_for_progress(&run_id, 25).await;
    
    // Simulate API crash
    env.kill_service("datacloak-api").await;
    tokio::time::sleep(Duration::from_secs(2)).await;
    
    // Restart API
    env.start_service("datacloak-api").await;
    
    // Verify analysis resumes
    let status = env.get_analysis_status(&run_id).await;
    assert_eq!(status.state, "resuming");
    
    // Wait for completion
    env.wait_for_completion(&run_id).await;
    
    // Verify no data loss
    let results = env.get_results(&run_id).await;
    assert_eq!(results.total_rows, results.processed_rows);
}
```

**Test Environment**:
1. Docker-based service orchestration
2. Test data generation
3. Service health checks
4. Log aggregation
5. Metric collection

**Deliverables**:
- E2E test framework
- CI/CD integration
- Test environments
- Regression suite

### Task 4.6: Performance Testing Suite
**Story**: As a team, we need automated performance tests to prevent regressions.

**Performance Tests**:
```rust
#[bench]
fn bench_column_profiling_1000_columns(b: &mut Bencher) {
    let file = generate_csv_with_columns(1000, 10_000); // 1000 cols, 10k rows
    let profiler = ColumnProfiler::new();
    
    b.iter(|| {
        let candidates = profiler.profile_file(&file).unwrap();
        black_box(candidates);
    });
}

#[tokio::test]
async fn test_load_concurrent_analyses() {
    let env = TestEnvironment::new().await;
    env.start_services_all().await;
    
    // Upload test file
    let file_id = env.upload_file("testdata/medium_file.csv").await;
    
    // Run 50 concurrent analyses
    let start = Instant::now();
    let mut handles = vec![];
    
    for i in 0..50 {
        let env = env.clone();
        let file = file_id.clone();
        
        let handle = tokio::spawn(async move {
            let columns = vec![format!("col_{}", i % 10)];
            env.run_analysis(&file, columns).await
        });
        handles.push(handle);
    }
    
    // Wait for all to complete
    let mut successes = 0;
    for handle in handles {
        if handle.await.unwrap().is_ok() {
            successes += 1;
        }
    }
    
    let elapsed = start.elapsed();
    
    // Performance assertions
    assert_eq!(successes, 50); // All complete
    assert!(elapsed.as_secs() < 300); // Under 5 minutes
    
    // Check resource usage
    let metrics = env.get_metrics().await;
    assert!(metrics["max_memory_gb"] < 4.0);
    assert!(metrics["avg_cpu_percent"] < 80.0);
}

#[test]
fn test_memory_usage_20gb_file() {
    let large_file = "testdata/20gb_test.csv";
    let monitor = MemoryMonitor::new();
    
    let profiler = ColumnProfiler::new();
    let _result = profiler.profile_file(large_file).unwrap();
    
    // Should use <300MB for profiling
    assert!(monitor.peak_usage() < 300 * MB);
}
```

**Performance Baselines**:
```yaml
performance_baselines:
  column_profiling:
    1000_columns: 8.5s
    10000_columns: 95s
  
  ml_inference:
    single_column: 0.8ms
    batch_1000: 950ms
  
  graph_ranking:
    100_nodes: 15ms
    1000_nodes: 180ms
  
  streaming_throughput:
    single_column: 150MB/s
    10_columns: 120MB/s
  
  memory_usage:
    profiling_1000_cols: 250MB
    analysis_streaming: 800MB
```

**Deliverables**:
- Performance test suite
- Regression detection
- Load testing scenarios
- Resource monitoring

## Sprint 3 Tasks (Days 11-15)

### Task 4.7: Documentation and Examples
**Story**: As a user, I need comprehensive documentation for the new multi-field features.

**Documentation Tests**:
```rust
#[test]
fn test_readme_examples_compile() {
    let examples = extract_code_examples("README.md");
    
    for example in examples {
        let result = compile_example(&example);
        assert!(result.is_ok(), "Example failed: {}", example.name);
    }
}

#[test]
fn test_api_docs_completeness() {
    let api_spec = load_openapi_spec("openapi.yaml");
    
    // All endpoints documented
    let endpoints = extract_endpoints_from_code();
    for endpoint in endpoints {
        assert!(api_spec.has_path(&endpoint.path), 
                "Missing docs for {}", endpoint.path);
    }
    
    // All responses have examples
    for path in api_spec.paths() {
        for response in path.responses() {
            assert!(response.has_example(), 
                    "Missing example for {} {}", path.method, path.path);
        }
    }
}

#[test]
fn test_tutorial_workflows() {
    let tutorials = load_tutorials("docs/tutorials/");
    
    for tutorial in tutorials {
        let env = TestEnvironment::new();
        
        for step in tutorial.steps() {
            let result = env.execute_step(step);
            assert!(result.is_ok(), 
                    "Tutorial '{}' failed at step {}", tutorial.name, step.number);
        }
    }
}
```

**Documentation Structure**:
```markdown
# DataCloak Multi-Field Analysis

## Quick Start
1. Profile your CSV to discover text columns
2. Select columns for analysis (or use auto-discovery)
3. Get streaming sentiment results

## Examples

### Basic Usage
```bash
# Profile columns
datacloak profile --file customer_data.csv

# Analyze specific columns
datacloak analyze --file customer_data.csv \
  --columns feedback,comments,reviews

# Auto-discover and analyze
datacloak analyze --file customer_data.csv \
  --auto-discover --threshold 0.8
```

### Advanced Usage
```bash
# Dry run with cost estimate
datacloak analyze --file large_dataset.csv \
  --columns col1,col2,col3 --dry-run

# Stream results to another process
datacloak analyze --file data.csv \
  --columns text_field --output stream | \
  jq '.sentiment' | sort | uniq -c
```

## API Reference
[Generated from OpenAPI spec]

## Tutorials
- [Auto-Discovery Workflow](tutorials/auto-discovery.md)
- [Large File Processing](tutorials/large-files.md)
- [Custom Scoring](tutorials/custom-scoring.md)
```

**Deliverables**:
- User documentation
- API reference
- Code examples
- Video tutorials

### Task 4.8: Deployment and Operations
**Story**: As an operator, I need deployment artifacts and operational runbooks.

**Deployment Tests**:
```rust
#[test]
fn test_docker_build() {
    let output = Command::new("docker")
        .args(&["build", "-t", "datacloak:test", "."])
        .output()
        .unwrap();
    
    assert!(output.status.success());
}

#[test]
fn test_kubernetes_manifests() {
    let manifests = load_k8s_manifests("k8s/");
    
    for manifest in manifests {
        let output = Command::new("kubectl")
            .args(&["apply", "--dry-run=client", "-f", &manifest])
            .output()
            .unwrap();
        
        assert!(output.status.success());
    }
}

#[tokio::test]
async fn test_health_checks() {
    let container = start_container("datacloak:latest").await;
    
    // Wait for startup
    tokio::time::sleep(Duration::from_secs(5)).await;
    
    // Check health endpoint
    let client = Client::new();
    let resp = client.get("http://localhost:8080/health")
        .send()
        .await
        .unwrap();
    
    assert_eq!(resp.status(), 200);
    let health: HealthStatus = resp.json().await.unwrap();
    assert_eq!(health.status, "healthy");
    assert!(health.checks.database);
    assert!(health.checks.cache);
}
```

**Deployment Artifacts**:
```yaml
# docker-compose.yml
version: '3.8'
services:
  datacloak-api:
    image: datacloak:${VERSION}
    environment:
      - DATABASE_URL=postgres://...
      - REDIS_URL=redis://...
      - LLM_ENDPOINT=http://...
    ports:
      - "8080:8080"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=datacloak
      - POSTGRES_USER=datacloak
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 1gb --maxmemory-policy lru

volumes:
  postgres_data:
```

**Operational Runbooks**:
1. Deployment procedures
2. Monitoring setup
3. Troubleshooting guides
4. Performance tuning
5. Disaster recovery

**Deliverables**:
- Docker images
- Kubernetes manifests
- Helm charts
- Operational documentation

## Testing Requirements Summary
- CLI tests: 95% coverage
- Integration tests: Full scenario coverage
- Performance tests: No regressions
- E2E tests: All workflows
- Documentation tests: All examples work

## Monitoring Requirements
- Prometheus metrics for all components
- Distributed tracing with OpenTelemetry
- Structured logging with context
- Alerting rules for SLOs
- Grafana dashboards

## Integration Points
1. **With Dev 1**: ML model integration tests
2. **With Dev 2**: Graph algorithm performance tests
3. **With Dev 3**: API contract tests
4. **With Dev 4**: Full system integration

## TDD Best Practices for Integration
1. **Test the seams** - Focus on integration points
2. **Use test doubles** - Mock external dependencies
3. **Test failure scenarios** - Network, timeouts, errors
4. **Test at scale** - Use realistic data volumes
5. **Automate everything** - CI/CD must run all tests

## Definition of Done
- [ ] CLI fully supports multi-field analysis
- [ ] Integration tests cover all scenarios  
- [ ] Performance tests prevent regressions
- [ ] Monitoring captures all key metrics
- [ ] Documentation is comprehensive
- [ ] E2E tests validate user workflows
- [ ] Deployment is automated
- [ ] Operational runbooks complete