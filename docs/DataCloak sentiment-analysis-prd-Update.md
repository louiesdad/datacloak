# Product Requirements Document: Multi-Field Sentiment Analysis with Auto-Discovery for DataCloak

## Executive Summary

This PRD addresses DataCloak's current single-field limitation by introducing multi-field sentiment analysis with automatic candidate field discovery, time-to-finish estimation, and maintained performance for files ≥20GB. The solution preserves DataCloak's core strengths in sensitive data protection (PII, HIPAA, Financial) while enabling parallel processing of multiple text fields per record.

## Problem Statement

DataCloak currently processes only one field at a time due to architectural constraints:
- CLI/UI hardwired to single `--field` parameter
- Service layer accepts only one column per analysis run
- Worker threads bind to exactly one column
- Log schema structured for single field tracking

This limitation forces users to:
- Manually identify text-heavy fields
- Run multiple sequential analyses for multi-field records
- Lose contextual relationships between fields
- Experience longer overall processing times

## Solution Overview

### Core Innovation: Column Profiling First, Inference Second

1. **Automatic Field Discovery**: Profile CSV/Excel files to identify text-heavy columns suitable for sentiment analysis
2. **Parallel Multi-Field Processing**: Analyze multiple fields concurrently while maintaining field boundaries
3. **Time-to-Finish Estimation**: Provide accurate ETAs based on sample analysis
4. **Out-of-Core Processing**: Handle 20-50GB files on standard hardware using streaming

### Architecture Transformation

```
Current: [File] → [Single Field] → [Analysis] → [Result]

New:     [File] → [Column Profiler] → [Field Selector] → [Parallel Analyzer] → [Multi-Field Results]
                         ↓                    ↓                    ↓
                  [Auto-Discovery]      [ETA Estimator]    [Progress Monitor]
```

## Detailed Design

### 1. Advanced Column Profiling Engine

**Multi-Stage Detection Pipeline:**
```rust
pub struct ColumnProfiler {
    sample_size: usize, // 100k rows max
    ml_classifier: MLClassifier,
    graph_ranker: GraphRanker,
    heuristic_scorer: HeuristicScorer,
}

pub struct ColumnCandidate {
    name: String,
    ml_prob: f32, // ML classification probability
    graph_score: f32, // Graph-based relevance score
    final_score: f32, // Combined score
    stats: ColumnStats,
    predicted_type: ColumnType,
}

#[derive(Clone, Debug)]
pub enum ColumnType {
    TextLong,
    TextShort,
    Numeric,
    DateTime,
    IdUuid,
    Email,
    List,
    Other,
}

impl ColumnProfiler {
    pub async fn profile_file(&self, file_path: &Path) -> Vec<ColumnCandidate> {
        // Stage 1: Sample extraction via DuckDB
        let sample = self.extract_sample_stream(file_path).await?;
        
        // Stage 2: ML Classification
        let ml_predictions = self.ml_classifier.predict_batch(&sample).await?;
        
        // Stage 3: Graph-based ranking
        let text_columns = ml_predictions.iter()
            .filter(|p| p.is_text_probable())
            .collect::<Vec<_>>();
        
        let graph_scores = self.graph_ranker.rank_columns(&text_columns).await?;
        
        // Stage 4: Merge and rank
        self.merge_and_rank(ml_predictions, graph_scores)
    }
}
```

**ML Classifier Implementation:**
```rust
pub struct MLClassifier {
    model: OnnxModel,
    feature_extractor: FeatureExtractor,
}

impl MLClassifier {
    pub async fn predict_batch(&self, sample: &Sample) -> Vec<MLPrediction> {
        let mut predictions = Vec::new();
        
        for column in sample.columns() {
            let features = self.feature_extractor.extract(column);
            let prediction = self.model.predict(&features)?;
            
            predictions.push(MLPrediction {
                column_name: column.name.clone(),
                probabilities: prediction.class_probabilities,
                predicted_type: prediction.argmax_class(),
            });
        }
        
        predictions
    }
}

pub struct FeatureExtractor {
    fasttext_model: FastTextEmbedding,
}

impl FeatureExtractor {
    pub fn extract(&self, column: &Column) -> Features {
        Features {
            // Header embedding (300d)
            header_embedding: self.fasttext_model.encode(&column.name),
            
            // Shape statistics
            mean_char_length: column.mean_length(),
            std_char_length: column.std_length(),
            mean_token_count: column.mean_tokens(),
            
            // Character class ratios
            digit_ratio: column.digit_ratio(),
            punct_ratio: column.punctuation_ratio(),
            alpha_ratio: column.alphabetic_ratio(),
            
            // Information theory metrics
            entropy: column.shannon_entropy(),
            simpson_index: column.simpson_diversity(),
            
            // N-gram features (256d hashed)
            ngram_tfidf: column.hashed_ngrams(3, 5),
        }
    }
}
```

**Graph-Based Ranking:**
```rust
pub struct GraphRanker {
    similarity_threshold: f32,
    k_neighbors: usize,
}

impl GraphRanker {
    pub async fn rank_columns(&self, columns: &[&MLPrediction]) -> HashMap<String, f32> {
        // Build similarity graph
        let graph = self.build_similarity_graph(columns).await?;
        
        // Calculate centrality scores
        let pagerank_scores = self.calculate_pagerank(&graph);
        let laplacian_scores = self.calculate_laplacian_scores(&graph);
        
        // Community detection for deduplication
        let communities = self.detect_communities(&graph);
        
        // Combine scores
        let mut scores = HashMap::new();
        for (node_id, column) in graph.nodes() {
            let graph_score = 0.7 * pagerank_scores[node_id] + 
                             0.3 * (1.0 / laplacian_scores[node_id]);
            
            scores.insert(column.name.clone(), graph_score);
        }
        
        scores
    }
    
    async fn build_similarity_graph(&self, columns: &[&MLPrediction]) -> Graph {
        let mut graph = Graph::new();
        
        // Add nodes
        for column in columns {
            let node_vector = self.create_node_vector(column);
            graph.add_node(column, node_vector);
        }
        
        // Add edges using k-NN
        let index = self.build_faiss_index(&graph.node_vectors());
        
        for (i, node) in graph.nodes().enumerate() {
            let neighbors = index.search(&node.vector, self.k_neighbors)?;
            
            for (j, distance) in neighbors {
                if i != j {
                    let weight = self.calculate_edge_weight(
                        &graph.nodes[i],
                        &graph.nodes[j]
                    );
                    graph.add_edge(i, j, weight);
                }
            }
        }
        
        graph
    }
    
    fn calculate_edge_weight(&self, node_i: &Node, node_j: &Node) -> f32 {
        let header_similarity = cosine_similarity(
            &node_i.header_embedding,
            &node_j.header_embedding
        );
        
        let token_similarity = jaccard_similarity(
            &node_i.tokens,
            &node_j.tokens
        );
        
        0.6 * header_similarity + 0.4 * token_similarity
    }
}
```

### 2. Multi-Field Analysis Pipeline

**Parallel Processing Architecture:**
```rust
pub struct MultiFieldAnalyzer {
    sensitive_detector: SensitiveDataDetector,
    sentiment_analyzer: SentimentAnalyzer,
    max_parallel_fields: usize,
}

impl MultiFieldAnalyzer {
    pub async fn analyze_multi_field(
        &self,
        file: DataSource,
        selected_columns: Vec<String>,
        options: AnalysisOptions,
    ) -> Stream<MultiFieldResult> {
        // Create parallel processing pipeline
        let (tx, rx) = mpsc::channel(1000);
        
        // Spawn workers per column (up to CPU core count)
        let workers = min(selected_columns.len(), num_cpus::get());
        let columns_per_worker = selected_columns.chunks(workers);
        
        for (worker_id, columns) in columns_per_worker.enumerate() {
            let analyzer = self.clone();
            let tx = tx.clone();
            
            tokio::spawn(async move {
                analyzer.process_columns(file, columns, tx, worker_id).await
            });
        }
        
        // Return result stream
        ReceiverStream::new(rx)
    }
    
    async fn process_columns(
        &self,
        file: DataSource,
        columns: &[String],
        tx: Sender<MultiFieldResult>,
        worker_id: usize,
    ) {
        // Stream processing with checkpointing
        let mut checkpoint = Checkpoint::load_or_new(worker_id);
        
        let query = format!(
            "SELECT record_id, {} FROM read_csv_auto('{}') 
             OFFSET {} LIMIT {}",
            columns.join(", "),
            file.path,
            checkpoint.last_offset,
            BATCH_SIZE
        );
        
        let conn = DuckDBConnection::new();
        let mut stream = conn.execute_stream(&query);
        
        while let Some(batch) = stream.next().await {
            // Process each column in the batch
            for column in columns {
                let results = self.analyze_column(&batch, column).await;
                
                for result in results {
                    tx.send(result).await?;
                }
            }
            
            // Update checkpoint
            checkpoint.last_offset += batch.len();
            checkpoint.save().await?;
        }
    }
}
```

### 3. Time-to-Finish Estimator

**ETA Calculation Engine:**
```rust
pub struct ETAEstimator {
    sample_rows: usize, // 50,000
    history: Vec<SampleMetric>,
}

pub struct SampleMetric {
    rows: usize,
    columns: usize,
    elapsed_ms: u64,
    tokens_used: usize,
}

impl ETAEstimator {
    pub async fn estimate(
        &mut self,
        file: &DataSource,
        columns: &[String],
        chain_type: ChainType,
    ) -> ETAResult {
        // Run sample analysis
        let sample_start = Instant::now();
        let sample_results = self.analyze_sample(file, columns, chain_type).await?;
        let sample_elapsed = sample_start.elapsed();
        
        // Record metric
        let metric = SampleMetric {
            rows: self.sample_rows,
            columns: columns.len(),
            elapsed_ms: sample_elapsed.as_millis() as u64,
            tokens_used: sample_results.total_tokens,
        };
        
        self.history.push(metric);
        
        // Calculate estimates using linear regression
        let total_rows = file.estimated_rows()?;
        let rows_per_second = self.sample_rows as f64 / sample_elapsed.as_secs_f64();
        let estimated_seconds = (total_rows as f64 / rows_per_second) as u64;
        
        // Add confidence interval
        let confidence_interval = self.calculate_confidence_interval();
        
        ETAResult {
            estimated_seconds,
            confidence_lower: estimated_seconds as f64 * (1.0 - confidence_interval),
            confidence_upper: estimated_seconds as f64 * (1.0 + confidence_interval),
            estimated_cost: self.calculate_cost(total_rows, columns.len(), chain_type),
            sample_metrics: metric,
        }
    }
}
```

### 4. Service API Updates

**New Endpoints:**
```rust
// Advanced ML+Graph profiling endpoint
#[post("/api/v2/profile")]
async fn profile_columns(
    file_id: web::Json<FileId>,
    profiler: web::Data<ColumnProfiler>,
) -> Result<HttpResponse> {
    let candidates = profiler.profile_file(&file_id.path).await?;
    Ok(HttpResponse::Ok().json(ProfileResponse {
        candidates: candidates.into_iter().map(|c| CandidateJson {
            column: c.name,
            ml_prob: c.ml_prob,
            graph_score: c.graph_score,
            final_score: c.final_score,
            null_pct: c.stats.null_ratio,
            mean_len: c.stats.mean_length,
            predicted_type: c.predicted_type.to_string(),
        }).collect(),
        total_columns: candidates.len(),
        recommended: candidates.iter().filter(|c| c.final_score > 0.7).count(),
    }))
}

// ML-only profiling (fallback/debugging)
#[post("/api/v2/profile_ml")]
async fn profile_ml_only(
    file_id: web::Json<FileId>,
    classifier: web::Data<MLClassifier>,
) -> Result<HttpResponse> {
    let sample = extract_sample(&file_id.path).await?;
    let predictions = classifier.predict_batch(&sample).await?;
    Ok(HttpResponse::Ok().json(predictions))
}

// Graph ranking with custom threshold
#[post("/api/v2/profile_graph")]
async fn profile_graph(
    req: web::Json<GraphProfileRequest>,
    ranker: web::Data<GraphRanker>,
) -> Result<HttpResponse> {
    let ml_filtered = req.ml_predictions.iter()
        .filter(|p| p.text_probability() >= req.min_prob.unwrap_or(0.25))
        .collect();
    let scores = ranker.rank_columns(&ml_filtered).await?;
    Ok(HttpResponse::Ok().json(scores))
}

// ETA estimation endpoint
#[post("/api/v2/estimate")]
async fn estimate_runtime(
    req: web::Json<EstimateRequest>,
    estimator: web::Data<ETAEstimator>,
) -> Result<HttpResponse> {
    let eta = estimator.estimate(
        &req.file_id,
        &req.selected_columns,
        req.chain_type,
    ).await?;
    
    Ok(HttpResponse::Ok().json(eta))
}

// Multi-field analysis endpoint
#[post("/api/v2/analyze")]
async fn analyze_multi_field(
    req: web::Json<AnalyzeRequest>,
    analyzer: web::Data<MultiFieldAnalyzer>,
) -> Result<HttpResponse> {
    let stream = analyzer.analyze_multi_field(
        req.file_id,
        req.selected_columns,
        req.options,
    ).await?;
    
    // Stream results back to client
    Ok(HttpResponse::Ok()
        .streaming(stream.map(|r| Bytes::from(serde_json::to_vec(&r)?))))
}
```

### 5. Data Model Evolution

**Updated Schema:**
```sql
-- Analysis runs now track multiple columns
CREATE TABLE analysis_runs (
    run_id UUID PRIMARY KEY,
    file_id UUID NOT NULL,
    selected_columns JSON NOT NULL, -- Array of column names
    chain_type VARCHAR(50),
    started_at TIMESTAMP,
    estimated_seconds INTEGER,
    actual_seconds INTEGER,
    status VARCHAR(20)
);

-- Logs support multi-column results
CREATE TABLE analysis_logs (
    log_id UUID PRIMARY KEY,
    run_id UUID REFERENCES analysis_runs(run_id),
    record_id VARCHAR(255),
    column_name VARCHAR(100),
    step VARCHAR(50),
    result JSON,
    latency_ms INTEGER,
    sensitive_entities JSON, -- Detected PII/HIPAA/Financial
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Checkpointing for resilience
CREATE TABLE analysis_checkpoints (
    worker_id INTEGER,
    run_id UUID,
    last_offset BIGINT,
    last_record_id VARCHAR(255),
    updated_at TIMESTAMP,
    PRIMARY KEY (worker_id, run_id)
);
```

### 6. Performance Optimizations

**Memory-Efficient Streaming:**
```rust
pub struct StreamingProcessor {
    chunk_size: usize, // 100MB chunks
    max_memory: usize, // 1GB resident
}

impl StreamingProcessor {
    pub async fn process_file(&self, file: &Path) -> Result<()> {
        // Use Arrow/Parquet for efficient columnar processing
        let reader = ParquetReader::new(file)?;
        
        // Process in chunks
        while let Some(chunk) = reader.next_chunk(self.chunk_size).await? {
            // Zero-copy processing where possible
            self.process_chunk(chunk).await?;
            
            // Force memory cleanup between chunks
            self.cleanup_memory();
        }
        
        Ok(())
    }
}
```

**Parallel Column Processing:**
```rust
// CPU-bound parallelism
let thread_pool = rayon::ThreadPoolBuilder::new()
    .num_threads(num_cpus::get())
    .build()?;

thread_pool.install(|| {
    columns.par_iter()
        .map(|column| process_column(column))
        .collect()
});
```

## User Experience Flow

### 1. Upload & Auto-Discovery
```
User uploads file → System profiles columns → Shows ranked candidates with scores
```

### 2. Column Selection & Estimation
```
User selects columns → System runs sample → Shows ETA, cost, and confidence interval
```

### 3. Analysis Execution
```
User starts job → Progress bar with live updates → Pause/Resume capability
```

### 4. Results & Export
```
Streaming results → Download as CSV/Parquet → Per-field sentiment scores preserved
```

## Success Metrics

### Performance Targets
- **Profiling Speed**: <10 seconds for 1000 columns (including ML+Graph)
- **ML Inference**: <1 second for 1000 columns
- **Graph Ranking**: <4 seconds for filtered columns
- **Throughput**: 1M rows/minute on 8-core machine
- **Memory Usage**: ≤300MB for profiling, ≤1GB resident per 100MB processed
- **ETA Accuracy**: ±15% of actual runtime

### Quality Targets
- **Discovery Precision**: ≥90% of manually chosen columns in top-N suggestions
- **ML Classification**: micro-F1 ≥0.93, macro-recall ≥0.88
- **Sensitive Data Protection**: 100% detection rate
- **Checkpoint Recovery**: <1 minute to resume from failure

## Implementation Timeline

### Sprint 1 (Days 1-5)
- ML classifier integration with ONNX runtime
- Feature extraction pipeline (fastText, n-grams, statistics)
- Basic graph construction with similarity metrics
- Multi-select UI components

### Sprint 2 (Days 6-10)
- Graph ranking algorithms (PageRank, Laplacian scoring)
- Community detection for column deduplication
- Parallel processing pipeline
- ETA estimation with ML model inference costs

### Sprint 3 (Days 11-15)
- Performance optimization for 1000+ column files
- Integration testing with 50GB files
- A/B testing framework for ML vs heuristic comparison
- Documentation and deployment

## Testing & Validation

### ML Model Validation
- **Training Data**: VizNet 700k columns + SOCRATA public data + 20 anonymized client snapshots
- **Test Set**: Hold-out 20% for model evaluation
- **Regression Testing**: Nightly runs on production samples to detect drift

### Integration Testing
```rust
#[test]
async fn test_ml_profiling_accuracy() {
    let profiler = ColumnProfiler::new();
    let synthetic_csv = generate_test_file(1000, 5_000_000); // 1000 cols, 5M rows
    
    let candidates = profiler.profile_file(&synthetic_csv).await.unwrap();
    let text_columns = candidates.iter()
        .filter(|c| c.final_score > 0.7)
        .count();
    
    assert!(text_columns >= 90); // ≥90% precision on known text columns
}

#[test]
async fn test_profiling_performance() {
    let start = Instant::now();
    let result = profiler.profile_file(&large_file).await;
    
    assert!(start.elapsed().as_secs() < 10); // <10s for 1000 columns
}
```

## Technical Dependencies

### Required Libraries
```toml
[dependencies]
duckdb = "0.9"
arrow = "48.0"
tokio = { version = "1.35", features = ["full"] }
rayon = "1.8"
serde_json = "1.0"
uuid = "1.6"
```

### Infrastructure Requirements
- Rust 1.75+
- 8+ CPU cores recommended
- SSD storage for temp files
- 8GB+ RAM for optimal performance

## Risk Mitigation

### Technical Risks
1. **Memory pressure on large files**: Strict chunk-based processing with forced GC
2. **Column profiling accuracy**: Adjustable heuristics with user override
3. **ETA drift on complex content**: Adaptive sampling with history learning

### Operational Risks
1. **Disk space for checkpoints**: Configurable retention with auto-cleanup
2. **API rate limits**: Built-in throttling with backpressure
3. **Network interruptions**: Resumable uploads with chunked transfer

## Monitoring & Observability

### Key Metrics
```rust
pub struct AnalysisMetrics {
    // Performance
    pub columns_per_second: Gauge,
    pub memory_usage_mb: Gauge,
    pub checkpoint_saves: Counter,
    
    // Quality
    pub profiling_accuracy: Histogram,
    pub eta_accuracy: Histogram,
    pub sensitive_data_found: Counter,
    
    // Business
    pub columns_analyzed: Counter,
    pub files_processed: Counter,
    pub total_gb_processed: Counter,
}
```

## Future Enhancements

### Next Quarter
- Custom scoring weights per industry vertical
- Real-time sentiment streaming
- Integration with more LLM providers
- Advanced report generation with insights

### Next Year
- Multi-language support (Spanish, French, German)
- Fine-tuned domain-specific models
- Integration with BI tools
- Custom model training pipeline