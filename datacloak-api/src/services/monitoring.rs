use prometheus::{
    Counter, Histogram, Gauge, Registry, Encoder, TextEncoder,
    HistogramOpts, Opts, CounterOpts, GaugeOpts
};
use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;
use uuid::Uuid;
use anyhow::Result;
use tracing::{info, warn, error, span, Level};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

// Service metrics structure
#[derive(Clone)]
pub struct ServiceMetrics {
    // API metrics
    pub api_requests_total: Counter,
    pub api_request_duration: Histogram,
    pub api_errors_total: Counter,
    
    // Performance metrics
    pub cache_hit_rate: Gauge,
    pub active_streams: Gauge,
    pub worker_utilization: Gauge,
    pub analysis_duration: Histogram,
    
    // Resource metrics
    pub memory_usage_bytes: Gauge,
    pub active_connections: Gauge,
    pub queue_depth: Gauge,
    
    // Business metrics
    pub files_analyzed_total: Counter,
    pub columns_processed_total: Counter,
    pub successful_analyses: Counter,
    pub failed_analyses: Counter,
    
    registry: Registry,
}

impl ServiceMetrics {
    pub fn new() -> Result<Self> {
        let registry = Registry::new();
        
        // API metrics
        let api_requests_total = Counter::with_opts(
            CounterOpts::new("datacloak_api_requests_total", "Total number of API requests")
                .const_labels(vec![("service".to_string(), "datacloak-api".to_string())].into_iter().collect())
        )?;
        registry.register(Box::new(api_requests_total.clone()))?;
        
        let api_request_duration = Histogram::with_opts(
            HistogramOpts::new(
                "datacloak_api_request_duration_seconds",
                "Duration of API requests in seconds"
            ).buckets(vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
        )?;
        registry.register(Box::new(api_request_duration.clone()))?;
        
        let api_errors_total = Counter::with_opts(
            CounterOpts::new("datacloak_api_errors_total", "Total number of API errors")
        )?;
        registry.register(Box::new(api_errors_total.clone()))?;
        
        // Performance metrics
        let cache_hit_rate = Gauge::with_opts(
            GaugeOpts::new("datacloak_cache_hit_rate", "Current cache hit rate (0-1)")
        )?;
        registry.register(Box::new(cache_hit_rate.clone()))?;
        
        let active_streams = Gauge::with_opts(
            GaugeOpts::new("datacloak_active_streams", "Number of active SSE streams")
        )?;
        registry.register(Box::new(active_streams.clone()))?;
        
        let worker_utilization = Gauge::with_opts(
            GaugeOpts::new("datacloak_worker_utilization", "Worker utilization percentage (0-100)")
        )?;
        registry.register(Box::new(worker_utilization.clone()))?;
        
        let analysis_duration = Histogram::with_opts(
            HistogramOpts::new(
                "datacloak_analysis_duration_seconds",
                "Duration of analysis operations in seconds"
            ).buckets(vec![1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0])
        )?;
        registry.register(Box::new(analysis_duration.clone()))?;
        
        // Resource metrics
        let memory_usage_bytes = Gauge::with_opts(
            GaugeOpts::new("datacloak_memory_usage_bytes", "Current memory usage in bytes")
        )?;
        registry.register(Box::new(memory_usage_bytes.clone()))?;
        
        let active_connections = Gauge::with_opts(
            GaugeOpts::new("datacloak_active_connections", "Number of active database connections")
        )?;
        registry.register(Box::new(active_connections.clone()))?;
        
        let queue_depth = Gauge::with_opts(
            GaugeOpts::new("datacloak_queue_depth", "Current analysis queue depth")
        )?;
        registry.register(Box::new(queue_depth.clone()))?;
        
        // Business metrics
        let files_analyzed_total = Counter::with_opts(
            CounterOpts::new("datacloak_files_analyzed_total", "Total number of files analyzed")
        )?;
        registry.register(Box::new(files_analyzed_total.clone()))?;
        
        let columns_processed_total = Counter::with_opts(
            CounterOpts::new("datacloak_columns_processed_total", "Total number of columns processed")
        )?;
        registry.register(Box::new(columns_processed_total.clone()))?;
        
        let successful_analyses = Counter::with_opts(
            CounterOpts::new("datacloak_successful_analyses_total", "Total number of successful analyses")
        )?;
        registry.register(Box::new(successful_analyses.clone()))?;
        
        let failed_analyses = Counter::with_opts(
            CounterOpts::new("datacloak_failed_analyses_total", "Total number of failed analyses")
        )?;
        registry.register(Box::new(failed_analyses.clone()))?;
        
        Ok(Self {
            api_requests_total,
            api_request_duration,
            api_errors_total,
            cache_hit_rate,
            active_streams,
            worker_utilization,
            analysis_duration,
            memory_usage_bytes,
            active_connections,
            queue_depth,
            files_analyzed_total,
            columns_processed_total,
            successful_analyses,
            failed_analyses,
            registry,
        })
    }
    
    pub fn render_metrics(&self) -> Result<String> {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer)?)
    }
    
    pub fn record_api_request(&self, endpoint: &str, method: &str, duration_seconds: f64, status_code: u16) {
        self.api_requests_total.inc();
        self.api_request_duration.observe(duration_seconds);
        
        if status_code >= 400 {
            self.api_errors_total.inc();
        }
        
        info!(
            endpoint = endpoint,
            method = method,
            duration_ms = duration_seconds * 1000.0,
            status_code = status_code,
            "API request completed"
        );
    }
    
    pub fn update_cache_metrics(&self, hit_rate: f64) {
        self.cache_hit_rate.set(hit_rate);
    }
    
    pub fn update_resource_metrics(&self, memory_bytes: u64, active_conns: usize, queue_depth: usize) {
        self.memory_usage_bytes.set(memory_bytes as f64);
        self.active_connections.set(active_conns as f64);
        self.queue_depth.set(queue_depth as f64);
    }
    
    pub fn record_analysis_complete(&self, file_id: Uuid, columns: &[String], duration_seconds: f64, success: bool) {
        self.files_analyzed_total.inc();
        self.columns_processed_total.inc_by(columns.len() as f64);
        self.analysis_duration.observe(duration_seconds);
        
        if success {
            self.successful_analyses.inc();
            info!(
                file_id = %file_id,
                columns = columns.len(),
                duration_seconds = duration_seconds,
                "Analysis completed successfully"
            );
        } else {
            self.failed_analyses.inc();
            warn!(
                file_id = %file_id,
                columns = columns.len(),
                duration_seconds = duration_seconds,
                "Analysis failed"
            );
        }
    }
    
    pub fn update_worker_metrics(&self, utilization_percent: f64) {
        self.worker_utilization.set(utilization_percent);
    }
    
    pub fn update_stream_count(&self, count: usize) {
        self.active_streams.set(count as f64);
    }
}

// Distributed tracing support
pub struct TracingService {
    active_traces: Arc<RwLock<HashMap<Uuid, TraceContext>>>,
}

#[derive(Debug, Clone)]
pub struct TraceContext {
    pub trace_id: Uuid,
    pub parent_span_id: Option<Uuid>,
    pub operation: String,
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub metadata: HashMap<String, String>,
}

impl TracingService {
    pub fn new() -> Self {
        Self {
            active_traces: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn start_trace(&self, operation: &str, metadata: HashMap<String, String>) -> Uuid {
        let trace_id = Uuid::new_v4();
        let context = TraceContext {
            trace_id,
            parent_span_id: None,
            operation: operation.to_string(),
            started_at: chrono::Utc::now(),
            metadata,
        };
        
        let mut traces = self.active_traces.write().await;
        traces.insert(trace_id, context);
        
        let span = span!(Level::INFO, "operation", trace_id = %trace_id, operation = operation);
        let _enter = span.enter();
        info!("Trace started");
        
        trace_id
    }
    
    pub async fn start_span(&self, parent_trace_id: Uuid, operation: &str, metadata: HashMap<String, String>) -> Option<Uuid> {
        let span_id = Uuid::new_v4();
        
        // Check if parent trace exists
        {
            let traces = self.active_traces.read().await;
            if !traces.contains_key(&parent_trace_id) {
                warn!(parent_trace_id = %parent_trace_id, "Parent trace not found");
                return None;
            }
        }
        
        let context = TraceContext {
            trace_id: parent_trace_id,
            parent_span_id: Some(span_id),
            operation: operation.to_string(),
            started_at: chrono::Utc::now(),
            metadata,
        };
        
        let mut traces = self.active_traces.write().await;
        traces.insert(span_id, context);
        
        let span = span!(
            Level::INFO, 
            "span", 
            trace_id = %parent_trace_id,
            span_id = %span_id,
            operation = operation
        );
        let _enter = span.enter();
        info!("Span started");
        
        Some(span_id)
    }
    
    pub async fn finish_trace(&self, trace_id: Uuid, success: bool, error_message: Option<String>) {
        let mut traces = self.active_traces.write().await;
        if let Some(context) = traces.remove(&trace_id) {
            let duration = chrono::Utc::now().signed_duration_since(context.started_at);
            
            let span = span!(
                Level::INFO,
                "trace_complete",
                trace_id = %trace_id,
                operation = %context.operation,
                duration_ms = duration.num_milliseconds(),
                success = success
            );
            let _enter = span.enter();
            
            if success {
                info!("Trace completed successfully");
            } else {
                error!(error = ?error_message, "Trace completed with error");
            }
        }
    }
    
    pub async fn add_trace_metadata(&self, trace_id: Uuid, key: String, value: String) {
        let mut traces = self.active_traces.write().await;
        if let Some(context) = traces.get_mut(&trace_id) {
            context.metadata.insert(key, value);
        }
    }
    
    pub async fn get_active_trace_count(&self) -> usize {
        let traces = self.active_traces.read().await;
        traces.len()
    }
}

// Health check endpoints
#[derive(Debug, Clone, serde::Serialize)]
pub struct HealthCheckResponse {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub checks: HashMap<String, ComponentHealth>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ComponentHealth {
    pub status: String,
    pub message: Option<String>,
    pub response_time_ms: Option<u64>,
}

pub struct HealthService {
    start_time: chrono::DateTime<chrono::Utc>,
    version: String,
}

impl HealthService {
    pub fn new(version: String) -> Self {
        Self {
            start_time: chrono::Utc::now(),
            version,
        }
    }
    
    pub async fn check_health(&self) -> HealthCheckResponse {
        let uptime = chrono::Utc::now().signed_duration_since(self.start_time);
        let mut checks = HashMap::new();
        
        // Database health check
        let db_health = self.check_database().await;
        checks.insert("database".to_string(), db_health);
        
        // Cache health check
        let cache_health = self.check_cache().await;
        checks.insert("cache".to_string(), cache_health);
        
        // Worker health check
        let worker_health = self.check_workers().await;
        checks.insert("workers".to_string(), worker_health);
        
        // Overall status
        let overall_status = if checks.values().all(|h| h.status == "healthy") {
            "healthy".to_string()
        } else {
            "degraded".to_string()
        };
        
        HealthCheckResponse {
            status: overall_status,
            version: self.version.clone(),
            uptime_seconds: uptime.num_seconds() as u64,
            checks,
        }
    }
    
    async fn check_database(&self) -> ComponentHealth {
        // In real implementation, would check database connectivity
        // For now, simulate health check
        ComponentHealth {
            status: "healthy".to_string(),
            message: Some("Database connection pool healthy".to_string()),
            response_time_ms: Some(5),
        }
    }
    
    async fn check_cache(&self) -> ComponentHealth {
        // In real implementation, would check cache connectivity
        ComponentHealth {
            status: "healthy".to_string(),
            message: Some("Cache layer responding".to_string()),
            response_time_ms: Some(2),
        }
    }
    
    async fn check_workers(&self) -> ComponentHealth {
        // In real implementation, would check worker coordinator
        ComponentHealth {
            status: "healthy".to_string(),
            message: Some("4/4 workers healthy".to_string()),
            response_time_ms: Some(1),
        }
    }
}

// SLO monitoring
pub struct SLODashboard {
    pub error_rate_threshold: f64,  // e.g., 0.01 for 1%
    pub latency_p95_threshold_ms: f64, // e.g., 500.0 for 500ms
    pub availability_threshold: f64, // e.g., 0.999 for 99.9%
}

impl SLODashboard {
    pub fn new() -> Self {
        Self {
            error_rate_threshold: 0.01, // 1% error rate
            latency_p95_threshold_ms: 500.0, // 500ms P95 latency
            availability_threshold: 0.999, // 99.9% availability
        }
    }
    
    pub fn check_slo_compliance(&self, _metrics: &ServiceMetrics) -> SLOStatus {
        // In real implementation, would calculate from actual metrics
        // For now, return simulated status
        SLOStatus {
            error_rate_compliant: true,
            latency_compliant: true,
            availability_compliant: true,
            overall_score: 1.0,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SLOStatus {
    pub error_rate_compliant: bool,
    pub latency_compliant: bool,
    pub availability_compliant: bool,
    pub overall_score: f64, // 0.0 to 1.0
}

// Initialize tracing for the application
pub fn init_tracing() -> Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "datacloak_api=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer().json())
        .init();
    
    Ok(())
}

// Monitoring service that ties everything together
pub struct MonitoringService {
    pub metrics: ServiceMetrics,
    pub tracing: TracingService,
    pub health: HealthService,
    pub slo_dashboard: SLODashboard,
}

impl MonitoringService {
    pub fn new(version: String) -> Result<Self> {
        Ok(Self {
            metrics: ServiceMetrics::new()?,
            tracing: TracingService::new(),
            health: HealthService::new(version),
            slo_dashboard: SLODashboard::new(),
        })
    }
    
    pub async fn start_background_tasks(&self) {
        // Start periodic metric collection
        let metrics = self.metrics.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(30));
            loop {
                interval.tick().await;
                
                // Collect system metrics
                if let Ok(memory) = sys_info::mem_info() {
                    metrics.memory_usage_bytes.set(memory.total as f64 * 1024.0);
                }
            }
        });
        
        info!("Monitoring background tasks started");
    }
}

// Additional dependencies needed in Cargo.toml:
// sys-info = "0.9"