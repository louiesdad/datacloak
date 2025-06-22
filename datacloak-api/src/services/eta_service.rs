use crate::models::{ChainType, ETAResponse};
use crate::repositories::ETAHistoryRepository;
use uuid::Uuid;
use anyhow::Result;
use std::collections::HashMap;
use tokio::sync::RwLock;

#[derive(Debug, Clone)]
pub struct SampleMetric {
    pub rows: usize,
    pub columns: usize,
    pub elapsed_ms: u64,
    pub tokens_used: usize,
    pub chain_type: ChainType,
}

#[derive(Debug, Clone)]
pub struct ETAEstimate {
    pub estimated_seconds: u64,
    pub confidence_lower: u64,
    pub confidence_upper: u64,
    pub estimated_cost: f64,
    pub total_rows: usize,
    pub total_tokens_estimate: usize,
}

pub struct ETAEstimator {
    // Historical data for regression analysis
    sample_metrics: RwLock<Vec<SampleMetric>>,
    // Chain type specific multipliers
    chain_multipliers: HashMap<String, f64>,
    // Pricing configuration
    token_cost_per_1k: f64,
    repository: Option<ETAHistoryRepository>,
}

impl ETAEstimator {
    pub fn new() -> Self {
        let mut chain_multipliers = HashMap::new();
        chain_multipliers.insert("sentiment".to_string(), 1.0);
        chain_multipliers.insert("entity".to_string(), 1.5);  // Entity extraction is slower
        chain_multipliers.insert("classification".to_string(), 0.8); // Classification is faster
        
        Self {
            sample_metrics: RwLock::new(Vec::new()),
            chain_multipliers,
            token_cost_per_1k: 0.10, // $0.10 per 1000 tokens
            repository: None,
        }
    }
    
    pub fn with_repository(mut self, repository: ETAHistoryRepository) -> Self {
        self.repository = Some(repository);
        self
    }
    
    pub async fn add_sample_metric(&self, metric: SampleMetric) {
        let mut metrics = self.sample_metrics.write().await;
        metrics.push(metric);
        
        // Keep only last 1000 samples to prevent unbounded growth
        if metrics.len() > 1000 {
            metrics.remove(0);
        }
    }
    
    pub async fn estimate_analysis(
        &self,
        file_id: Uuid,
        columns: &[String],
        chain_type: ChainType,
        total_rows: usize,
    ) -> Result<ETAEstimate> {
        // Try to get estimate from historical data first
        if let Some(repo) = &self.repository {
            if let Ok(historical) = repo.get_similar_estimates(
                total_rows,
                columns.len(),
                &chain_type,
                5 // Get 5 similar estimates
            ).await {
                if !historical.is_empty() {
                    return Ok(self.calculate_from_historical(&historical, total_rows, columns.len()));
                }
            }
        }
        
        // Fall back to regression analysis
        self.calculate_eta(total_rows, columns.len(), chain_type).await
    }
    
    pub async fn calculate_eta(
        &self,
        total_rows: usize,
        column_count: usize,
        chain_type: ChainType,
    ) -> Result<ETAEstimate> {
        let metrics = self.sample_metrics.read().await;
        
        if metrics.is_empty() {
            // No historical data, use default estimates
            return Ok(self.default_estimate(total_rows, column_count, chain_type));
        }
        
        // Filter metrics for the same chain type
        let relevant_metrics: Vec<_> = metrics.iter()
            .filter(|m| std::mem::discriminant(&m.chain_type) == std::mem::discriminant(&chain_type))
            .collect();
        
        if relevant_metrics.is_empty() {
            return Ok(self.default_estimate(total_rows, column_count, chain_type));
        }
        
        // Calculate average processing rate (rows per second)
        let mut total_rate = 0.0;
        let mut rate_samples = Vec::new();
        
        for metric in &relevant_metrics {
            let total_cells = metric.rows * metric.columns;
            let seconds = metric.elapsed_ms as f64 / 1000.0;
            let rate = total_cells as f64 / seconds;
            rate_samples.push(rate);
            total_rate += rate;
        }
        
        let avg_rate = total_rate / relevant_metrics.len() as f64;
        
        // Calculate standard deviation for confidence intervals
        let variance: f64 = rate_samples.iter()
            .map(|rate| (rate - avg_rate).powi(2))
            .sum::<f64>() / rate_samples.len() as f64;
        let std_dev = variance.sqrt();
        
        // Apply chain type multiplier
        let chain_key = chain_type.to_string().to_lowercase();
        let multiplier = self.chain_multipliers.get(&chain_key).unwrap_or(&1.0);
        let adjusted_rate = avg_rate / multiplier;
        
        // Calculate estimates
        let total_cells = total_rows * column_count;
        let estimated_seconds = (total_cells as f64 / adjusted_rate) as u64;
        
        // Confidence intervals (±1 std dev)
        let confidence_factor = std_dev / adjusted_rate;
        let confidence_lower = ((estimated_seconds as f64) * (1.0 - confidence_factor)).max(0.0) as u64;
        let confidence_upper = ((estimated_seconds as f64) * (1.0 + confidence_factor)) as u64;
        
        // Calculate cost
        let cost_estimate = self.calculate_cost(total_rows, column_count, chain_type).await;
        
        Ok(ETAEstimate {
            estimated_seconds,
            confidence_lower,
            confidence_upper,
            estimated_cost: cost_estimate.estimated_cost,
            total_rows,
            total_tokens_estimate: cost_estimate.total_tokens_estimate,
        })
    }
    
    pub async fn calculate_cost(
        &self,
        total_rows: usize,
        column_count: usize,
        chain_type: ChainType,
    ) -> ETAEstimate {
        // Estimate tokens per cell based on chain type
        let tokens_per_cell = match chain_type {
            ChainType::Sentiment => 10,     // Simple sentiment analysis
            ChainType::Entity => 25,        // Entity extraction needs more context
            ChainType::Classification => 8,  // Classification is simpler
            ChainType::Custom(_) => 15,     // Default for custom chains
        };
        
        let total_cells = total_rows * column_count;
        let total_tokens = total_cells * tokens_per_cell;
        let estimated_cost = (total_tokens as f64 / 1000.0) * self.token_cost_per_1k;
        
        ETAEstimate {
            estimated_seconds: 0,
            confidence_lower: 0,
            confidence_upper: 0,
            estimated_cost,
            total_rows,
            total_tokens_estimate: total_tokens,
        }
    }
    
    fn default_estimate(&self, total_rows: usize, column_count: usize, chain_type: ChainType) -> ETAEstimate {
        // Default processing rate: 100 cells per second
        let default_rate = 100.0;
        let chain_key = chain_type.to_string().to_lowercase();
        let multiplier = self.chain_multipliers.get(&chain_key).unwrap_or(&1.0);
        let adjusted_rate = default_rate / multiplier;
        
        let total_cells = total_rows * column_count;
        let estimated_seconds = (total_cells as f64 / adjusted_rate) as u64;
        
        // Wide confidence intervals for default estimates
        let confidence_lower = (estimated_seconds as f64 * 0.5) as u64;
        let confidence_upper = (estimated_seconds as f64 * 2.0) as u64;
        
        let cost_estimate = futures::executor::block_on(
            self.calculate_cost(total_rows, column_count, chain_type)
        );
        
        ETAEstimate {
            estimated_seconds,
            confidence_lower,
            confidence_upper,
            estimated_cost: cost_estimate.estimated_cost,
            total_rows,
            total_tokens_estimate: cost_estimate.total_tokens_estimate,
        }
    }
    
    fn calculate_from_historical(
        &self,
        historical: &[HistoricalEstimate],
        total_rows: usize,
        column_count: usize,
    ) -> ETAEstimate {
        // Use weighted average based on similarity to current request
        let mut weighted_time = 0.0;
        let mut total_weight = 0.0;
        
        for hist in historical {
            // Calculate similarity weight based on row count and column count
            let row_ratio = (total_rows as f64 / hist.row_count as f64).min(hist.row_count as f64 / total_rows as f64);
            let col_ratio = (column_count as f64 / hist.column_count as f64).min(hist.column_count as f64 / column_count as f64);
            let weight = row_ratio * col_ratio;
            
            if let Some(actual_seconds) = hist.actual_seconds {
                weighted_time += actual_seconds as f64 * weight;
                total_weight += weight;
            }
        }
        
        let estimated_seconds = if total_weight > 0.0 {
            (weighted_time / total_weight) as u64
        } else {
            // Fallback to simple scaling
            let avg_actual: f64 = historical.iter()
                .filter_map(|h| h.actual_seconds.map(|s| s as f64))
                .sum::<f64>() / historical.len() as f64;
            avg_actual as u64
        };
        
        // Calculate confidence based on variance in historical data
        let variance: f64 = historical.iter()
            .filter_map(|h| h.actual_seconds.map(|s| (s as f64 - estimated_seconds as f64).powi(2)))
            .sum::<f64>() / historical.len() as f64;
        let std_dev = variance.sqrt();
        
        ETAEstimate {
            estimated_seconds,
            confidence_lower: (estimated_seconds as f64 - std_dev).max(0.0) as u64,
            confidence_upper: (estimated_seconds as f64 + std_dev) as u64,
            estimated_cost: 0.0, // Would calculate separately
            total_rows,
            total_tokens_estimate: 0, // Would calculate separately
        }
    }
}

// Service wrapper for dependency injection
pub struct ETAService {
    estimator: ETAEstimator,
}

impl ETAService {
    pub fn new(estimator: ETAEstimator) -> Self {
        Self { estimator }
    }
    
    pub async fn estimate(&self, file_id: Uuid, columns: &[String], chain_type: ChainType, total_rows: usize) -> Result<ETAResponse> {
        let estimate = self.estimator.estimate_analysis(file_id, columns, chain_type, total_rows).await?;
        
        Ok(ETAResponse {
            estimated_seconds: estimate.estimated_seconds,
            confidence_lower: estimate.confidence_lower,
            confidence_upper: estimate.confidence_upper,
            estimated_cost: estimate.estimated_cost,
            total_rows: estimate.total_rows,
            total_tokens_estimate: estimate.total_tokens_estimate,
        })
    }
    
    pub async fn record_actual_result(&self, file_id: Uuid, estimated_seconds: u64, actual_seconds: u64) {
        // Record for future accuracy improvements
        if let Some(repo) = &self.estimator.repository {
            let _ = repo.record_actual_result(file_id, estimated_seconds, actual_seconds).await;
        }
    }
}

#[derive(Debug, Clone)]
pub struct HistoricalEstimate {
    pub row_count: usize,
    pub column_count: usize,
    pub estimated_seconds: u64,
    pub actual_seconds: Option<u64>,
    pub chain_type: ChainType,
}

impl std::fmt::Display for ChainType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChainType::Sentiment => write!(f, "sentiment"),
            ChainType::Entity => write!(f, "entity"),
            ChainType::Classification => write!(f, "classification"),
            ChainType::Custom(s) => write!(f, "{}", s),
        }
    }
}