use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{Duration, Instant};
use std::collections::HashMap;

#[derive(Clone)]
pub struct ProgressTracker {
    inner: Arc<RwLock<ProgressTrackerInner>>,
}

struct ProgressTrackerInner {
    total_items: u64,
    processed_items: HashMap<String, u64>, // column -> processed count
    start_time: Instant,
    last_update: Instant,
    processing_rate: f64, // items per second
}

impl ProgressTracker {
    pub fn new(total_items: u64, columns: Vec<String>) -> Self {
        let mut processed_items = HashMap::new();
        for column in columns {
            processed_items.insert(column, 0);
        }
        
        Self {
            inner: Arc::new(RwLock::new(ProgressTrackerInner {
                total_items,
                processed_items,
                start_time: Instant::now(),
                last_update: Instant::now(),
                processing_rate: 0.0,
            })),
        }
    }
    
    pub async fn update(&self, column: &str, processed: u64) {
        let mut inner = self.inner.write().await;
        
        if let Some(count) = inner.processed_items.get_mut(column) {
            *count = processed;
        }
        
        // Calculate processing rate
        let elapsed = inner.last_update.elapsed().as_secs_f64();
        if elapsed > 1.0 {
            let total_processed: u64 = inner.processed_items.values().sum();
            inner.processing_rate = total_processed as f64 / inner.start_time.elapsed().as_secs_f64();
            inner.last_update = Instant::now();
        }
    }
    
    pub async fn increment(&self, column: &str) {
        let mut inner = self.inner.write().await;
        
        if let Some(count) = inner.processed_items.get_mut(column) {
            *count += 1;
        }
    }
    
    pub async fn get_progress(&self) -> Progress {
        let inner = self.inner.read().await;
        
        let total_processed: u64 = inner.processed_items.values().sum();
        let percentage = if inner.total_items > 0 {
            (total_processed as f64 / inner.total_items as f64) * 100.0
        } else {
            0.0
        };
        
        let eta_seconds = if inner.processing_rate > 0.0 {
            let remaining = inner.total_items.saturating_sub(total_processed);
            Some((remaining as f64 / inner.processing_rate) as u64)
        } else {
            None
        };
        
        Progress {
            total: inner.total_items,
            processed: total_processed,
            percentage: percentage as f32,
            eta_seconds,
            elapsed_seconds: inner.start_time.elapsed().as_secs(),
            processing_rate: inner.processing_rate,
            column_progress: inner.processed_items.clone(),
        }
    }
    
    pub async fn is_complete(&self) -> bool {
        let inner = self.inner.read().await;
        let total_processed: u64 = inner.processed_items.values().sum();
        total_processed >= inner.total_items
    }
}

#[derive(Debug, Clone)]
pub struct Progress {
    pub total: u64,
    pub processed: u64,
    pub percentage: f32,
    pub eta_seconds: Option<u64>,
    pub elapsed_seconds: u64,
    pub processing_rate: f64,
    pub column_progress: HashMap<String, u64>,
}

impl Progress {
    pub fn to_stream_event(&self) -> crate::streaming::StreamEvent {
        crate::streaming::StreamEvent::Progress {
            processed: self.processed,
            total: self.total,
            percentage: self.percentage,
            eta_seconds: self.eta_seconds,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_progress_tracking() {
        let tracker = ProgressTracker::new(100, vec!["col1".to_string(), "col2".to_string()]);
        
        // Update progress for different columns
        tracker.update("col1", 30).await;
        tracker.update("col2", 20).await;
        
        let progress = tracker.get_progress().await;
        assert_eq!(progress.processed, 50);
        assert_eq!(progress.percentage, 50.0);
        
        // Test incremental updates
        for _ in 0..10 {
            tracker.increment("col1").await;
        }
        
        let progress = tracker.get_progress().await;
        assert_eq!(progress.processed, 60);
        assert_eq!(progress.column_progress["col1"], 40);
    }
    
    #[tokio::test]
    async fn test_eta_calculation() {
        let tracker = ProgressTracker::new(1000, vec!["col1".to_string()]);
        
        // Simulate processing over time
        for i in 0..100 {
            tracker.update("col1", i).await;
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        
        let progress = tracker.get_progress().await;
        assert!(progress.eta_seconds.is_some());
        assert!(progress.processing_rate > 0.0);
    }
}