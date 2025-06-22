use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use tokio::sync::mpsc;
use tokio::time::{interval, Duration};
use anyhow::Result;

use super::{StreamEvent, StreamConnection};

#[derive(Clone)]
pub struct StreamManager {
    connections: Arc<RwLock<HashMap<Uuid, StreamConnection>>>,
    cleanup_interval: Duration,
    stale_timeout: Duration,
}

impl StreamManager {
    pub fn new() -> Self {
        let manager = Self {
            connections: Arc::new(RwLock::new(HashMap::new())),
            cleanup_interval: Duration::from_secs(60),
            stale_timeout: Duration::from_secs(300), // 5 minutes
        };
        
        // Start cleanup task
        manager.start_cleanup_task();
        manager
    }
    
    pub async fn create_stream(&self, buffer_size: usize) -> (Uuid, mpsc::Sender<StreamEvent>) {
        let stream_id = Uuid::new_v4();
        let (sender, _) = mpsc::channel(buffer_size);
        
        let connection = StreamConnection {
            id: stream_id,
            sender: sender.clone(),
            created_at: chrono::Utc::now(),
            last_event_at: chrono::Utc::now(),
        };
        
        let mut connections = self.connections.write().await;
        connections.insert(stream_id, connection);
        
        (stream_id, sender)
    }
    
    pub async fn get_connection(&self, stream_id: &Uuid) -> Option<mpsc::Sender<StreamEvent>> {
        let connections = self.connections.read().await;
        connections.get(stream_id).map(|conn| conn.sender.clone())
    }
    
    pub async fn remove_stream(&self, stream_id: &Uuid) {
        let mut connections = self.connections.write().await;
        connections.remove(stream_id);
    }
    
    pub async fn broadcast_to_run(&self, run_id: &Uuid, event: StreamEvent) -> Result<()> {
        // In a real implementation, we'd track which streams belong to which run
        // For now, broadcast to all active streams
        let connections = self.connections.read().await;
        
        for (_, conn) in connections.iter() {
            // Ignore send errors for individual connections
            let _ = conn.sender.try_send(event.clone());
        }
        
        Ok(())
    }
    
    pub async fn get_active_count(&self) -> usize {
        let connections = self.connections.read().await;
        connections.len()
    }
    
    fn start_cleanup_task(&self) {
        let connections = self.connections.clone();
        let cleanup_interval = self.cleanup_interval;
        let stale_timeout = self.stale_timeout;
        
        tokio::spawn(async move {
            let mut interval = interval(cleanup_interval);
            
            loop {
                interval.tick().await;
                
                let mut conn_write = connections.write().await;
                let now = chrono::Utc::now();
                
                // Remove stale connections
                conn_write.retain(|_, conn| {
                    let elapsed = now.signed_duration_since(conn.last_event_at);
                    elapsed < chrono::Duration::from_std(stale_timeout).unwrap()
                });
            }
        });
    }
}

// Recovery mechanism for reconnecting streams
pub struct StreamRecovery {
    last_sequence: HashMap<Uuid, u64>,
    buffer_size: usize,
}

impl StreamRecovery {
    pub fn new(buffer_size: usize) -> Self {
        Self {
            last_sequence: HashMap::new(),
            buffer_size,
        }
    }
    
    pub fn record_sequence(&mut self, stream_id: Uuid, sequence: u64) {
        self.last_sequence.insert(stream_id, sequence);
    }
    
    pub fn get_last_sequence(&self, stream_id: &Uuid) -> Option<u64> {
        self.last_sequence.get(stream_id).copied()
    }
    
    pub async fn recover_stream(
        &mut self,
        stream_id: Uuid,
        from_sequence: u64,
        log_repository: &crate::repositories::AnalysisLogRepository,
    ) -> Result<Vec<StreamEvent>> {
        // Fetch logs after the given sequence
        // This would need additional repository methods in a real implementation
        Ok(vec![])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_stream_manager() {
        let manager = StreamManager::new();
        
        // Create multiple streams
        let (id1, sender1) = manager.create_stream(100).await;
        let (id2, sender2) = manager.create_stream(100).await;
        
        assert_eq!(manager.get_active_count().await, 2);
        
        // Send events
        let event = StreamEvent::Heartbeat;
        sender1.send(event.clone()).await.unwrap();
        sender2.send(event).await.unwrap();
        
        // Remove stream
        manager.remove_stream(&id1).await;
        assert_eq!(manager.get_active_count().await, 1);
    }
    
    #[tokio::test]
    async fn test_broadcast() {
        let manager = StreamManager::new();
        
        // Create streams
        let (_, mut rx1) = mpsc::channel::<StreamEvent>(10);
        let (_, mut rx2) = mpsc::channel::<StreamEvent>(10);
        
        manager.create_stream(10).await;
        manager.create_stream(10).await;
        
        // Broadcast event
        let event = StreamEvent::Progress {
            processed: 50,
            total: 100,
            percentage: 50.0,
            eta_seconds: Some(30),
        };
        
        manager.broadcast_to_run(&Uuid::new_v4(), event.clone()).await.unwrap();
    }
}