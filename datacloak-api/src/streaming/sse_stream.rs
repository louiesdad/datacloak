use actix_web_lab::sse::{self, Sse};
use futures::{stream::{self, Stream}, StreamExt};
use tokio::sync::mpsc;
use tokio::time::{interval, Duration};
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use anyhow::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    Result {
        record_id: String,
        column: String,
        result: serde_json::Value,
        sequence: u64,
    },
    Progress {
        processed: u64,
        total: u64,
        percentage: f32,
        eta_seconds: Option<u64>,
    },
    Error {
        message: String,
        column: Option<String>,
        recoverable: bool,
    },
    Complete {
        total_processed: u64,
        duration_ms: u64,
    },
    Heartbeat,
}

pub struct SseStream {
    receiver: mpsc::Receiver<StreamEvent>,
    heartbeat_interval: Duration,
    buffer_size: usize,
}

impl SseStream {
    pub fn new(buffer_size: usize) -> (Self, mpsc::Sender<StreamEvent>) {
        let (sender, receiver) = mpsc::channel(buffer_size);
        
        Self {
            receiver,
            heartbeat_interval: Duration::from_secs(30),
            buffer_size,
        }
        .with_sender(sender)
    }
    
    fn with_sender(self, sender: mpsc::Sender<StreamEvent>) -> (Self, mpsc::Sender<StreamEvent>) {
        (self, sender)
    }
    
    pub fn with_heartbeat(mut self, interval: Duration) -> Self {
        self.heartbeat_interval = interval;
        self
    }
    
    pub fn into_sse(self) -> impl actix_web::Responder {
        let mut receiver = self.receiver;
        
        let event_stream = async_stream::stream! {
            while let Some(event) = receiver.recv().await {
                match &event {
                    StreamEvent::Heartbeat => {
                        yield Ok::<_, anyhow::Error>(sse::Event::Comment("heartbeat".into()));
                    }
                    _ => {
                        let data = sse::Data::new_json(&event).unwrap();
                        yield Ok(sse::Event::Data(data));
                        
                        if matches!(event, StreamEvent::Complete { .. }) {
                            break;
                        }
                    }
                }
            }
        };
        
        Sse::from_stream(event_stream)
            .with_retry_duration(Duration::from_secs(5))
            .with_keep_alive(Duration::from_secs(15))
    }
}

// Stream manager for handling multiple concurrent streams
pub struct StreamConnection {
    pub id: Uuid,
    pub sender: mpsc::Sender<StreamEvent>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_event_at: chrono::DateTime<chrono::Utc>,
}

impl StreamConnection {
    pub async fn send_event(&mut self, event: StreamEvent) -> Result<()> {
        // Implement backpressure handling
        match self.sender.try_send(event.clone()) {
            Ok(()) => {
                self.last_event_at = chrono::Utc::now();
                Ok(())
            }
            Err(mpsc::error::TrySendError::Full(_)) => {
                // Channel is full, wait with timeout
                tokio::time::timeout(
                    Duration::from_secs(5),
                    self.sender.send(event)
                ).await??;
                self.last_event_at = chrono::Utc::now();
                Ok(())
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                Err(anyhow::anyhow!("Stream connection closed"))
            }
        }
    }
    
    pub fn is_stale(&self, timeout: Duration) -> bool {
        let elapsed = chrono::Utc::now() - self.last_event_at;
        elapsed > chrono::Duration::from_std(timeout).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_sse_stream_events() {
        let (stream, sender) = SseStream::new(10);
        
        // Send test events
        let events = vec![
            StreamEvent::Progress {
                processed: 50,
                total: 100,
                percentage: 50.0,
                eta_seconds: Some(30),
            },
            StreamEvent::Result {
                record_id: "rec_1".to_string(),
                column: "description".to_string(),
                result: serde_json::json!({"sentiment": "positive"}),
                sequence: 1,
            },
            StreamEvent::Complete {
                total_processed: 100,
                duration_ms: 5000,
            },
        ];
        
        for event in events {
            sender.send(event).await.unwrap();
        }
    }
    
    #[tokio::test]
    async fn test_backpressure_handling() {
        let (stream, sender) = SseStream::new(2); // Small buffer
        
        // Fill the buffer
        for i in 0..5 {
            let event = StreamEvent::Progress {
                processed: i,
                total: 100,
                percentage: i as f32,
                eta_seconds: None,
            };
            
            // Should handle backpressure gracefully
            if let Err(e) = sender.try_send(event.clone()) {
                // Wait and retry
                tokio::time::sleep(Duration::from_millis(10)).await;
                sender.send(event).await.unwrap();
            }
        }
    }
}