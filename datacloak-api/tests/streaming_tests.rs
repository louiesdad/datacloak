use datacloak_api::streaming::{SseStream, StreamEvent, ProgressTracker, StreamManager};
use tokio::time::{timeout, Duration};
use futures::StreamExt;

#[tokio::test]
async fn test_sse_stream_with_backpressure() {
    let (sse_stream, sender) = SseStream::new(5); // Small buffer
    
    // Send events rapidly
    let send_task = tokio::spawn(async move {
        for i in 0..20 {
            let event = StreamEvent::Result {
                record_id: format!("rec_{}", i),
                column: "test_column".to_string(),
                result: serde_json::json!({"value": i}),
                sequence: i,
            };
            
            // Should handle backpressure
            if let Err(_) = timeout(Duration::from_secs(1), sender.send(event)).await {
                eprintln!("Send timeout at {}", i);
                break;
            }
        }
        
        // Send completion
        let _ = sender.send(StreamEvent::Complete {
            total_processed: 20,
            duration_ms: 1000,
        }).await;
    });
    
    // Consume events
    let mut count = 0;
    let mut stream = sse_stream.into_sse();
    
    while let Ok(Some(event)) = timeout(Duration::from_millis(100), stream.next()).await {
        count += 1;
        if count >= 20 {
            break;
        }
    }
    
    send_task.await.unwrap();
    assert!(count >= 10); // Should receive at least some events
}

#[tokio::test]
async fn test_progress_tracker_multi_column() {
    let columns = vec!["col1".to_string(), "col2".to_string(), "col3".to_string()];
    let tracker = ProgressTracker::new(300, columns); // 100 items per column
    
    // Simulate processing different columns at different rates
    for i in 0..100 {
        tracker.update("col1", i).await;
        if i % 2 == 0 {
            tracker.update("col2", i / 2).await;
        }
        if i % 3 == 0 {
            tracker.update("col3", i / 3).await;
        }
        
        if i % 10 == 0 {
            let progress = tracker.get_progress().await;
            println!("Progress at {}: {:.2}%", i, progress.percentage);
            
            // Verify column-specific progress
            assert_eq!(progress.column_progress["col1"], i);
        }
    }
    
    // Final progress check
    let final_progress = tracker.get_progress().await;
    assert_eq!(final_progress.column_progress["col1"], 99);
    assert_eq!(final_progress.column_progress["col2"], 49);
    assert_eq!(final_progress.column_progress["col3"], 33);
}

#[tokio::test]
async fn test_stream_reconnection() {
    let (sse_stream, sender) = SseStream::new(100);
    
    // Send some events
    for i in 0..5 {
        let event = StreamEvent::Result {
            record_id: format!("rec_{}", i),
            column: "test".to_string(),
            result: serde_json::json!({"seq": i}),
            sequence: i,
        };
        sender.send(event).await.unwrap();
    }
    
    // Simulate disconnect by dropping sender
    drop(sender);
    
    // Stream should complete gracefully
    let mut stream = sse_stream.into_sse();
    let mut received = 0;
    
    while let Some(_) = stream.next().await {
        received += 1;
    }
    
    assert_eq!(received, 5);
}

#[tokio::test]
async fn test_stream_manager_lifecycle() {
    let manager = StreamManager::new();
    
    // Create multiple streams
    let mut stream_ids = vec![];
    for _ in 0..5 {
        let (id, _sender) = manager.create_stream(100).await;
        stream_ids.push(id);
    }
    
    assert_eq!(manager.get_active_count().await, 5);
    
    // Remove some streams
    manager.remove_stream(&stream_ids[0]).await;
    manager.remove_stream(&stream_ids[2]).await;
    
    assert_eq!(manager.get_active_count().await, 3);
    
    // Test broadcast
    let event = StreamEvent::Progress {
        processed: 50,
        total: 100,
        percentage: 50.0,
        eta_seconds: Some(30),
    };
    
    manager.broadcast_to_run(&uuid::Uuid::new_v4(), event).await.unwrap();
}

#[tokio::test]
async fn test_heartbeat_mechanism() {
    let (sse_stream, sender) = SseStream::new(10)
        .with_heartbeat(Duration::from_millis(100));
    
    // Don't send any events
    let _send_handle = sender;
    
    // Should receive heartbeats
    let mut stream = sse_stream.into_sse();
    let mut heartbeat_count = 0;
    
    let start = tokio::time::Instant::now();
    while start.elapsed() < Duration::from_millis(350) {
        if let Ok(Some(_)) = timeout(Duration::from_millis(150), stream.next()).await {
            heartbeat_count += 1;
        }
    }
    
    assert!(heartbeat_count >= 2); // Should receive at least 2 heartbeats
}