use datacloak_api::services::{ETAEstimator, ETAService};
use datacloak_api::models::{ChainType, ETAResponse};
use uuid::Uuid;

#[tokio::test]
async fn test_eta_estimation_endpoint() {
    let estimator = ETAEstimator::new();
    
    let estimate = estimator.estimate_analysis(
        Uuid::new_v4(),
        &["col1".to_string(), "col2".to_string()],
        ChainType::Sentiment,
        100_000, // total rows
    ).await.unwrap();
    
    assert!(estimate.estimated_seconds > 0);
    assert!(estimate.confidence_upper > estimate.estimated_seconds);
    assert!(estimate.estimated_cost > 0.0);
    assert_eq!(estimate.total_rows, 100_000);
}

#[tokio::test]
async fn test_eta_calculation_accuracy() {
    let estimator = ETAEstimator::new();
    
    // Add sample metric for regression analysis
    estimator.add_sample_metric(SampleMetric {
        rows: 1000,
        columns: 2,
        elapsed_ms: 5000,
        tokens_used: 2000,
        chain_type: ChainType::Sentiment,
    }).await;
    
    // Estimate for larger file
    let estimate = estimator.calculate_eta(
        100_000, // total rows
        2,       // columns
        ChainType::Sentiment,
    ).await.unwrap();
    
    // Should scale roughly linearly (100x data = ~100x time, with variance)
    assert!(estimate.estimated_seconds >= 450); // At least 450s
    assert!(estimate.estimated_seconds <= 550); // At most 550s
    
    // Check confidence intervals
    assert!(estimate.confidence_lower < estimate.estimated_seconds);
    assert!(estimate.confidence_upper > estimate.estimated_seconds);
}

#[tokio::test]
async fn test_eta_cost_calculation() {
    let estimator = ETAEstimator::new();
    
    let cost = estimator.calculate_cost(
        100_000,           // rows
        3,                 // columns
        ChainType::Sentiment,
    ).await;
    
    // Based on typical LLM pricing: $0.10 per 1000 tokens, ~10 tokens per cell
    let expected_tokens = 100_000 * 3 * 10;
    let expected_cost = (expected_tokens as f64 / 1000.0) * 0.10;
    
    assert!((cost.estimated_cost - expected_cost).abs() < 10.0); // Within $10
    assert_eq!(cost.total_tokens_estimate, expected_tokens);
}

#[tokio::test]
async fn test_eta_with_different_chain_types() {
    let estimator = ETAEstimator::new();
    
    let sentiment_eta = estimator.calculate_eta(10_000, 2, ChainType::Sentiment).await.unwrap();
    let entity_eta = estimator.calculate_eta(10_000, 2, ChainType::Entity).await.unwrap();
    let classification_eta = estimator.calculate_eta(10_000, 2, ChainType::Classification).await.unwrap();
    
    // Entity extraction should be slower than sentiment
    assert!(entity_eta.estimated_seconds > sentiment_eta.estimated_seconds);
    
    // Classification should be fastest
    assert!(classification_eta.estimated_seconds <= sentiment_eta.estimated_seconds);
}

#[tokio::test]
async fn test_eta_regression_learning() {
    let estimator = ETAEstimator::new();
    
    // Add multiple samples to improve accuracy
    let samples = vec![
        SampleMetric { rows: 1000, columns: 1, elapsed_ms: 2000, tokens_used: 1000, chain_type: ChainType::Sentiment },
        SampleMetric { rows: 2000, columns: 1, elapsed_ms: 4000, tokens_used: 2000, chain_type: ChainType::Sentiment },
        SampleMetric { rows: 5000, columns: 2, elapsed_ms: 12000, tokens_used: 5000, chain_type: ChainType::Sentiment },
        SampleMetric { rows: 10000, columns: 2, elapsed_ms: 22000, tokens_used: 10000, chain_type: ChainType::Sentiment },
    ];
    
    for sample in samples {
        estimator.add_sample_metric(sample).await;
    }
    
    // Test prediction
    let eta = estimator.calculate_eta(20_000, 2, ChainType::Sentiment).await.unwrap();
    
    // Should be more accurate with more data points
    assert!(eta.estimated_seconds > 40); // Roughly 44 seconds expected
    assert!(eta.estimated_seconds < 50);
    
    // Confidence should be narrower with more data
    let confidence_range = eta.confidence_upper - eta.confidence_lower;
    assert!(confidence_range < eta.estimated_seconds); // Range < estimate
}

#[derive(Debug, Clone)]
struct SampleMetric {
    rows: usize,
    columns: usize,
    elapsed_ms: u64,
    tokens_used: usize,
    chain_type: ChainType,
}