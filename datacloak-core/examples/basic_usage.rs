//! Basic usage example for DataCloak library

use datacloak_core::{
    DataCloak, DataCloakConfig, DataSource, DataSourceConfig,
    Pattern, PatternSet, PatternType,
};
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    // Configure DataCloak
    let mut config = DataCloakConfig::default();
    
    // Set API key from environment
    config.llm_config.api_key = env::var("OPENAI_API_KEY")
        .unwrap_or_else(|_| "your-api-key-here".to_string());
    
    // Adjust for demo
    config.batch_size = 100;
    config.llm_config.batch_size = 5;
    config.llm_config.rate_limit = Some(2.0); // 2 requests per second
    
    // Create DataCloak instance
    let datacloak = DataCloak::new(config);
    
    // Example 1: Detect patterns in CSV data
    println!("=== Pattern Detection Example ===");
    
    // Create sample data source
    let csv_source = DataSource::csv("sample_customers.csv".into());
    
    // Detect patterns
    match datacloak.detect_patterns(csv_source.clone()).await {
        Ok(detection) => {
            println!("Detection complete!");
            println!("Sample size: {}", detection.sample_size);
            println!("\nDetected patterns:");
            for pattern in &detection.detected_patterns {
                println!("  - {:?}: {} occurrences (confidence: {:.2}%)", 
                    pattern.pattern_type, 
                    pattern.match_count,
                    pattern.confidence * 100.0
                );
                if !pattern.sample_matches.is_empty() {
                    println!("    Sample: {}", pattern.sample_matches[0]);
                }
            }
            
            println!("\nRecommendations:");
            for rec in &detection.recommendations {
                println!("  - {:?}: {:.2}% confidence - {}", 
                    rec.pattern_type,
                    rec.confidence * 100.0,
                    rec.reason
                );
            }
        }
        Err(e) => {
            println!("Pattern detection failed: {}", e);
        }
    }
    
    // Example 2: Process with custom patterns
    println!("\n=== Custom Pattern Example ===");
    
    // Create custom patterns
    let mut patterns = PatternSet::default_pii();
    
    // Add a custom employee ID pattern
    patterns.add(Pattern::new(
        PatternType::Custom(1),
        r"\bEMP\d{6}\b".to_string(),
    ).with_description("Employee ID".to_string()));
    
    // Example 3: PostgreSQL data source
    println!("\n=== PostgreSQL Example ===");
    
    let _pg_source = DataSource::postgres(
        "postgresql://user:password@localhost/testdb".to_string(),
        "SELECT id, email, phone, ssn, notes FROM customers LIMIT 1000".to_string(),
    );
    
    // Example 4: In-memory data processing
    println!("\n=== In-Memory Data Example ===");
    
    // Create sample data
    let sample_data = vec![
        serde_json::json!({
            "id": "CUST001",
            "email": "john.doe@example.com",
            "phone": "555-123-4567",
            "ssn": "123-45-6789",
            "last_purchase": "2024-01-15",
            "total_spent": 1250.00,
            "support_tickets": 3,
            "satisfaction_score": 3.5
        }),
        serde_json::json!({
            "id": "CUST002",
            "email": "jane.smith@example.com",
            "phone": "555-987-6543",
            "ssn": "987-65-4321",
            "last_purchase": "2023-11-20",
            "total_spent": 450.00,
            "support_tickets": 8,
            "satisfaction_score": 2.1
        }),
    ];
    
    let memory_source = DataSource::new(DataSourceConfig::Memory { data: sample_data });
    
    // Run churn analysis
    println!("Running churn analysis on sample data...");
    
    match datacloak.analyze_churn(
        memory_source,
        patterns.to_vec(),
        Some(10),
    ).await {
        Ok(results) => {
            println!("\nChurn Analysis Results:");
            println!("Total records: {}", results.total_records);
            println!("Average churn probability: {:.2}%", results.average_churn_probability * 100.0);
            println!("High risk customers: {}", results.high_risk_customers);
            
            if !results.errors.is_empty() {
                println!("\nErrors encountered:");
                for error in &results.errors {
                    println!("  - {}", error);
                }
            }
            
            println!("\nIndividual predictions:");
            for prediction in results.predictions.iter().take(5) {
                println!("  Customer {}: {:.2}% churn probability (confidence: {:.2}%)",
                    prediction.customer_id.as_ref().unwrap_or(&"Unknown".to_string()),
                    prediction.churn_probability * 100.0,
                    prediction.confidence * 100.0
                );
                println!("    Reasoning: {}", prediction.reasoning);
            }
        }
        Err(e) => {
            println!("Churn analysis failed: {}", e);
        }
    }
    
    // Example 5: Cache management
    println!("\n=== Cache Statistics ===");
    let stats = datacloak.cache_stats();
    println!("Cache entries: {}", stats.total_entries);
    println!("Estimated memory usage: {} bytes", stats.memory_size_estimate);
    
    Ok(())
}

// Helper function to create sample CSV file
#[allow(dead_code)]
fn create_sample_csv() -> std::io::Result<()> {
    use std::fs::File;
    use std::io::Write;
    
    let mut file = File::create("sample_customers.csv")?;
    writeln!(file, "id,email,phone,ssn,credit_card,ip_address,notes")?;
    writeln!(file, "1,john@example.com,555-123-4567,123-45-6789,4532015112830366,192.168.1.1,Customer since 2020")?;
    writeln!(file, "2,jane@test.com,555-987-6543,987-65-4321,4916338506082832,10.0.0.1,VIP customer")?;
    writeln!(file, "3,bob@demo.org,555-456-7890,456-78-9012,4485393141463880,172.16.0.1,Frequent buyer")?;
    
    Ok(())
}
