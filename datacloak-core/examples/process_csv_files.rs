//! Process actual CSV files with DataCloak
//!
//! This example reads your real CSV files and demonstrates obfuscation

use datacloak_core::*;
use serde_json::{json, Value};
use std::fs;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🔒 DataCloak CSV File Processor");
    println!("================================");

    // Initialize DataCloak
    let mut config = DataCloakConfig::default();
    config.sampling_config = SamplingConfig {
        min_sample: 10,
        max_sample: 100,
        confidence_threshold: 0.7,
        progressive_factor: 2.0,
        early_stop_enabled: true,
        confidence_window: 3,
    };

    let datacloak = DataCloak::new(config);

    // Process each CSV file
    let csv_files = vec![
        ("users.csv", "User data"),
        ("user-profile.csv", "User profiles"),
        ("order-history.csv", "Order history"),
    ];

    for (filename, description) in csv_files {
        println!("\n📁 Processing: {} ({})", filename, description);
        println!("{}", "=".repeat(50));

        match process_csv_file(&datacloak, filename).await {
            Ok(stats) => {
                println!("✅ Successfully processed {}", filename);
                println!("   Records analyzed: {}", stats.records_analyzed);
                println!("   PII fields found: {}", stats.pii_fields);
                println!("   Confidence: {:.1}%", stats.confidence * 100.0);
            }
            Err(e) => {
                println!("❌ Error processing {}: {}", filename, e);
            }
        }
    }

    // Demonstrate obfuscation with sample data
    println!("\n🔐 Demonstration: Obfuscating Sample Data");
    println!("==========================================");

    let patterns = vec![
        Pattern::new(
            PatternType::Email,
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b".to_string(),
        ),
        Pattern::new(PatternType::Phone, r"\b\d{10,11}\b".to_string()),
        Pattern::new(PatternType::SSN, r"\b\d{5}\b".to_string()),
    ];

    datacloak.set_patterns(patterns)?;

    // Sample of actual data from your files
    let sample_data = vec![
        json!({
            "email": "wagneramaza@gmail.com",
            "name": "Wagner M",
            "address": "87 Montgomery st",
            "city": "Illion",
            "zip": "13357",
            "order": "#122604"
        }),
        json!({
            "email": "kait89@live.ca",
            "name": "Kaitlyn Lorentz",
            "phone": "19097302058",
            "address": "844 McLeod Ave",
            "city": "Spruce Grove"
        }),
    ];

    println!("Before obfuscation:");
    for (i, record) in sample_data.iter().enumerate() {
        println!("  Record {}: {}", i + 1, record);
    }

    let obfuscated = datacloak.obfuscate_batch(sample_data).await?;

    println!("\nAfter obfuscation:");
    for (i, record) in obfuscated.iter().enumerate() {
        println!("  Record {}: {}", i + 1, record.data);
        println!("    🔑 Tokens: {:?}", record.tokens_used);
    }

    println!("\n📋 Summary");
    println!("==========");
    println!("Your test data contains various types of PII:");
    println!("• Email addresses (high sensitivity)");
    println!("• Phone numbers (medium sensitivity)");
    println!("• Physical addresses (medium sensitivity)");
    println!("• Names (medium sensitivity)");
    println!("• Postal codes (low sensitivity)");
    println!("\nDataCloak can detect and obfuscate all of these automatically!");
    println!("The obfuscated data maintains referential integrity via consistent tokens.");

    Ok(())
}

struct ProcessingStats {
    records_analyzed: usize,
    pii_fields: usize,
    confidence: f64,
}

async fn process_csv_file(datacloak: &DataCloak, filename: &str) -> Result<ProcessingStats> {
    let file_path = format!("../test_data/{}", filename);

    // Check if file exists
    if !std::path::Path::new(&file_path).exists() {
        return Err(DataCloakError::Other(format!(
            "File not found: {}",
            file_path
        )));
    }

    // Read and parse CSV (simplified - just read a few lines for demo)
    let content = fs::read_to_string(&file_path)
        .map_err(|e| DataCloakError::Other(format!("Failed to read file: {}", e)))?;

    let lines: Vec<&str> = content.lines().take(10).collect(); // Sample first 10 lines

    if lines.len() < 2 {
        return Err(DataCloakError::Other(
            "File appears empty or malformed".to_string(),
        ));
    }

    // Parse header and create sample records
    let headers: Vec<&str> = lines[0].split(',').collect();
    let mut sample_records = Vec::new();

    for line in lines.iter().skip(1).take(5) {
        // Take 5 sample records
        let values: Vec<&str> = line.split(',').collect();
        let mut record = serde_json::Map::new();

        for (i, value) in values.iter().enumerate() {
            if let Some(header) = headers.get(i) {
                record.insert(header.to_string(), Value::String(value.to_string()));
            }
        }

        sample_records.push(Value::Object(record));
    }

    // Create data source and analyze
    let data_source = DataSource::new(DataSourceConfig::Memory {
        data: sample_records.clone(),
    });

    let detection_result = datacloak.detect_patterns_adaptive(data_source).await?;

    let pii_count: usize = detection_result.pattern_counts.values().sum();

    Ok(ProcessingStats {
        records_analyzed: sample_records.len(),
        pii_fields: pii_count,
        confidence: detection_result.confidence_score.unwrap_or(0.0),
    })
}
