use clap::{Parser, Subcommand};
use std::path::PathBuf;
use anyhow::Result;
use datacloak_core::{
    DataCloak, DataCloakConfig, DataSource,
    PatternDetector, Pattern, LlmBatchConfig
};
use serde_json;
use tracing::{info, warn, error};

#[derive(Parser)]
#[command(name = "datacloak-cli")]
#[command(about = "DataCloak CLI for functional testing and analysis")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Detect sensitive patterns in data
    Detect {
        #[arg(short, long)]
        file: PathBuf,
        #[arg(short, long, default_value = "100")]
        rows: usize,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Obfuscate data with given patterns
    Obfuscate {
        #[arg(short, long)]
        file: PathBuf,
        #[arg(short, long)]
        patterns: Option<PathBuf>,
        #[arg(short, long, default_value = "100")]
        rows: usize,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Full pipeline: detect → obfuscate → LLM analysis
    Analyze {
        #[arg(short, long)]
        file: PathBuf,
        #[arg(short, long, default_value = "100")]
        rows: usize,
        #[arg(short, long)]
        patterns: Option<PathBuf>,
        #[arg(short = 'k', long)]
        api_key: Option<String>,
        #[arg(long)]
        dry_run: bool,
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Start mock LLM server
    MockServer {
        #[arg(short, long, default_value = "3001")]
        port: u16,
        #[arg(short, long)]
        scenario: Option<String>,
    },
    /// Run a specific test scenario
    TestScenario {
        #[arg(short, long)]
        scenario: String,
        #[arg(long, default_value = "3001")]
        mock_port: u16,
    },
    /// Run all test scenarios
    TestAll {
        #[arg(long, default_value = "3001")]
        mock_port: u16,
    },
}

pub async fn detect_command(
    file: PathBuf, 
    rows: usize, 
    output: Option<PathBuf>
) -> Result<()> {
    info!("Detecting patterns in {} (first {} rows)", file.display(), rows);
    
    // Create data source
    let data_source = DataSource::csv(file.to_path_buf());
    
    // Create detector
    let detector = PatternDetector::new(0.1); // 10% confidence threshold
    
    // Detect patterns
    let detection_result = detector.detect_patterns(data_source).await?;
    
    // Display results
    println!("🔍 Pattern Detection Results");
    println!("===========================");
    println!("Analyzed {} records", detection_result.sample_size);
    println!();
    
    if detection_result.detected_patterns.is_empty() {
        println!("No sensitive patterns detected.");
        return Ok(());
    }
    
    for detected_pattern in &detection_result.detected_patterns {
        let sample = detected_pattern.sample_matches.first()
            .map(|s| format!(" - sample: {}", mask_sample(s)))
            .unwrap_or_default();
        
        println!("✓ {:?} ({} matches){}", detected_pattern.pattern_type, detected_pattern.match_count, sample);
    }
    
    // Ask user if they want to save pattern config
    println!();
    print!("Save pattern config? (y/n): ");
    use std::io::{self, Write};
    io::stdout().flush()?;
    
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    
    if input.trim().to_lowercase() == "y" {
        use datacloak_core::patterns::PatternSet;
        let default_set = PatternSet::default_pii();
        
        let patterns: Vec<Pattern> = detection_result.detected_patterns.iter()
            .filter_map(|detected_pattern| {
                default_set.as_slice().iter()
                    .find(|p| p.pattern_type == detected_pattern.pattern_type)
                    .cloned()
            })
            .collect();
        
        let output_path = output.unwrap_or_else(|| "detected_patterns.yaml".into());
        let yaml_content = serde_yaml::to_string(&patterns)?;
        tokio::fs::write(&output_path, yaml_content).await?;
        
        info!("Pattern config saved to {}", output_path.display());
    }
    
    Ok(())
}

pub async fn obfuscate_command(
    file: PathBuf,
    patterns: Option<PathBuf>,
    rows: usize,
    output: Option<PathBuf>,
) -> Result<()> {
    info!("Obfuscating data in {} (first {} rows)", file.display(), rows);
    
    // Load patterns
    let patterns = if let Some(pattern_file) = patterns {
        let content = tokio::fs::read_to_string(pattern_file).await?;
        serde_yaml::from_str::<Vec<Pattern>>(&content)?
    } else {
        // Default patterns
        use datacloak_core::patterns::PatternSet;
        let default_set = PatternSet::default_pii();
        default_set.to_vec()
    };
    
    // Create DataCloak instance
    let config = DataCloakConfig::default();
    let datacloak = DataCloak::new(config);
    datacloak.set_patterns(patterns)?;
    
    // Create data source
    let data_source = DataSource::csv(file.to_path_buf());
    
    // Process data
    let mut stream = data_source.stream(1000).await?;
    let mut all_obfuscated = Vec::new();
    
    use futures::StreamExt;
    while let Some(batch) = stream.next().await {
        let batch = batch?;
        let obfuscated_batch = datacloak.obfuscate_batch(batch).await?;
        all_obfuscated.extend(obfuscated_batch);
    }
    
    // Display results
    println!("🔒 Obfuscation Results");
    println!("======================");
    let stats = datacloak.obfuscator_stats();
    println!("Processed {} records", all_obfuscated.len());
    println!("Generated {} obfuscation tokens", stats.total_tokens);
    println!("Loaded {} patterns", stats.patterns_loaded);
    
    // Save output
    if let Some(output_path) = output {
        let json_output = serde_json::to_string_pretty(&all_obfuscated)?;
        tokio::fs::write(&output_path, json_output).await?;
        info!("Obfuscated data saved to {}", output_path.display());
    }
    
    Ok(())
}

pub async fn analyze_command(
    file: PathBuf,
    rows: usize,
    patterns: Option<PathBuf>,
    api_key: Option<String>,
    dry_run: bool,
    output: Option<PathBuf>,
) -> Result<()> {
    info!("Running full analysis pipeline on {} (first {} rows)", file.display(), rows);
    
    if dry_run {
        println!("🔍 DRY RUN MODE - No LLM calls will be made");
        println!("============================================");
    }
    
    // Load patterns
    let patterns = if let Some(pattern_file) = patterns {
        let content = tokio::fs::read_to_string(pattern_file).await?;
        serde_yaml::from_str::<Vec<Pattern>>(&content)?
    } else {
        // Auto-detect patterns first
        warn!("No pattern file provided, auto-detecting...");
        
        let data_source = DataSource::csv(file.to_path_buf());
        let detector = PatternDetector::new(0.1);
        let detection_result = detector.detect_patterns(data_source).await?;
        
        use datacloak_core::patterns::PatternSet;
        let default_set = PatternSet::default_pii();
        
        detection_result.detected_patterns.iter()
            .filter_map(|detected| {
                default_set.as_slice().iter()
                    .find(|p| p.pattern_type == detected.pattern_type)
                    .cloned()
            })
            .collect()
    };
    
    println!("Loading patterns: {:?}", patterns.iter().map(|p| &p.pattern_type).collect::<Vec<_>>());
    
    // Create DataCloak instance
    let mut config = DataCloakConfig::default();
    
    // Configure LLM
    if let Some(key) = api_key.or_else(|| std::env::var("OPENAI_API_KEY").ok()) {
        config.llm_config = LlmBatchConfig {
            endpoint: "https://api.openai.com/v1/chat/completions".to_string(),
            api_key: key,
            model: "gpt-3.5-turbo".to_string(),
            batch_size: 10,
            max_concurrent_calls: 2,
            timeout: std::time::Duration::from_secs(30),
            rate_limit: Some(10.0),
            system_prompt: "Analyze customer data for churn prediction.".to_string(),
        };
    } else if !dry_run {
        error!("API key required for LLM analysis. Use --api-key or set OPENAI_API_KEY env var");
        return Err(anyhow::anyhow!("Missing API key"));
    }
    
    let datacloak = DataCloak::new(config);
    
    // Create data source
    let data_source = DataSource::csv(file.to_path_buf());
    
    if dry_run {
        // Just show what would be processed
        datacloak.set_patterns(patterns)?;
        let mut stream = data_source.stream(1000).await?;
        
        use futures::StreamExt;
        while let Some(batch) = stream.next().await {
            let batch = batch?;
            let obfuscated_batch = datacloak.obfuscate_batch(batch).await?;
            
            println!("📊 Would process batch of {} records", obfuscated_batch.len());
            if let Some(first_record) = obfuscated_batch.first() {
                println!("Sample obfuscated record: {}", 
                    serde_json::to_string_pretty(&first_record.data)?
                );
            }
        }
        
        let stats = datacloak.obfuscator_stats();
        println!("\n📈 Obfuscation Stats:");
        println!("Total tokens generated: {}", stats.total_tokens);
        println!("Patterns loaded: {}", stats.patterns_loaded);
        
    } else {
        // Run full churn analysis
        println!("🤖 Sending to LLM for churn analysis...");
        
        let analysis_result = datacloak.analyze_churn(data_source, patterns, Some(rows)).await?;
        
        println!("\n📊 Churn Analysis Results");
        println!("=========================");
        println!("Total records processed: {}", analysis_result.total_records);
        println!("Average churn probability: {:.1}%", analysis_result.average_churn_probability * 100.0);
        println!("High-risk customers (>70%): {}", analysis_result.high_risk_customers);
        
        if !analysis_result.errors.is_empty() {
            println!("\n⚠️  Errors encountered:");
            for error in &analysis_result.errors {
                println!("  - {}", error);
            }
        }
        
        // Show top predictions
        println!("\n🔍 Top Risk Customers:");
        let mut sorted_predictions = analysis_result.predictions.clone();
        sorted_predictions.sort_by(|a, b| b.churn_probability.partial_cmp(&a.churn_probability).unwrap());
        
        for prediction in sorted_predictions.iter().take(5) {
            let customer_id = prediction.customer_id.as_deref().unwrap_or("Unknown");
            println!("  Customer {}: {:.1}% churn risk (confidence: {:.1}%)", 
                customer_id,
                prediction.churn_probability * 100.0,
                prediction.confidence * 100.0
            );
            if !prediction.reasoning.is_empty() {
                println!("    Reasoning: {}", prediction.reasoning);
            }
        }
        
        // Save output (TODO: implement Serialize for ChurnAnalysisResult)
        if let Some(output_path) = output {
            let simple_output = format!(
                "Churn Analysis Results\n======================\nTotal records: {}\nAverage churn: {:.1}%\nHigh risk: {}\n",
                analysis_result.total_records,
                analysis_result.average_churn_probability * 100.0,
                analysis_result.high_risk_customers
            );
            tokio::fs::write(&output_path, simple_output).await?;
            info!("Analysis results saved to {}", output_path.display());
        }
    }
    
    Ok(())
}

fn mask_sample(value: &str) -> String {
    if value.len() <= 4 {
        "*".repeat(value.len())
    } else {
        format!("{}***{}", &value[..2], &value[value.len()-2..])
    }
}