use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use anyhow::Result;
use datacloak_core::{
    DataCloak, DataCloakConfig, DataSource,
    PatternDetector, Pattern, LlmBatchConfig
};
use serde_json;
use tracing::{info, warn, error};

#[derive(Parser)]
#[command(name = "datacloak")]
#[command(about = "Multi-field sentiment analysis with auto-discovery")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Clone, Debug, ValueEnum, PartialEq)]
pub enum OutputFormat {
    Json,
    Csv,
    Stream,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Profile columns to find text-heavy candidates
    Profile {
        #[arg(short, long)]
        file: PathBuf,
        
        #[arg(short, long, default_value = "json")]
        output: OutputFormat,
        
        #[arg(long)]
        ml_only: bool,
        
        #[arg(long)]
        graph_only: bool,
    },
    
    /// Analyze sentiment in multiple fields
    Analyze {
        #[arg(short, long)]
        file: PathBuf,
        
        #[arg(short, long, value_delimiter = ',')]
        columns: Option<Vec<String>>,
        
        #[arg(long)]
        auto_discover: bool,
        
        #[arg(long, default_value = "0.7")]
        threshold: f32,
        
        #[arg(long)]
        dry_run: bool,
        
        #[arg(short, long, default_value = "json")]
        output: OutputFormat,
        
        #[arg(long)]
        mock_llm: bool,
        
        #[arg(short = 'k', long)]
        api_key: Option<String>,
    },
    
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
        #[arg(long)]
        dry_run: bool,
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
    dry_run: bool,
) -> Result<()> {
    if dry_run {
        info!("Obfuscating data in {} (first {} rows) - DRY RUN", file.display(), rows);
        println!("🔍 DRY RUN MODE - No output file will be created");
        println!("================================================");
    } else {
        info!("Obfuscating data in {} (first {} rows)", file.display(), rows);
    }
    
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
    datacloak.set_patterns(patterns.clone())?;
    
    // Create data source
    let data_source = DataSource::csv(file.to_path_buf());
    
    // Process data
    let mut stream = data_source.stream(1000).await?;
    let mut all_obfuscated = Vec::new();
    let mut _total_records = 0;
    let mut row_count = 0;
    
    use futures::StreamExt;
    while let Some(batch) = stream.next().await {
        let batch = batch?;
        _total_records += batch.len();
        
        let obfuscated_batch = datacloak.obfuscate_batch(batch).await?;
        all_obfuscated.extend(obfuscated_batch);
        
        row_count += all_obfuscated.len();
        if row_count >= rows {
            all_obfuscated.truncate(rows);
            break;
        }
    }
    
    // Get obfuscation stats
    let stats = datacloak.obfuscator_stats();
    
    if dry_run {
        // Create summary JSON for dry run
        let summary = serde_json::json!({
            "mode": "dry-run",
            "input_file": file.display().to_string(),
            "records_processed": all_obfuscated.len(),
            "patterns_loaded": stats.patterns_loaded,
            "tokens_generated": stats.total_tokens,
            "sample_obfuscation": if !all_obfuscated.is_empty() {
                serde_json::to_value(&all_obfuscated[0])?
            } else {
                serde_json::Value::Null
            },
            "pattern_types": patterns.iter()
                .map(|p| format!("{:?}", p.pattern_type))
                .collect::<Vec<_>>(),
            "output_would_be_written_to": output.as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "stdout".to_string()),
        });
        
        // Print summary JSON
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        // Display results
        println!("🔒 Obfuscation Results");
        println!("======================");
        println!("Processed {} records", all_obfuscated.len());
        println!("Generated {} obfuscation tokens", stats.total_tokens);
        println!("Loaded {} patterns", stats.patterns_loaded);
        
        // Save output
        if let Some(output_path) = output {
            let json_output = serde_json::to_string_pretty(&all_obfuscated)?;
            tokio::fs::write(&output_path, json_output).await?;
            info!("Obfuscated data saved to {}", output_path.display());
        }
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

#[derive(Debug, serde::Serialize)]
pub struct ProfileOutput {
    pub candidates: Vec<ColumnCandidate>,
}

#[derive(Debug, serde::Serialize)]
pub struct ColumnCandidate {
    pub name: String,
    pub score: f32,
    pub ml_probability: f32,
    pub graph_score: f32,
    pub column_type: String,
}

pub async fn profile_command(
    file: PathBuf,
    output: OutputFormat,
    ml_only: bool,
    graph_only: bool,
) -> Result<()> {
    info!("Profiling columns in {}", file.display());
    
    // Read CSV file to get columns
    let csv_content = tokio::fs::read_to_string(&file).await?;
    let mut reader = csv::Reader::from_reader(csv_content.as_bytes());
    
    let headers = reader.headers()?.clone();
    let column_names: Vec<String> = headers.iter().map(|h| h.to_string()).collect();
    
    // Collect sample data for each column
    let mut column_data: Vec<Vec<String>> = vec![vec![]; column_names.len()];
    let mut row_count = 0;
    
    for result in reader.records() {
        let record = result?;
        for (i, value) in record.iter().enumerate() {
            if i < column_data.len() {
                column_data[i].push(value.to_string());
            }
        }
        row_count += 1;
        if row_count >= 1000 { // Sample first 1000 rows
            break;
        }
    }
    
    // Create ML classifier
    use datacloak_core::ml_classifier::{MLClassifier, Column};
    let classifier = MLClassifier::new();
    
    // Analyze each column
    let mut candidates = Vec::new();
    
    for (i, name) in column_names.iter().enumerate() {
        let column = Column::new(name, column_data[i].iter().map(|s| s.as_str()).collect());
        
        // ML prediction
        let ml_prediction = if !graph_only {
            classifier.predict(&column)
        } else {
            datacloak_core::ml_classifier::Prediction {
                column_type: datacloak_core::ml_classifier::ColumnType::Unknown,
                confidence: 0.0,
            }
        };
        
        // Graph score (placeholder for now - Dev 2 will implement)
        let graph_score = if !ml_only {
            // Simple heuristic based on column name
            if name.contains("description") || name.contains("comment") || 
               name.contains("feedback") || name.contains("review") || 
               name.contains("notes") || name.contains("text") {
                0.8
            } else {
                0.2
            }
        } else {
            0.0
        };
        
        // Calculate combined score
        let ml_prob = ml_prediction.confidence;
        let combined_score = if ml_only {
            ml_prob
        } else if graph_only {
            graph_score
        } else {
            (ml_prob + graph_score) / 2.0
        };
        
        candidates.push(ColumnCandidate {
            name: name.clone(),
            score: combined_score,
            ml_probability: ml_prob,
            graph_score,
            column_type: format!("{:?}", ml_prediction.column_type),
        });
    }
    
    // Sort by score descending
    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    
    let profile_output = ProfileOutput { candidates };
    
    // Output results
    match output {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&profile_output)?);
        }
        OutputFormat::Csv => {
            println!("name,score,ml_probability,graph_score,column_type");
            for c in &profile_output.candidates {
                println!("{},{},{},{},{}", c.name, c.score, c.ml_probability, c.graph_score, c.column_type);
            }
        }
        OutputFormat::Stream => {
            for c in &profile_output.candidates {
                println!("{}", serde_json::to_string(&c)?);
            }
        }
    }
    
    Ok(())
}

#[derive(Debug, serde::Serialize)]
pub struct AnalysisResult {
    pub record_id: String,
    pub column: String,
    pub sentiment: String,
    pub confidence: f32,
}

#[derive(Debug, serde::Serialize)]
pub struct DryRunOutput {
    pub estimated_time_seconds: u64,
    pub estimated_cost_usd: f64,
    pub selected_columns: Vec<String>,
}

pub async fn analyze_multi_field_command(
    file: PathBuf,
    columns: Option<Vec<String>>,
    auto_discover: bool,
    _threshold: f32,
    dry_run: bool,
    output: OutputFormat,
    mock_llm: bool,
    _api_key: Option<String>,
) -> Result<()> {
    info!("Running multi-field analysis on {}", file.display());
    
    // Determine which columns to analyze
    let selected_columns = if auto_discover {
        // Run profiling to discover columns
        let csv_content = tokio::fs::read_to_string(&file).await?;
        let mut reader = csv::Reader::from_reader(csv_content.as_bytes());
        let headers = reader.headers()?.clone();
        let column_names: Vec<String> = headers.iter().map(|h| h.to_string()).collect();
        
        // Profile columns (simplified for now)
        let mut discovered = Vec::new();
        for name in &column_names {
            // Simple heuristic - in real implementation would use ML classifier
            if name.contains("feedback") || name.contains("comment") || 
               name.contains("review") || name.contains("description") ||
               name.contains("notes") || name.contains("text") {
                discovered.push(name.clone());
            }
        }
        
        eprintln!("Auto-discovered {} columns for analysis: {:?}", discovered.len(), discovered);
        discovered
    } else if let Some(cols) = columns {
        // Validate columns exist
        let csv_content = tokio::fs::read_to_string(&file).await?;
        let mut reader = csv::Reader::from_reader(csv_content.as_bytes());
        let headers = reader.headers()?.clone();
        let header_set: std::collections::HashSet<_> = headers.iter().collect();
        
        for col in &cols {
            if !header_set.contains(col.as_str()) {
                return Err(anyhow::anyhow!("Column '{}' not found in file", col));
            }
        }
        cols
    } else {
        return Err(anyhow::anyhow!("Must specify --columns or use --auto-discover"));
    };
    
    if dry_run {
        // Estimate time and cost
        let csv_content = tokio::fs::read_to_string(&file).await?;
        let line_count = csv_content.lines().count() - 1; // Subtract header
        
        let estimated_time = (line_count * selected_columns.len()) / 100; // 100 items/second
        let estimated_cost = (line_count * selected_columns.len()) as f64 * 0.0001; // $0.0001 per item
        
        let dry_run_output = DryRunOutput {
            estimated_time_seconds: estimated_time as u64,
            estimated_cost_usd: estimated_cost,
            selected_columns: selected_columns.clone(),
        };
        
        println!("{}", serde_json::to_string_pretty(&dry_run_output)?);
        return Ok(());
    }
    
    // Process the file
    let csv_content = tokio::fs::read_to_string(&file).await?;
    let mut reader = csv::Reader::from_reader(csv_content.as_bytes());
    let headers = reader.headers()?.clone();
    
    // Find column indices
    let column_indices: Vec<usize> = selected_columns.iter()
        .filter_map(|col| headers.iter().position(|h| h == col))
        .collect();
    
    let mut results = Vec::new();
    let mut progress = 0;
    
    // Count total records first
    let record_count = csv_content.lines().count() - 1; // Subtract header
    let total_items = record_count * selected_columns.len();
    
    // Report initial progress
    if output != OutputFormat::Stream && total_items > 0 {
        eprintln!("Processing {} items across {} columns...", total_items, selected_columns.len());
    }
    
    // Re-create reader
    let mut reader = csv::Reader::from_reader(csv_content.as_bytes());
    reader.headers()?; // Skip header
    
    for (row_idx, result) in reader.records().enumerate() {
        let record = result?;
        let record_id = record.get(0).unwrap_or(&row_idx.to_string()).to_string();
        
        for (col_idx, &field_idx) in column_indices.iter().enumerate() {
            let column_name = &selected_columns[col_idx];
            let text = record.get(field_idx).unwrap_or("");
            
            // Analyze sentiment (mock for now)
            let (sentiment, confidence) = if mock_llm {
                // Simple mock sentiment based on keywords
                if text.contains("great") || text.contains("excellent") || text.contains("love") {
                    ("positive", 0.9)
                } else if text.contains("bad") || text.contains("terrible") || text.contains("hate") {
                    ("negative", 0.85)
                } else {
                    ("neutral", 0.7)
                }
            } else {
                // Would call real LLM here
                ("neutral", 0.5)
            };
            
            results.push(AnalysisResult {
                record_id: record_id.clone(),
                column: column_name.clone(),
                sentiment: sentiment.to_string(),
                confidence,
            });
            
            progress += 1;
            if output == OutputFormat::Stream {
                // Stream results immediately
                println!("{}", serde_json::to_string(&results.last().unwrap())?);
            } else if progress % 10 == 0 || progress == total_items {
                // Report progress
                let percent = (progress * 100) / total_items;
                eprintln!("Processing... {}%", percent);
            }
        }
    }
    
    // Output results
    match output {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&results)?);
        }
        OutputFormat::Csv => {
            // Output CSV with sentiment columns
            print!("id");
            for col in &selected_columns {
                print!(",{}_sentiment,{}_confidence", col, col);
            }
            println!();
            
            // Group results by record_id
            let mut record_map: std::collections::HashMap<String, Vec<&AnalysisResult>> = std::collections::HashMap::new();
            for result in &results {
                record_map.entry(result.record_id.clone()).or_default().push(result);
            }
            
            for (record_id, record_results) in record_map {
                print!("{}", record_id);
                for col in &selected_columns {
                    if let Some(result) = record_results.iter().find(|r| r.column == *col) {
                        print!(",{},{}", result.sentiment, result.confidence);
                    } else {
                        print!(",,");
                    }
                }
                println!();
            }
        }
        OutputFormat::Stream => {
            // Already streamed
        }
    }
    
    Ok(())
}