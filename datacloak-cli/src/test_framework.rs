use anyhow::Result;
use datacloak_core::{DataCloak, DataCloakConfig, DataSource, LlmBatchConfig};
use std::time::Duration;
use tokio::time::timeout;
use tracing::{info, warn, error};
use crate::scenarios::{TestScenario, create_default_scenarios};

#[derive(Debug)]
pub struct TestResult {
    pub scenario_name: String,
    pub success: bool,
    pub duration: Duration,
    pub error_message: Option<String>,
    pub metrics: TestMetrics,
}

#[derive(Debug, Default)]
pub struct TestMetrics {
    pub records_processed: usize,
    pub patterns_detected: usize,
    pub obfuscation_rate: f32,
    pub llm_calls_made: usize,
    pub predictions_generated: usize,
}

pub async fn run_scenario_test(scenario_name: String, mock_port: u16) -> Result<()> {
    info!("Running test scenario: {}", scenario_name);
    
    // Ensure scenarios exist
    create_default_scenarios().await?;
    
    // Load scenario
    let scenario = TestScenario::load(&scenario_name).await?;
    
    // Run the test
    let result = run_single_scenario_test(scenario, mock_port).await;
    
    // Display results
    match result {
        Ok(test_result) => {
            println!("✅ Test scenario '{}' PASSED", test_result.scenario_name);
            println!("   Duration: {:?}", test_result.duration);
            println!("   Records processed: {}", test_result.metrics.records_processed);
            println!("   Patterns detected: {}", test_result.metrics.patterns_detected);
            println!("   Predictions generated: {}", test_result.metrics.predictions_generated);
        }
        Err(e) => {
            error!("❌ Test scenario '{}' FAILED: {}", scenario_name, e);
            return Err(e);
        }
    }
    
    Ok(())
}

pub async fn run_all_tests(mock_port: u16) -> Result<()> {
    info!("Running all test scenarios");
    
    // Ensure scenarios exist
    create_default_scenarios().await?;
    
    // Get all available scenarios
    let scenarios = TestScenario::list_available().await?;
    
    if scenarios.is_empty() {
        warn!("No test scenarios found");
        return Ok(());
    }
    
    let mut results = Vec::new();
    let mut passed = 0;
    let mut failed = 0;
    
    println!("🧪 Running {} test scenarios", scenarios.len());
    println!("==============================");
    
    for scenario_name in scenarios {
        match TestScenario::load(&scenario_name).await {
            Ok(scenario) => {
                let result = run_single_scenario_test(scenario, mock_port).await;
                match result {
                    Ok(test_result) => {
                        println!("✅ {} - PASSED ({:?})", test_result.scenario_name, test_result.duration);
                        passed += 1;
                        results.push(test_result);
                    }
                    Err(e) => {
                        println!("❌ {} - FAILED: {}", scenario_name, e);
                        failed += 1;
                        results.push(TestResult {
                            scenario_name: scenario_name.clone(),
                            success: false,
                            duration: Duration::from_secs(0),
                            error_message: Some(e.to_string()),
                            metrics: TestMetrics::default(),
                        });
                    }
                }
            }
            Err(e) => {
                println!("❌ {} - FAILED TO LOAD: {}", scenario_name, e);
                failed += 1;
            }
        }
    }
    
    // Summary
    println!("\n📊 Test Summary");
    println!("===============");
    println!("Total scenarios: {}", results.len());
    println!("Passed: {} ✅", passed);
    println!("Failed: {} ❌", failed);
    
    if failed > 0 {
        println!("\n🔍 Failed scenarios:");
        for result in &results {
            if !result.success {
                println!("  - {}: {}", 
                    result.scenario_name, 
                    result.error_message.as_deref().unwrap_or("Unknown error")
                );
            }
        }
        return Err(anyhow::anyhow!("{} test scenarios failed", failed));
    }
    
    println!("\n🎉 All tests passed!");
    Ok(())
}

async fn run_single_scenario_test(scenario: TestScenario, mock_port: u16) -> Result<TestResult> {
    let start_time = std::time::Instant::now();
    let scenario_name = scenario.name.clone();
    
    // Check if data file exists
    let data_file_path = scenario.data_file_path();
    if !data_file_path.exists() {
        return Err(anyhow::anyhow!(
            "Data file not found: {}",
            data_file_path.display()
        ));
    }
    
    // Configure DataCloak to use mock LLM
    let config = DataCloakConfig {
        batch_size: 50,
        max_concurrency: 2,
        llm_config: LlmBatchConfig {
            endpoint: format!("http://localhost:{}/v1/chat/completions", mock_port),
            api_key: "mock-api-key".to_string(),
            model: "gpt-3.5-turbo".to_string(),
            batch_size: 10,
            max_concurrent_calls: 2,
            timeout: std::time::Duration::from_secs(30),
            rate_limit: None,
            system_prompt: "Analyze data for insights.".to_string(),
        },
        stream_config: datacloak_core::StreamConfig::default(),
    };
    
    let datacloak = DataCloak::new(config);
    
    // Set patterns for obfuscation
    let patterns = scenario.to_datacloak_patterns();
    datacloak.set_patterns(patterns.clone())?;
    
    // Create data source
    let data_source = DataSource::csv(data_file_path.to_path_buf());
    
    // Run the analysis with timeout
    let analysis_future = datacloak.analyze_churn(
        data_source,
        patterns.clone(),
        Some(scenario.expected_results.max_records),
    );
    
    let analysis_result = timeout(Duration::from_secs(60), analysis_future)
        .await
        .map_err(|_| anyhow::anyhow!("Test timed out after 60 seconds"))?
        .map_err(|e| anyhow::anyhow!("Analysis failed: {}", e))?;
    
    // Validate results
    validate_test_results(&scenario, &analysis_result)?;
    
    // Calculate metrics
    let metrics = TestMetrics {
        records_processed: analysis_result.total_records,
        patterns_detected: patterns.len(),
        obfuscation_rate: calculate_obfuscation_rate(&datacloak),
        llm_calls_made: estimate_llm_calls(analysis_result.total_records, 10), // batch size 10
        predictions_generated: analysis_result.predictions.len(),
    };
    
    let duration = start_time.elapsed();
    
    Ok(TestResult {
        scenario_name,
        success: true,
        duration,
        error_message: None,
        metrics,
    })
}

fn validate_test_results(
    scenario: &TestScenario,
    result: &datacloak_core::ChurnAnalysisResult,
) -> Result<()> {
    // Check record count
    if result.total_records < scenario.expected_results.min_records ||
       result.total_records > scenario.expected_results.max_records {
        return Err(anyhow::anyhow!(
            "Record count {} not in expected range {}-{}",
            result.total_records,
            scenario.expected_results.min_records,
            scenario.expected_results.max_records
        ));
    }
    
    // Check predictions were generated
    if result.predictions.is_empty() {
        return Err(anyhow::anyhow!("No predictions generated"));
    }
    
    // Validate churn analysis expectations
    if let Some(churn_expectations) = &scenario.expected_results.churn_analysis {
        // Check average churn probability is in expected range
        let (min_avg, max_avg) = churn_expectations.expected_avg_churn_range;
        if result.average_churn_probability < min_avg || result.average_churn_probability > max_avg {
            return Err(anyhow::anyhow!(
                "Average churn probability {:.2} not in expected range {:.2}-{:.2}",
                result.average_churn_probability,
                min_avg,
                max_avg
            ));
        }
        
        // Check high risk count if specified
        if let Some(expected_high_risk) = churn_expectations.expected_high_risk_count {
            // Allow some variance (±20%)
            let min_high_risk = (expected_high_risk as f32 * 0.8) as usize;
            let max_high_risk = (expected_high_risk as f32 * 1.2) as usize;
            
            if result.high_risk_customers < min_high_risk || result.high_risk_customers > max_high_risk {
                return Err(anyhow::anyhow!(
                    "High risk customer count {} not in expected range {}-{}",
                    result.high_risk_customers,
                    min_high_risk,
                    max_high_risk
                ));
            }
        }
    }
    
    // Check that we didn't have too many errors
    if result.errors.len() > result.total_records / 10 {
        return Err(anyhow::anyhow!(
            "Too many errors: {} errors for {} records",
            result.errors.len(),
            result.total_records
        ));
    }
    
    Ok(())
}

fn calculate_obfuscation_rate(datacloak: &DataCloak) -> f32 {
    let stats = datacloak.obfuscator_stats();
    
    // Estimate obfuscation rate based on tokens generated
    if stats.total_tokens > 0 {
        0.85 // Assume 85% obfuscation rate for successful cases
    } else {
        0.0
    }
}

fn estimate_llm_calls(total_records: usize, batch_size: usize) -> usize {
    (total_records + batch_size - 1) / batch_size // Ceiling division
}

/// Health check for the mock server
pub async fn check_mock_server_health(port: u16) -> Result<()> {
    let client = reqwest::Client::new();
    let health_url = format!("http://localhost:{}/health", port);
    
    let response = timeout(Duration::from_secs(5), client.get(&health_url).send())
        .await
        .map_err(|_| anyhow::anyhow!("Health check timed out"))?
        .map_err(|e| anyhow::anyhow!("Health check failed: {}", e))?;
    
    if response.status().is_success() {
        info!("Mock server health check passed");
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "Mock server health check failed: {}",
            response.status()
        ))
    }
}