use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use datacloak_core::{Pattern, PatternType};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestScenario {
    pub name: String,
    pub description: String,
    pub data_file: String,
    pub expected_patterns: Vec<PatternType>,
    pub llm_context: String,
    pub expected_results: ExpectedResults,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedResults {
    pub min_records: usize,
    pub max_records: usize,
    pub patterns_detected: Vec<PatternType>,
    pub churn_analysis: Option<ChurnExpectations>,
    pub fraud_analysis: Option<FraudExpectations>,
    pub medical_analysis: Option<MedicalExpectations>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChurnExpectations {
    pub expected_high_risk_count: Option<usize>,
    pub expected_avg_churn_range: (f32, f32), // min, max
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FraudExpectations {
    pub expected_fraud_alerts: usize,
    pub expected_high_risk_transactions: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MedicalExpectations {
    pub expected_patient_count: usize,
    pub expected_findings: usize,
}

impl TestScenario {
    /// Load a test scenario from file
    pub async fn load(scenario_name: &str) -> Result<Self> {
        let file_path = format!("test-scenarios/{}/config.yaml", scenario_name);
        let content = tokio::fs::read_to_string(&file_path).await?;
        let scenario: TestScenario = serde_yaml::from_str(&content)?;
        Ok(scenario)
    }
    
    /// Get all available scenarios
    pub async fn list_available() -> Result<Vec<String>> {
        let mut scenarios = Vec::new();
        let scenarios_dir = PathBuf::from("test-scenarios");
        
        if scenarios_dir.exists() {
            let mut entries = tokio::fs::read_dir(scenarios_dir).await?;
            while let Some(entry) = entries.next_entry().await? {
                if entry.file_type().await?.is_dir() {
                    if let Some(name) = entry.file_name().to_str() {
                        scenarios.push(name.to_string());
                    }
                }
            }
        }
        
        Ok(scenarios)
    }
    
    /// Get the data file path for this scenario
    pub fn data_file_path(&self) -> PathBuf {
        PathBuf::from("test-scenarios")
            .join(&self.name)
            .join(&self.data_file)
    }
    
    /// Convert scenario patterns to DataCloak patterns
    pub fn to_datacloak_patterns(&self) -> Vec<Pattern> {
        use datacloak_core::patterns::PatternSet;
        let default_set = PatternSet::default_pii();
        
        self.expected_patterns.iter().filter_map(|pattern_type| {
            // Find matching pattern from default set
            default_set.as_slice().iter()
                .find(|p| p.pattern_type == *pattern_type)
                .cloned()
        }).collect()
    }
}

/// Create all default test scenarios
pub async fn create_default_scenarios() -> Result<()> {
    // Create scenarios directory
    tokio::fs::create_dir_all("test-scenarios").await?;
    
    // Customer Churn Scenario
    create_customer_churn_scenario().await?;
    
    // Medical Records Scenario
    create_medical_records_scenario().await?;
    
    // Financial Fraud Scenario
    create_financial_fraud_scenario().await?;
    
    Ok(())
}

async fn create_customer_churn_scenario() -> Result<()> {
    let scenario_dir = PathBuf::from("test-scenarios/customer-churn");
    tokio::fs::create_dir_all(&scenario_dir).await?;
    
    // Create scenario config
    let scenario = TestScenario {
        name: "customer-churn".to_string(),
        description: "Detect churn risk in customer data".to_string(),
        data_file: "customer_data.csv".to_string(),
        expected_patterns: vec![
            PatternType::Email,
            PatternType::Phone,
            PatternType::SSN,
        ],
        llm_context: "Analyze customer behavior for churn risk based on support tickets, payment history, and usage patterns.".to_string(),
        expected_results: ExpectedResults {
            min_records: 95,
            max_records: 105,
            patterns_detected: vec![PatternType::Email, PatternType::Phone],
            churn_analysis: Some(ChurnExpectations {
                expected_high_risk_count: Some(5),
                expected_avg_churn_range: (0.3, 0.7),
            }),
            fraud_analysis: None,
            medical_analysis: None,
        },
    };
    
    let config_content = serde_yaml::to_string(&scenario)?;
    tokio::fs::write(scenario_dir.join("config.yaml"), config_content).await?;
    
    // Create sample data
    let csv_content = create_customer_churn_data();
    tokio::fs::write(scenario_dir.join("customer_data.csv"), csv_content).await?;
    
    Ok(())
}

async fn create_medical_records_scenario() -> Result<()> {
    let scenario_dir = PathBuf::from("test-scenarios/medical-records");
    tokio::fs::create_dir_all(&scenario_dir).await?;
    
    let scenario = TestScenario {
        name: "medical-records".to_string(),
        description: "Obfuscate patient data for research".to_string(),
        data_file: "patient_records.csv".to_string(),
        expected_patterns: vec![
            PatternType::SSN,
            PatternType::Phone,
            PatternType::Email,
        ],
        llm_context: "Identify health trends and patterns in anonymized patient data without exposing PII.".to_string(),
        expected_results: ExpectedResults {
            min_records: 45,
            max_records: 55,
            patterns_detected: vec![PatternType::SSN, PatternType::Phone],
            churn_analysis: None,
            fraud_analysis: None,
            medical_analysis: Some(MedicalExpectations {
                expected_patient_count: 50,
                expected_findings: 3,
            }),
        },
    };
    
    let config_content = serde_yaml::to_string(&scenario)?;
    tokio::fs::write(scenario_dir.join("config.yaml"), config_content).await?;
    
    let csv_content = create_medical_records_data();
    tokio::fs::write(scenario_dir.join("patient_records.csv"), csv_content).await?;
    
    Ok(())
}

async fn create_financial_fraud_scenario() -> Result<()> {
    let scenario_dir = PathBuf::from("test-scenarios/financial-fraud");
    tokio::fs::create_dir_all(&scenario_dir).await?;
    
    let scenario = TestScenario {
        name: "financial-fraud".to_string(),
        description: "Detect fraudulent transactions".to_string(),
        data_file: "transactions.csv".to_string(),
        expected_patterns: vec![
            PatternType::CreditCard,
            PatternType::SSN,
            PatternType::Phone,
        ],
        llm_context: "Analyze transaction patterns for fraud indicators while protecting customer PII.".to_string(),
        expected_results: ExpectedResults {
            min_records: 195,
            max_records: 205,
            patterns_detected: vec![PatternType::CreditCard, PatternType::Phone],
            churn_analysis: None,
            fraud_analysis: Some(FraudExpectations {
                expected_fraud_alerts: 2,
                expected_high_risk_transactions: 5,
            }),
            medical_analysis: None,
        },
    };
    
    let config_content = serde_yaml::to_string(&scenario)?;
    tokio::fs::write(scenario_dir.join("config.yaml"), config_content).await?;
    
    let csv_content = create_financial_fraud_data();
    tokio::fs::write(scenario_dir.join("transactions.csv"), csv_content).await?;
    
    Ok(())
}

fn create_customer_churn_data() -> String {
    let mut csv = "customer_id,email,phone,last_purchase,support_tickets,payment_delays,monthly_usage\n".to_string();
    
    // Generate 100 customer records
    for i in 1..=100 {
        let email = format!("customer{}@example.com", i);
        let phone = format!("555-{:03}-{:04}", (i % 900) + 100, i % 10000);
        let last_purchase = format!("2024-{:02}-{:02}", (i % 12) + 1, (i % 28) + 1);
        let support_tickets = i % 7; // 0-6 support tickets
        let payment_delays = if i % 4 == 0 { i % 4 } else { 0 }; // Some customers have payment delays
        let monthly_usage = 50 + (i % 200); // Usage between 50-250
        
        csv.push_str(&format!(
            "CUST-{:06},{},{},{},{},{},{}\n",
            i, email, phone, last_purchase, support_tickets, payment_delays, monthly_usage
        ));
    }
    
    csv
}

fn create_medical_records_data() -> String {
    let mut csv = "patient_id,ssn,phone,date_of_birth,diagnosis_codes,visit_count,last_visit\n".to_string();
    
    let diagnoses = [
        "E11.9", "I10", "Z00.00", "J44.1", "M79.3", "F32.9", "K21.9", "I25.10"
    ];
    
    for i in 1..=50 {
        let patient_id = format!("PAT-{:06}", i);
        let ssn = format!("{:03}-{:02}-{:04}", 
            (i % 900) + 100, 
            (i % 99) + 1, 
            (i % 9000) + 1000
        );
        let phone = format!("555-{:03}-{:04}", (i % 900) + 100, i % 10000);
        let dob = format!("{}-{:02}-{:02}", 
            1950 + (i % 50), 
            (i % 12) + 1, 
            (i % 28) + 1
        );
        let diagnosis = diagnoses[i as usize % diagnoses.len()];
        let visit_count = 1 + (i % 12);
        let last_visit = format!("2024-{:02}-{:02}", (i % 12) + 1, (i % 28) + 1);
        
        csv.push_str(&format!(
            "{},{},{},{},{},{},{}\n",
            patient_id, ssn, phone, dob, diagnosis, visit_count, last_visit
        ));
    }
    
    csv
}

fn create_financial_fraud_data() -> String {
    let mut csv = "transaction_id,credit_card,ssn,phone,amount,merchant,timestamp,location\n".to_string();
    
    let merchants = [
        "Amazon", "Walmart", "Target", "Starbucks", "Shell", "McDonalds", "Home Depot", "Best Buy"
    ];
    
    let locations = [
        "New York,NY", "Los Angeles,CA", "Chicago,IL", "Houston,TX", "Phoenix,AZ", "Philadelphia,PA"
    ];
    
    for i in 1..=200 {
        let transaction_id = format!("TXN-{:08}", i);
        let credit_card = format!("{:04}-{:04}-{:04}-{:04}", 
            4000 + (i % 1000), 
            (i % 9000) + 1000,
            (i % 9000) + 1000,
            (i % 9000) + 1000
        );
        let ssn = format!("{:03}-{:02}-{:04}", 
            (i % 900) + 100, 
            (i % 99) + 1, 
            (i % 9000) + 1000
        );
        let phone = format!("555-{:03}-{:04}", (i % 900) + 100, i % 10000);
        
        // Most transactions are normal, some are suspicious
        let amount = if i % 50 == 0 {
            // Suspicious high amount
            format!("{:.2}", 5000.0 + (i as f32 * 10.0))
        } else {
            // Normal amount
            format!("{:.2}", 10.0 + (i as f32 % 500.0))
        };
        
        let merchant = merchants[i as usize % merchants.len()];
        let timestamp = format!("2024-{:02}-{:02}T{:02}:{:02}:00Z", 
            (i % 12) + 1, 
            (i % 28) + 1,
            i % 24,
            (i * 7) % 60
        );
        let location = locations[i as usize % locations.len()];
        
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{}\n",
            transaction_id, credit_card, ssn, phone, amount, merchant, timestamp, location
        ));
    }
    
    csv
}