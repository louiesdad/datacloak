use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::{sleep, Instant};
use tracing::info;
use uuid::Uuid;
use warp::Filter;
use governor::{Quota, DefaultDirectRateLimiter};
use std::num::NonZeroU32;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone)]
pub enum ScenarioType {
    CustomerChurn,
    MedicalRecords,
    FinancialFraud,
    Generic,
}

impl ScenarioType {
    fn from_str(s: &str) -> Self {
        match s {
            "customer-churn" => ScenarioType::CustomerChurn,
            "medical-records" => ScenarioType::MedicalRecords,
            "financial-fraud" => ScenarioType::FinancialFraud,
            _ => ScenarioType::Generic,
        }
    }
}

pub struct MockLlmServer {
    scenario: ScenarioType,
    error_rate: f32,
    latency_range: (u64, u64), // min, max in milliseconds
    rate_limiter: Arc<DefaultDirectRateLimiter>,
}

impl MockLlmServer {
    pub fn new(scenario: Option<String>) -> Self {
        let scenario = scenario.map(|s| ScenarioType::from_str(&s)).unwrap_or(ScenarioType::Generic);
        
        // Rate limiter: 10 requests per second
        let quota = Quota::per_second(NonZeroU32::new(10).unwrap());
        let rate_limiter = Arc::new(governor::RateLimiter::direct(quota));
        
        Self {
            scenario,
            error_rate: 0.02, // 2% error rate
            latency_range: (50, 200), // 50-200ms latency
            rate_limiter,
        }
    }
    
    pub async fn process_chat_request(&self, request: ChatRequest) -> Result<ChatResponse, warp::Rejection> {
        let start_time = Instant::now();
        
        // Check rate limit
        if self.rate_limiter.check().is_err() {
            return Err(warp::reject::custom(RateLimitError));
        }
        
        // Simulate network latency
        let latency = rand::random::<u64>() % (self.latency_range.1 - self.latency_range.0) + self.latency_range.0;
        sleep(Duration::from_millis(latency)).await;
        
        // Simulate random errors
        if rand::random::<f32>() < self.error_rate {
            return Err(warp::reject::custom(SimulatedError));
        }
        
        // Generate response based on scenario
        let response_content = self.generate_response(&request).await;
        
        // Calculate usage
        let prompt_tokens = self.estimate_tokens(&request);
        let completion_tokens = self.estimate_tokens_for_text(&response_content);
        
        let response = ChatResponse {
            id: format!("chatcmpl-{}", Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp() as u64,
            model: request.model,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: response_content,
                },
                finish_reason: "stop".to_string(),
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        };
        
        info!("Generated response in {:?} for scenario {:?}", start_time.elapsed(), self.scenario);
        Ok(response)
    }
    
    async fn generate_response(&self, request: &ChatRequest) -> String {
        // Extract the user message content
        let user_content = request.messages
            .iter()
            .find(|m| m.role == "user")
            .map(|m| &m.content)
            .map_or("", |v| v);
        
        match self.scenario {
            ScenarioType::CustomerChurn => self.generate_churn_response(user_content),
            ScenarioType::MedicalRecords => self.generate_medical_response(user_content),
            ScenarioType::FinancialFraud => self.generate_fraud_response(user_content),
            ScenarioType::Generic => self.generate_generic_response(user_content),
        }
    }
    
    fn generate_churn_response(&self, content: &str) -> String {
        // Parse obfuscated customer data and generate churn predictions
        let predictions = self.extract_customer_records(content)
            .into_iter()
            .map(|record| {
                let churn_score = self.calculate_churn_score(&record);
                let confidence = 0.85 + (rand::random::<f32>() * 0.15); // 85-100% confidence
                
                serde_json::json!({
                    "customer_id": record.get("customer_id").unwrap_or(&"UNKNOWN".to_string()),
                    "churn_probability": churn_score,
                    "confidence": confidence,
                    "reasoning": self.generate_churn_reasoning(&record, churn_score),
                    "risk_factors": self.identify_risk_factors(&record)
                })
            })
            .collect::<Vec<_>>();
        
        serde_json::json!({
            "predictions": predictions,
            "analysis_type": "churn_prediction",
            "model_version": "mock-v1.0"
        }).to_string()
    }
    
    fn generate_medical_response(&self, _content: &str) -> String {
        serde_json::json!({
            "analysis": "Anonymized health trends analysis",
            "findings": [
                "Diabetes prevalence increased by 12% in Q3",
                "Hypertension cases show seasonal pattern",
                "Mental health visits up 25% year-over-year"
            ],
            "patient_count": 1247,
            "analysis_type": "health_trends"
        }).to_string()
    }
    
    fn generate_fraud_response(&self, _content: &str) -> String {
        serde_json::json!({
            "fraud_alerts": [
                {
                    "transaction_id": "[TXN-001]",
                    "fraud_probability": 0.89,
                    "indicators": ["unusual_location", "high_amount", "off_hours"]
                },
                {
                    "transaction_id": "[TXN-002]", 
                    "fraud_probability": 0.23,
                    "indicators": ["normal_pattern"]
                }
            ],
            "analysis_type": "fraud_detection"
        }).to_string()
    }
    
    fn generate_generic_response(&self, _content: &str) -> String {
        "I have analyzed the obfuscated data and provided insights while maintaining privacy.".to_string()
    }
    
    fn extract_customer_records(&self, content: &str) -> Vec<HashMap<String, String>> {
        // Simple parsing of JSON-like obfuscated data
        // In reality, this would be more sophisticated
        let mut records = Vec::new();
        
        // Look for customer ID patterns and extract surrounding data
        if content.contains("CUST-") {
            // Extract customer records from the content
            for line in content.lines() {
                if line.contains("CUST-") {
                    let mut record = HashMap::new();
                    
                    // Extract customer ID
                    if let Some(start) = line.find("CUST-") {
                        if let Some(end) = line[start..].find(',').or_else(|| line[start..].find(' ')) {
                            let customer_id = &line[start..start + end];
                            record.insert("customer_id".to_string(), customer_id.to_string());
                        }
                    }
                    
                    // Look for support tickets
                    if line.contains("support_tickets") {
                        if let Some(tickets) = self.extract_number_after("support_tickets", line) {
                            record.insert("support_tickets".to_string(), tickets.to_string());
                        }
                    }
                    
                    // Look for payment delays
                    if line.contains("payment_delays") {
                        if let Some(delays) = self.extract_number_after("payment_delays", line) {
                            record.insert("payment_delays".to_string(), delays.to_string());
                        }
                    }
                    
                    records.push(record);
                }
            }
        }
        
        // If no structured data found, create some mock records
        if records.is_empty() {
            for i in 1..=3 {
                let mut record = HashMap::new();
                record.insert("customer_id".to_string(), format!("CUST-{:06}", i));
                record.insert("support_tickets".to_string(), (rand::random::<u8>() % 6).to_string());
                record.insert("payment_delays".to_string(), (rand::random::<u8>() % 4).to_string());
                records.push(record);
            }
        }
        
        records
    }
    
    fn extract_number_after(&self, key: &str, text: &str) -> Option<u32> {
        if let Some(start) = text.find(key) {
            let remaining = &text[start + key.len()..];
            // Look for ": number" or similar patterns
            for char in remaining.chars() {
                if char.is_ascii_digit() {
                    let num_str = remaining.chars()
                        .skip_while(|c| !c.is_ascii_digit())
                        .take_while(|c| c.is_ascii_digit())
                        .collect::<String>();
                    return num_str.parse().ok();
                }
            }
        }
        None
    }
    
    fn calculate_churn_score(&self, record: &HashMap<String, String>) -> f32 {
        let mut score = 0.3; // Base score
        
        // Support tickets increase churn risk
        if let Some(tickets) = record.get("support_tickets").and_then(|s| s.parse::<u32>().ok()) {
            score += (tickets as f32) * 0.1;
        }
        
        // Payment delays increase churn risk
        if let Some(delays) = record.get("payment_delays").and_then(|s| s.parse::<u32>().ok()) {
            score += (delays as f32) * 0.15;
        }
        
        // Add some deterministic randomness based on customer ID
        if let Some(customer_id) = record.get("customer_id") {
            let hash = customer_id.chars().map(|c| c as u32).sum::<u32>();
            let random_factor = (hash % 100) as f32 / 100.0 * 0.3; // 0-30% additional
            score += random_factor;
        }
        
        score.min(1.0) // Cap at 100%
    }
    
    fn generate_churn_reasoning(&self, record: &HashMap<String, String>, score: f32) -> String {
        let mut reasons = Vec::new();
        
        if let Some(tickets) = record.get("support_tickets").and_then(|s| s.parse::<u32>().ok()) {
            if tickets > 3 {
                reasons.push(format!("High support contact ({} tickets)", tickets));
            }
        }
        
        if let Some(delays) = record.get("payment_delays").and_then(|s| s.parse::<u32>().ok()) {
            if delays > 0 {
                reasons.push(format!("Payment delays ({})", delays));
            }
        }
        
        if score > 0.7 {
            reasons.push("Multiple risk indicators".to_string());
        } else if score < 0.3 {
            reasons.push("Good customer profile".to_string());
        }
        
        if reasons.is_empty() {
            "Standard risk assessment".to_string()
        } else {
            reasons.join(", ")
        }
    }
    
    fn identify_risk_factors(&self, record: &HashMap<String, String>) -> Vec<String> {
        let mut factors = Vec::new();
        
        if let Some(tickets) = record.get("support_tickets").and_then(|s| s.parse::<u32>().ok()) {
            if tickets > 3 {
                factors.push("high_support_contact".to_string());
            }
        }
        
        if let Some(delays) = record.get("payment_delays").and_then(|s| s.parse::<u32>().ok()) {
            if delays > 0 {
                factors.push("payment_delays".to_string());
            }
        }
        
        if factors.is_empty() {
            factors.push("normal_profile".to_string());
        }
        
        factors
    }
    
    fn estimate_tokens(&self, request: &ChatRequest) -> u32 {
        let total_chars: usize = request.messages.iter()
            .map(|m| m.content.len())
            .sum();
        (total_chars / 4) as u32 // Rough estimate: 4 chars per token
    }
    
    fn estimate_tokens_for_text(&self, text: &str) -> u32 {
        (text.len() / 4) as u32
    }
}

pub async fn start_mock_server(port: u16, scenario: Option<String>) -> Result<()> {
    info!("Starting mock LLM server on port {} with scenario {:?}", port, scenario);
    
    let mock_server = Arc::new(MockLlmServer::new(scenario));
    
    let chat_route = warp::path("v1")
        .and(warp::path("chat"))
        .and(warp::path("completions"))
        .and(warp::post())
        .and(warp::header::<String>("authorization"))
        .and(warp::body::json())
        .and(warp::any().map(move || mock_server.clone()))
        .and_then(handle_chat_request);
    
    let health_route = warp::path("health")
        .and(warp::get())
        .map(|| warp::reply::with_status("OK", warp::http::StatusCode::OK));
    
    let routes = chat_route
        .or(health_route)
        .recover(handle_rejection);
    
    info!("Mock LLM server ready at http://localhost:{}", port);
    warp::serve(routes)
        .run(([127, 0, 0, 1], port))
        .await;
    
    Ok(())
}

async fn handle_chat_request(
    _auth: String,
    request: ChatRequest,
    server: Arc<MockLlmServer>,
) -> Result<impl warp::Reply, warp::Rejection> {
    let response = server.process_chat_request(request).await?;
    Ok(warp::reply::json(&response))
}

async fn handle_rejection(err: warp::Rejection) -> Result<impl warp::Reply, std::convert::Infallible> {
    if err.find::<RateLimitError>().is_some() {
        let response = serde_json::json!({
            "error": {
                "type": "rate_limit_exceeded",
                "message": "Rate limit exceeded. Please try again later."
            }
        });
        Ok(warp::reply::with_status(
            warp::reply::json(&response),
            warp::http::StatusCode::TOO_MANY_REQUESTS,
        ))
    } else if err.find::<SimulatedError>().is_some() {
        let response = serde_json::json!({
            "error": {
                "type": "server_error", 
                "message": "Simulated server error for testing"
            }
        });
        Ok(warp::reply::with_status(
            warp::reply::json(&response),
            warp::http::StatusCode::INTERNAL_SERVER_ERROR,
        ))
    } else {
        let response = serde_json::json!({
            "error": {
                "type": "unknown_error",
                "message": "An unknown error occurred"
            }
        });
        Ok(warp::reply::with_status(
            warp::reply::json(&response),
            warp::http::StatusCode::INTERNAL_SERVER_ERROR,
        ))
    }
}

#[derive(Debug)]
struct RateLimitError;
impl warp::reject::Reject for RateLimitError {}

#[derive(Debug)]
struct SimulatedError;
impl warp::reject::Reject for SimulatedError {}