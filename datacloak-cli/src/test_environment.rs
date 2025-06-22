use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use tokio::time::{sleep, Duration};

/// Test environment for E2E testing
pub struct TestEnvironment {
    pub id: String,
    pub temp_dir: PathBuf,
    pub services: Arc<Mutex<HashMap<String, ServiceHandle>>>,
    pub config: TestConfig,
}

#[derive(Clone, Debug)]
pub struct TestConfig {
    pub mock_llm_port: u16,
    pub api_port: u16,
    pub enable_monitoring: bool,
    pub log_level: String,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            mock_llm_port: 3100,
            api_port: 8080,
            enable_monitoring: true,
            log_level: "info".to_string(),
        }
    }
}

pub struct ServiceHandle {
    pub name: String,
    pub process: Option<Child>,
    pub port: Option<u16>,
    pub health_endpoint: Option<String>,
}

impl TestEnvironment {
    pub async fn new() -> Self {
        let id = Uuid::new_v4().to_string();
        let temp_dir = std::env::temp_dir().join(format!("datacloak_e2e_{}", id));
        std::fs::create_dir_all(&temp_dir).unwrap();
        
        Self {
            id,
            temp_dir,
            services: Arc::new(Mutex::new(HashMap::new())),
            config: TestConfig::default(),
        }
    }
    
    pub async fn with_config(config: TestConfig) -> Self {
        let mut env = Self::new().await;
        env.config = config;
        env
    }
    
    pub async fn start_services(&self, services: &[&str]) -> Result<()> {
        for service in services {
            self.start_service(service).await?;
        }
        
        // Wait for all services to be healthy
        for service in services {
            self.wait_for_service_health(service).await?;
        }
        
        Ok(())
    }
    
    pub async fn start_service(&self, service_name: &str) -> Result<()> {
        let mut services = self.services.lock().await;
        
        match service_name {
            "mock-llm" => {
                let port = self.config.mock_llm_port;
                let handle = ServiceHandle {
                    name: service_name.to_string(),
                    process: None, // Will spawn in a task
                    port: Some(port),
                    health_endpoint: Some(format!("http://localhost:{}/health", port)),
                };
                services.insert(service_name.to_string(), handle);
                
                // Start mock LLM server in background
                let port = self.config.mock_llm_port;
                tokio::spawn(async move {
                    crate::mock_llm::start_mock_server(port, Some("multi-field".to_string())).await.ok();
                });
            }
            "datacloak-api" => {
                // In real implementation, would start the API service
                let handle = ServiceHandle {
                    name: service_name.to_string(),
                    process: None,
                    port: Some(self.config.api_port),
                    health_endpoint: Some(format!("http://localhost:{}/health", self.config.api_port)),
                };
                services.insert(service_name.to_string(), handle);
            }
            _ => {
                // Generic service (postgres, redis, etc)
                let handle = ServiceHandle {
                    name: service_name.to_string(),
                    process: None,
                    port: None,
                    health_endpoint: None,
                };
                services.insert(service_name.to_string(), handle);
            }
        }
        
        Ok(())
    }
    
    pub async fn wait_for_service_health(&self, service_name: &str) -> Result<()> {
        let services = self.services.lock().await;
        let service = services.get(service_name)
            .ok_or_else(|| anyhow::anyhow!("Service {} not found", service_name))?;
        
        if let Some(health_endpoint) = &service.health_endpoint {
            let client = reqwest::Client::new();
            let max_retries = 30;
            let mut retries = 0;
            
            loop {
                match client.get(health_endpoint).send().await {
                    Ok(resp) if resp.status().is_success() => {
                        println!("Service {} is healthy", service_name);
                        return Ok(());
                    }
                    _ => {
                        retries += 1;
                        if retries >= max_retries {
                            return Err(anyhow::anyhow!("Service {} failed to become healthy", service_name));
                        }
                        sleep(Duration::from_secs(1)).await;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    pub async fn kill_service(&self, service_name: &str) -> Result<()> {
        let mut services = self.services.lock().await;
        if let Some(service) = services.remove(service_name) {
            if let Some(mut process) = service.process {
                process.kill()?;
            }
        }
        Ok(())
    }
    
    pub async fn run_cli(&self, args: &[&str]) -> String {
        let output = Command::new("cargo")
            .args(&["run", "--bin", "datacloak-cli", "--"])
            .args(args)
            .env("DATACLOAK_TEST_MODE", "1")
            .output()
            .expect("Failed to run CLI");
        
        String::from_utf8_lossy(&output.stdout).to_string()
    }
    
    pub async fn start_analysis(&self, file: &str, columns: Vec<&str>) -> String {
        let run_id = Uuid::new_v4().to_string();
        
        // In real implementation, would start analysis via API
        // For now, run CLI command
        let columns_str = columns.join(",");
        let args = vec![
            "analyze",
            "--file", file,
            "--columns", &columns_str,
            "--output", "json",
            "--mock-llm",
        ];
        
        self.run_cli(&args).await;
        run_id
    }
    
    pub async fn wait_for_progress(&self, _run_id: &str, target_percent: u32) -> Result<()> {
        // Simulate waiting for progress
        let wait_time = (target_percent as u64) * 100; // 100ms per percent
        sleep(Duration::from_millis(wait_time)).await;
        Ok(())
    }
    
    pub async fn get_analysis_status(&self, run_id: &str) -> AnalysisStatus {
        // In real implementation, would query API
        AnalysisStatus {
            id: run_id.to_string(),
            state: "resuming".to_string(),
            progress: 25,
            error: None,
        }
    }
    
    pub async fn wait_for_completion(&self, _run_id: &str) -> Result<()> {
        // Simulate completion
        sleep(Duration::from_secs(2)).await;
        Ok(())
    }
    
    pub async fn get_results(&self, run_id: &str) -> AnalysisResults {
        // In real implementation, would fetch from API
        AnalysisResults {
            id: run_id.to_string(),
            total_rows: 1000,
            processed_rows: 1000,
            results: vec![],
        }
    }
    
    pub async fn get_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        metrics.insert("datacloak_columns_analyzed".to_string(), 3.0);
        metrics.insert("max_memory_gb".to_string(), 2.5);
        metrics.insert("avg_cpu_percent".to_string(), 65.0);
        metrics
    }
    
    pub async fn start_services_all(&self) -> Result<()> {
        self.start_services(&["mock-llm", "datacloak-api"]).await
    }
}

impl Drop for TestEnvironment {
    fn drop(&mut self) {
        // Clean up temp directory
        if self.temp_dir.exists() {
            std::fs::remove_dir_all(&self.temp_dir).ok();
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisStatus {
    pub id: String,
    pub state: String,
    pub progress: u32,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AnalysisResults {
    pub id: String,
    pub total_rows: usize,
    pub processed_rows: usize,
    pub results: Vec<serde_json::Value>,
}

/// Extract file ID from CLI output
pub fn extract_file_id(output: &str) -> String {
    // Look for pattern like "File ID: xxx"
    if let Some(line) = output.lines().find(|l| l.contains("File ID:")) {
        line.split(':').nth(1).unwrap_or("").trim().to_string()
    } else {
        Uuid::new_v4().to_string()
    }
}