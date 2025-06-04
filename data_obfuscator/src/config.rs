use serde::Deserialize;
use std::fs;
use thiserror::Error;

#[derive(Debug, Deserialize, Clone)]
pub struct Rule {
    pub pattern: String,
    pub label: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct AppConfig {
    pub rules: Vec<Rule>,
    pub llm_endpoint: String,
    pub api_key: String,
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

pub fn load_config(path: &str, llm_endpoint: &str, api_key: &str) -> Result<AppConfig, ConfigError> {
    let content = fs::read_to_string(path)?;
    let rules: Vec<Rule> = serde_json::from_str(&content)?;
    Ok(AppConfig {
        rules,
        llm_endpoint: llm_endpoint.to_string(),
        api_key: api_key.to_string(),
    })
}
