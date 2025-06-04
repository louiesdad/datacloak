use serde::Deserialize;
use std::fs;
use thiserror::Error;
use config as config_rs;

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
    #[error("config error: {0}")]
    Config(#[from] config_rs::ConfigError),
}

pub fn load_config(path: &str, llm_endpoint: &str, api_key: &str) -> Result<AppConfig, ConfigError> {
    // Load rule definitions from the provided JSON file
    let content = fs::read_to_string(path)?;
    let rules: Vec<Rule> = serde_json::from_str(&content)?;

    // Build layered configuration for dynamic values
    let mut builder = config_rs::Config::builder();

    if let Ok(endpoint) = std::env::var("LLM_ENDPOINT") {
        builder = builder.set_override("llm_endpoint", endpoint)?;
    }
    if let Ok(key) = std::env::var("OPENAI_API_KEY") {
        builder = builder.set_override("api_key", key)?;
    }

    // CLI flags take precedence over environment variables
    builder = builder
        .set_override("llm_endpoint", llm_endpoint.to_string())?
        .set_override("api_key", api_key.to_string())?;

    let cfg = builder.build()?;

    Ok(AppConfig {
        rules,
        llm_endpoint: cfg.get::<String>("llm_endpoint")?,
        api_key: cfg.get::<String>("api_key")?,
    })
}
