use std::fs;
use serde::{Deserialize, Serialize};
use config as config_rs;
use thiserror::Error;

#[derive(Debug, Serialize, Deserialize)]
pub struct Rule {
    pub pattern: String,
    pub label: String,
}

#[derive(Serialize, Deserialize)]
pub struct AppConfig {
    pub rules: Vec<Rule>,
    pub llm_endpoint: String,
    #[serde(skip_serializing_if = "String::is_empty", default)]
    pub api_key: String,
}

impl std::fmt::Debug for AppConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppConfig")
            .field("rules", &self.rules)
            .field("llm_endpoint", &self.llm_endpoint)
            .field("api_key", &"***")
            .finish()
    }
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse error: {0}")]
    Parse(#[from] serde_json::Error),
    #[error("config error: {0}")]
    Config(#[from] config_rs::ConfigError),
}

pub fn load_config(
    path: &str,
    llm_endpoint: &str,
    api_key: &Option<String>,
) -> Result<AppConfig, ConfigError> {
    // Load rules from JSON file
    let content = fs::read_to_string(path)?;
    let rules: Vec<Rule> = serde_json::from_str(&content)?;

    // Build layered config for endpoint/key, using env and CLI overrides
    let mut builder = config_rs::Config::builder();

    // Set defaults
    builder = builder
        .set_default("llm_endpoint", "https://api.openai.com/v1/chat/completions")?
        .set_default("api_key", "")?;

    // Environment variables override defaults
    if let Ok(endpoint) = std::env::var("LLM_ENDPOINT") {
        builder = builder.set_override("llm_endpoint", endpoint)?;
    }
    if let Ok(key) = std::env::var("OPENAI_API_KEY") {
        builder = builder.set_override("api_key", key)?;
    }

    // CLI flags take precedence (only if non-empty)
    if !llm_endpoint.is_empty() {
        builder = builder.set_override("llm_endpoint", llm_endpoint.to_string())?;
    }
    if let Some(ref key) = api_key {
        if !key.is_empty() {
            builder = builder.set_override("api_key", key.clone())?;
        }
    }

    let cfg = builder.build()?;

    Ok(AppConfig {
        rules,
        llm_endpoint: cfg.get::<String>("llm_endpoint")?,
        api_key: cfg.get::<String>("api_key")?,
    })
}