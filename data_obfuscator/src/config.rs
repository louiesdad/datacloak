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

    if let Ok(endpoint) = std::env::var("LLM_ENDPOINT") {
        builder = builder.set_override("llm_endpoint", endpoint)?;
    }
    if let Ok(key) = std::env::var("OPENAI_API_KEY") {
        builder = builder.set_override("api_key", key)?;
    }

    // CLI flags take precedence
    builder = builder
        .set_override("llm_endpoint", llm_endpoint.to_string())?
        .set_override(
            "api_key",
            api_key.clone().unwrap_or_else(|| std::env::var("OPENAI_API_KEY").unwrap_or_default()),
        )?;

    let cfg = builder.build()?;

    Ok(AppConfig {
        rules,
        llm_endpoint: cfg.get::<String>("llm_endpoint")?,
        api_key: cfg.get::<String>("api_key")?,
    })
}