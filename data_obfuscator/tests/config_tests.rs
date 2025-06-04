use data_obfuscator::config::{load_config, Rule};

#[test]
fn load_default_rules() {
    let cfg = load_config("config/obfuscation_rules.json", "http://localhost", "key").unwrap();
    assert!(!cfg.rules.is_empty());
    assert_eq!(cfg.llm_endpoint, "http://localhost");
    assert_eq!(cfg.api_key, "key");
}

#[test]
fn env_overrides_defaults() {
    std::env::set_var("LLM_ENDPOINT", "http://env");
    std::env::set_var("OPENAI_API_KEY", "envkey");
    let cfg = load_config("config/obfuscation_rules.json", "", "").unwrap();
    assert_eq!(cfg.llm_endpoint, "http://env");
    assert_eq!(cfg.api_key, "envkey");
    std::env::remove_var("LLM_ENDPOINT");
    std::env::remove_var("OPENAI_API_KEY");
}
