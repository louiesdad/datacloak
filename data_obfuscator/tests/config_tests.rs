use data_obfuscator::config::{load_config, Rule};

#[test]
fn load_default_rules() {
    let cfg = load_config(
        "config/obfuscation_rules.json",
        "http://localhost",
        &Some("key".into()),
    )
    .unwrap();
    assert!(!cfg.rules.is_empty());
    assert_eq!(cfg.llm_endpoint, "http://localhost");
    assert_eq!(cfg.api_key, "key");
}
