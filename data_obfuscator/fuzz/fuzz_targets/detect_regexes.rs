#![no_main]

use libfuzzer_sys::fuzz_target;
use data_obfuscator::obfuscator::Obfuscator;
use data_obfuscator::secure_obfuscator::SecureObfuscator;
use data_obfuscator::config::Rule;

fuzz_target!(|data: &[u8]| {
    // Convert bytes to string, handling invalid UTF-8 gracefully
    let input = String::from_utf8_lossy(data);
    
    // Skip empty or extremely long inputs to focus fuzzing efforts
    if input.is_empty() || input.len() > 100_000 {
        return;
    }
    
    // Test rules based on OWASP ReDoS corpus patterns
    let test_rules = vec![
        Rule {
            pattern: r"\b\d{3}-\d{2}-\d{4}\b".to_string(),
            label: "SSN".to_string(),
        },
        Rule {
            pattern: r"\(\d{3}\) \d{3}-\d{4}".to_string(),
            label: "PHONE".to_string(),
        },
        Rule {
            pattern: r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b".to_string(),
            label: "EMAIL".to_string(),
        },
        Rule {
            pattern: r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b".to_string(),
            label: "CREDIT_CARD".to_string(),
        },
    ];
    
    // Test both obfuscator implementations
    if let Ok(mut obfuscator) = Obfuscator::new(&test_rules) {
        let _ = obfuscator.obfuscate_text(&input);
    }
    
    if let Ok(mut secure_obfuscator) = SecureObfuscator::new(&test_rules) {
        let _ = secure_obfuscator.obfuscate_text(&input);
    }
    
    // Test with potential ReDoS patterns from OWASP corpus
    let redos_patterns = vec![
        Rule {
            pattern: r"^(a+)+$".to_string(),
            label: "REDOS_TEST".to_string(),
        },
        Rule {
            pattern: r"(a|a)*".to_string(),
            label: "REDOS_TEST2".to_string(),
        },
        Rule {
            pattern: r"([a-zA-Z]+)*".to_string(),
            label: "REDOS_TEST3".to_string(),
        },
    ];
    
    // Only test ReDoS patterns with small inputs to prevent infinite loops
    if input.len() <= 1000 {
        if let Ok(mut redos_obfuscator) = Obfuscator::new(&redos_patterns) {
            // Use timeout mechanism or bounded execution
            let _ = redos_obfuscator.obfuscate_text(&input);
        }
    }
});