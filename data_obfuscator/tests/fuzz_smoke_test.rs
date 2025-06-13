// Smoke test for fuzz-like scenarios with stable Rust
use data_obfuscator::obfuscator::Obfuscator;
use data_obfuscator::secure_obfuscator::SecureObfuscator;
use data_obfuscator::config::Rule;
use std::time::{Duration, Instant};

#[test]
fn test_potential_redos_patterns() {
    let redos_rules = vec![
        Rule {
            pattern: r"\b\d{3}-\d{2}-\d{4}\b".to_string(),
            label: "SSN".to_string(),
        },
        Rule {
            pattern: r"\(\d{3}\) \d{3}-\d{4}".to_string(),
            label: "PHONE".to_string(),
        },
    ];
    
    // Test inputs that could cause ReDoS
    let test_inputs = vec![
        "a".repeat(10000),
        "123-45".repeat(5000),
        "(555) 123".repeat(3000),
        format!("{}X", "a".repeat(50000)),
        format!("{}@{}.com", "test".repeat(10000), "domain".repeat(5000)),
    ];
    
    for input in &test_inputs {
        let start = Instant::now();
        
        if let Ok(mut obfuscator) = Obfuscator::new(&redos_rules) {
            let _ = obfuscator.obfuscate_text(input);
        }
        
        let duration = start.elapsed();
        
        // Ensure processing completes quickly (ReDoS protection)
        assert!(
            duration < Duration::from_millis(100),
            "Processing took {}ms for input length {}, should be <100ms",
            duration.as_millis(),
            input.len()
        );
    }
}

#[test]
fn test_secure_obfuscator_with_malformed_input() {
    let rules = vec![
        Rule {
            pattern: r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b".to_string(),
            label: "EMAIL".to_string(),
        },
    ];
    
    let malformed_inputs = vec![
        "invalid@".to_string(), 
        "@invalid.com".to_string(),
        "test@.com".to_string(),
        "test@com.".to_string(),
        "test@@test.com".to_string(),
        format!("{}@test.com", "a".repeat(100000)),
        format!("test@{}.com", "a".repeat(100000)),
    ];
    
    for input in &malformed_inputs {
        let start = Instant::now();
        
        if let Ok(mut obfuscator) = SecureObfuscator::new(&rules) {
            let result = obfuscator.obfuscate_text(input);
            // SecureObfuscator uses validator which might detect some patterns as valid
            // Just ensure it doesn't crash and produces output
            assert!(!result.is_empty());
        }
        
        let duration = start.elapsed();
        assert!(
            duration < Duration::from_millis(50),
            "Secure obfuscation took {}ms for input length {}, should be <50ms",
            duration.as_millis(),
            input.len()
        );
    }
}

#[test]
fn test_large_input_handling() {
    let rules = vec![
        Rule {
            pattern: r"\b\d{3}-\d{2}-\d{4}\b".to_string(),
            label: "SSN".to_string(),
        },
    ];
    
    // Test with very large input (should be skipped for ReDoS protection)
    let large_input = "a".repeat(200_000);
    
    let start = Instant::now();
    
    if let Ok(mut obfuscator) = Obfuscator::new(&rules) {
        let result = obfuscator.obfuscate_text(&large_input);
        // Should return unchanged due to size limit
        assert_eq!(large_input, result);
    }
    
    let duration = start.elapsed();
    assert!(
        duration < Duration::from_millis(50),
        "Large input processing took {}ms, should be <50ms due to early return",
        duration.as_millis()
    );
}

#[test]
fn test_corpus_patterns() {
    let rules = vec![
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
    ];
    
    // Test patterns from fuzz corpus
    let corpus_inputs = vec![
        "123-45-6789",
        "user@example.com", 
        "(555) 123-4567",
        "Customer: john@test.com SSN: 123-45-6789 Phone: (555) 123-4567",
        "{\"user\": \"test@example.com\", \"ssn\": \"123-45-6789\", \"phone\": \"(555) 123-4567\"}",
        "aaaaaaaaaaaaaaaaaaaaX",
        "aaaaabbbbbX",
    ];
    
    for input in &corpus_inputs {
        let start = Instant::now();
        
        // Test both obfuscator implementations
        if let Ok(mut obfuscator) = Obfuscator::new(&rules) {
            let result1 = obfuscator.obfuscate_text(input);
            
            if let Ok(mut secure_obfuscator) = SecureObfuscator::new(&rules) {
                let result2 = secure_obfuscator.obfuscate_text(input);
                
                // Both should process without crashing
                assert!(!result1.is_empty());
                assert!(!result2.is_empty());
            }
        }
        
        let duration = start.elapsed();
        assert!(
            duration < Duration::from_millis(50),
            "Corpus input '{}' took {}ms, should be <50ms",
            input,
            duration.as_millis()
        );
    }
}