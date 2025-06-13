use data_obfuscator::secure_obfuscator::SecureObfuscator;
use data_obfuscator::config::Rule;
use std::time::Instant;

#[test]
fn debug_performance_email_only() {
    // Test with email patterns only (no SSN regex)
    let rules = vec![
        Rule {
            pattern: r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b".to_string(),
            label: "EMAIL".to_string(),
        },
    ];
    
    // The problematic input
    let redos_input = format!("a@{}", "a".repeat(50000));
    
    println!("Input length: {}", redos_input.len());
    
    let start = Instant::now();
    let mut obfuscator = SecureObfuscator::new(&rules).unwrap();
    let creation_time = start.elapsed();
    println!("Obfuscator creation took: {:?}", creation_time);
    
    let start = Instant::now();
    let result = obfuscator.obfuscate_text(&redos_input);
    let total_time = start.elapsed();
    println!("Total obfuscation took: {:?}", total_time);
    println!("Result length: {}, changed: {}", result.len(), result != redos_input);
}

#[test]
fn debug_performance_ssn_only() {
    // Test with SSN patterns only (no email)
    let rules = vec![
        Rule {
            pattern: r"\b\d{3}-\d{2}-\d{4}\b".to_string(),
            label: "SSN".to_string(),
        },
    ];
    
    // The problematic input
    let redos_input = format!("a@{}", "a".repeat(50000));
    
    println!("Input length: {}", redos_input.len());
    
    let start = Instant::now();
    let mut obfuscator = SecureObfuscator::new(&rules).unwrap();
    let creation_time = start.elapsed();
    println!("Obfuscator creation took: {:?}", creation_time);
    
    let start = Instant::now();
    let result = obfuscator.obfuscate_text(&redos_input);
    let total_time = start.elapsed();
    println!("Total obfuscation took: {:?}", total_time);
    println!("Result length: {}, changed: {}", result.len(), result != redos_input);
}