use data_obfuscator::secure_obfuscator::SecureObfuscator;
use data_obfuscator::config::Rule;
use validator::ValidateEmail;
use std::time::Instant;

#[test]
fn debug_performance_bottleneck() {
    let rules = vec![
        Rule {
            pattern: r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b".to_string(),
            label: "EMAIL".to_string(),
        },
        Rule {
            pattern: r"\b\d{3}-\d{2}-\d{4}\b".to_string(),
            label: "SSN".to_string(),
        },
    ];
    
    // The problematic input
    let redos_input = format!("a@{}", "a".repeat(50000));
    
    println!("Input length: {}", redos_input.len());
    
    // Test validator alone
    let start = Instant::now();
    let is_valid = redos_input.validate_email();
    let validator_time = start.elapsed();
    println!("Validator alone took: {:?}, result: {}", validator_time, is_valid);
    
    // Test obfuscator creation
    let start = Instant::now();
    let mut obfuscator = SecureObfuscator::new(&rules).unwrap();
    let creation_time = start.elapsed();
    println!("Obfuscator creation took: {:?}", creation_time);
    
    // Test individual steps
    let start = Instant::now();
    let result1 = obfuscator.obfuscate_text(&redos_input);
    let total_time = start.elapsed();
    println!("Total obfuscation took: {:?}", total_time);
    println!("Result length: {}", result1.len());
    
    // Test shorter input for comparison
    let normal_input = "Contact user@example.com for info";
    let start = Instant::now();
    let result2 = obfuscator.obfuscate_text(normal_input);
    let normal_time = start.elapsed();
    println!("Normal input took: {:?}, result: {}", normal_time, result2);
}