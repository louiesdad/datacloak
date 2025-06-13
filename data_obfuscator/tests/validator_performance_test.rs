use validator::ValidateEmail;
use std::time::Instant;

#[test]
fn test_validator_handles_redos_case_fast() {
    // This is the exact ReDoS case that was failing in the benchmark
    let redos_email = format!("a@{}", "a".repeat(50000));
    
    let start = Instant::now();
    let is_valid = redos_email.validate_email();
    let elapsed = start.elapsed();
    
    // Should be invalid (domain too long) and fast
    assert!(!is_valid, "Email should be invalid due to domain length");
    assert!(elapsed.as_millis() < 1, 
           "Validation took {}ms, should be under 1ms", elapsed.as_millis());
    
    println!("ReDoS test email validation took: {:?}", elapsed);
}

#[test]
fn test_validator_normal_email_cases() {
    let test_cases = vec![
        ("user@example.com", true),
        ("test.email+tag@example.co.uk", true),
        ("user.name@domain-name.com", true),
        ("not-an-email", false),
        ("@example.com", false),
        ("user@", false),
        ("user@.com", false),
    ];
    
    for (email, expected) in test_cases {
        let start = Instant::now();
        let is_valid = email.validate_email();
        let elapsed = start.elapsed();
        
        assert_eq!(is_valid, expected, "Email '{}' validation mismatch", email);
        assert!(elapsed.as_millis() < 1, 
               "Validation of '{}' took {}ms, should be under 1ms", email, elapsed.as_millis());
    }
}

#[test]
fn test_validator_rfc_length_limits() {
    // Test RFC 5321 length limits are properly enforced
    
    // Valid lengths
    let valid_local = "a".repeat(64); // max local part
    let valid_domain = format!("{}.com", "a".repeat(59)); // valid domain
    let valid_email = format!("{}@{}", valid_local, valid_domain);
    assert!(valid_email.validate_email(), "Should accept valid length email");
    
    // Invalid lengths  
    let long_local = "a".repeat(65); // exceeds 64 char limit
    let long_local_email = format!("{}@example.com", long_local);
    assert!(!long_local_email.validate_email(), "Should reject long local part");
    
    let long_domain = format!("{}.com", "a".repeat(252)); // exceeds 255 char limit  
    let long_domain_email = format!("user@{}", long_domain);
    assert!(!long_domain_email.validate_email(), "Should reject long domain");
}