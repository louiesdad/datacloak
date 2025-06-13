use data_obfuscator::config::Rule;
use data_obfuscator::obfuscator::Obfuscator;
use data_obfuscator::deobfuscator::deobfuscate_text;
use data_obfuscator::secure_obfuscator::SecureObfuscator;
use proptest::prelude::*;
use serde_json::{json, Value};

// Property test configuration
const PROPTEST_CASES: u32 = 100;
const MAX_STRING_LEN: usize = 1000;
const MAX_ARRAY_LEN: usize = 20;

// Strategy for generating realistic email addresses
fn email_strategy() -> impl Strategy<Value = String> {
    (
        "[a-z]{3,10}",
        "[a-z]{3,10}",
        "[a-z]{2,4}"
    ).prop_map(|(user, domain, tld)| {
        format!("{}@{}.{}", user, domain, tld)
    })
}

// Strategy for generating SSN patterns
fn ssn_strategy() -> impl Strategy<Value = String> {
    (100..999u32, 10..99u32, 1000..9999u32)
        .prop_map(|(area, group, serial)| {
            format!("{:03}-{:02}-{:04}", area, group, serial)
        })
}

// Strategy for generating phone numbers
fn phone_strategy() -> impl Strategy<Value = String> {
    (200..999u32, 200..999u32, 1000..9999u32)
        .prop_map(|(area, exchange, number)| {
            format!("({:03}) {:03}-{:04}", area, exchange, number)
        })
}

// Strategy for generating credit card numbers (valid Luhn)
fn credit_card_strategy() -> impl Strategy<Value = String> {
    // Generate valid Visa test numbers
    prop_oneof![
        Just("4111111111111111".to_string()), // Visa test number
        Just("4000000000000002".to_string()), // Visa test number
        Just("5555555555554444".to_string()), // Mastercard test number
        Just("378282246310005".to_string()),  // Amex test number
    ]
}

// Strategy for generating random text that might contain PII
fn text_with_pii_strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        // Plain text
        "[a-zA-Z0-9 .,!?]{10,100}",
        // Text with email
        ("[a-zA-Z ]{5,20}", email_strategy()).prop_map(|(prefix, email)| {
            format!("{} contact {}", prefix, email)
        }),
        // Text with SSN
        ("[a-zA-Z ]{5,20}", ssn_strategy()).prop_map(|(prefix, ssn)| {
            format!("{} SSN: {}", prefix, ssn)
        }),
        // Text with phone
        ("[a-zA-Z ]{5,20}", phone_strategy()).prop_map(|(prefix, phone)| {
            format!("{} call {}", prefix, phone)
        }),
        // Text with credit card
        ("[a-zA-Z ]{5,20}", credit_card_strategy()).prop_map(|(prefix, cc)| {
            format!("{} card {}", prefix, cc)
        }),
        // Mixed PII
        (email_strategy(), ssn_strategy(), phone_strategy()).prop_map(|(email, ssn, phone)| {
            format!("Customer: {} SSN: {} Phone: {}", email, ssn, phone)
        }),
    ]
}

// Strategy for generating JSON values
fn json_value_strategy() -> impl Strategy<Value = Value> {
    let leaf = prop_oneof![
        any::<bool>().prop_map(Value::Bool),
        (-1e6f64..1e6f64).prop_map(|f| json!(f)),
        (i64::MIN/1000..i64::MAX/1000).prop_map(Value::from),
        text_with_pii_strategy().prop_map(Value::String),
        email_strategy().prop_map(Value::String),
        ssn_strategy().prop_map(Value::String),
        phone_strategy().prop_map(Value::String),
        credit_card_strategy().prop_map(Value::String),
        Just(Value::Null),
    ];
    
    leaf.prop_recursive(
        3, // Max depth
        256, // Max total nodes
        10, // Items per collection
        |inner| prop_oneof![
            prop::collection::vec(inner.clone(), 0..MAX_ARRAY_LEN)
                .prop_map(Value::Array),
            prop::collection::hash_map(
                "[a-zA-Z_][a-zA-Z0-9_]{0,20}",
                inner,
                0..10
            ).prop_map(|map| {
                Value::Object(map.into_iter().collect())
            }),
        ]
    )
}

// Strategy for generating JSON lines (newline-delimited JSON)
fn json_lines_strategy() -> impl Strategy<Value = String> {
    prop::collection::vec(json_value_strategy(), 1..10)
        .prop_map(|values| {
            values.iter()
                .map(|v| serde_json::to_string(v).unwrap())
                .collect::<Vec<_>>()
                .join("\n")
        })
}

// Test rules for all PII types
fn get_test_rules() -> Vec<Rule> {
    vec![
        Rule {
            pattern: r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b".to_string(),
            label: "EMAIL".to_string(),
        },
        Rule {
            pattern: r"\b\d{3}-\d{2}-\d{4}\b".to_string(),
            label: "SSN".to_string(),
        },
        Rule {
            pattern: r"\(\d{3}\) \d{3}-\d{4}\b".to_string(),
            label: "PHONE".to_string(),
        },
        Rule {
            pattern: r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b".to_string(),
            label: "CREDIT_CARD".to_string(),
        },
    ]
}

#[cfg(test)]
mod round_trip_tests {
    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(PROPTEST_CASES))]

        #[test]
        fn prop_round_trip_obfuscator(json_lines in json_lines_strategy()) {
            let rules = get_test_rules();
            let mut obfuscator = Obfuscator::new(&rules).unwrap();
            
            // Obfuscate the JSON lines
            let obfuscated = obfuscator.obfuscate_text(&json_lines);
            
            // Get the placeholder map
            let placeholder_map = obfuscator.placeholder_map();
            
            // Deobfuscate back to original
            let deobfuscated = deobfuscate_text(&obfuscated, placeholder_map);
            
            // Verify round-trip equality
            prop_assert_eq!(
                json_lines.clone(), 
                deobfuscated.clone(),
                "Round-trip failed:\nOriginal: {}\nObfuscated: {}\nDeobfuscated: {}",
                json_lines,
                obfuscated, 
                deobfuscated
            );
        }

        #[test]
        fn prop_round_trip_secure_obfuscator(json_lines in json_lines_strategy()) {
            let rules = get_test_rules();
            let mut obfuscator = SecureObfuscator::new(&rules).unwrap();
            
            // Obfuscate the JSON lines
            let obfuscated = obfuscator.obfuscate_text(&json_lines);
            
            // Get the placeholder map
            let placeholder_map = obfuscator.placeholder_map();
            
            // Deobfuscate back to original
            let deobfuscated = deobfuscate_text(&obfuscated, placeholder_map);
            
            // Verify round-trip equality
            prop_assert_eq!(
                json_lines.clone(), 
                deobfuscated.clone(),
                "Secure round-trip failed:\nOriginal: {}\nObfuscated: {}\nDeobfuscated: {}",
                json_lines,
                obfuscated, 
                deobfuscated
            );
        }

        #[test]
        fn prop_consistent_obfuscation(text in text_with_pii_strategy()) {
            let rules = get_test_rules();
            let mut obfuscator = Obfuscator::new(&rules).unwrap();
            
            // Obfuscate the same text twice
            let obfuscated1 = obfuscator.obfuscate_text(&text);
            let obfuscated2 = obfuscator.obfuscate_text(&text);
            
            // Should produce identical results (deterministic)
            prop_assert_eq!(
                obfuscated1.clone(),
                obfuscated2.clone(),
                "Obfuscation not deterministic:\nInput: {}\nFirst: {}\nSecond: {}",
                text,
                obfuscated1,
                obfuscated2
            );
        }

        #[test]
        fn prop_token_reuse(
            email in email_strategy(),
            prefix in "[a-zA-Z ]{5,20}",
            _suffix in "[a-zA-Z ]{5,20}"
        ) {
            let rules = get_test_rules();
            let mut obfuscator = Obfuscator::new(&rules).unwrap();
            
            // Create text with the same email appearing twice
            let text = format!("{} {} and {}", prefix, email, email);
            let obfuscated = obfuscator.obfuscate_text(&text);
            
            // Count token occurrences
            let email_tokens: Vec<&str> = obfuscated.matches("[EMAIL-").collect();
            
            if !email_tokens.is_empty() {
                // If email was detected, both instances should use the same token
                let first_token = email_tokens[0];
                let token_count = obfuscated.matches(first_token).count();
                
                prop_assert_eq!(
                    token_count, 2,
                    "Same email should reuse token:\nInput: {}\nObfuscated: {}\nToken: {}",
                    text, obfuscated, first_token
                );
            }
        }

        #[test]
        fn prop_no_pii_unchanged(non_pii_text in "[a-zA-Z0-9 .,!?]{10,100}") {
            // Ensure text has no PII patterns
            prop_assume!(!non_pii_text.contains('@'));
            prop_assume!(!non_pii_text.contains('-'));
            prop_assume!(!non_pii_text.contains('('));
            
            let rules = get_test_rules();
            let mut obfuscator = Obfuscator::new(&rules).unwrap();
            
            let obfuscated = obfuscator.obfuscate_text(&non_pii_text);
            
            // Text without PII should remain unchanged
            prop_assert_eq!(
                non_pii_text.clone(),
                obfuscated.clone(),
                "Non-PII text was modified:\nOriginal: {}\nObfuscated: {}",
                non_pii_text,
                obfuscated
            );
        }

        #[test]
        fn prop_json_structure_preserved(json_value in json_value_strategy()) {
            let json_string = serde_json::to_string(&json_value).unwrap();
            
            let rules = get_test_rules();
            let mut obfuscator = Obfuscator::new(&rules).unwrap();
            
            let obfuscated = obfuscator.obfuscate_text(&json_string);
            let placeholder_map = obfuscator.placeholder_map();
            let deobfuscated = deobfuscate_text(&obfuscated, placeholder_map);
            
            // Verify JSON structure is preserved
            let original_parsed: Result<Value, _> = serde_json::from_str(&json_string);
            let deobfuscated_parsed: Result<Value, _> = serde_json::from_str(&deobfuscated);
            
            match (original_parsed, deobfuscated_parsed) {
                (Ok(orig), Ok(deobf)) => {
                    prop_assert_eq!(
                        orig, deobf,
                        "JSON structure not preserved:\nOriginal: {}\nDeobfuscated: {}",
                        json_string, deobfuscated
                    );
                },
                (Err(_), Err(_)) => {
                    // Both invalid JSON - that's fine
                },
                (Ok(_), Err(e)) => {
                    prop_assert!(
                        false,
                        "Valid JSON became invalid after round-trip:\nOriginal: {}\nDeobfuscated: {}\nError: {}",
                        json_string, deobfuscated, e
                    );
                },
                (Err(_), Ok(_)) => {
                    // Invalid became valid - unusual but not necessarily wrong
                }
            }
        }

        #[test]
        fn prop_placeholder_map_completeness(text in text_with_pii_strategy()) {
            let rules = get_test_rules();
            let mut obfuscator = Obfuscator::new(&rules).unwrap();
            
            let obfuscated = obfuscator.obfuscate_text(&text);
            let placeholder_map = obfuscator.placeholder_map();
            
            // Every placeholder in the obfuscated text should have a mapping
            let placeholders: Vec<String> = obfuscated
                .split('[')
                .skip(1)
                .filter_map(|s| s.split(']').next())
                .map(|s| format!("[{}]", s))
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            
            for placeholder in placeholders {
                prop_assert!(
                    placeholder_map.contains_key(&placeholder),
                    "Placeholder {} not found in map:\nObfuscated: {}\nMap: {:?}",
                    placeholder, obfuscated, placeholder_map
                );
            }
        }

        #[test]
        fn prop_streaming_consistency(json_lines in json_lines_strategy()) {
            // Test that streaming and direct obfuscation produce the same results
            let rules = get_test_rules();
            
            // Direct obfuscation
            let mut obfuscator1 = Obfuscator::new(&rules).unwrap();
            let direct_result = obfuscator1.obfuscate_text(&json_lines);
            
            // Streaming obfuscation (simulate by processing line by line)
            let mut obfuscator2 = Obfuscator::new(&rules).unwrap();
            let streaming_result = json_lines
                .lines()
                .map(|line| obfuscator2.obfuscate_text(line))
                .collect::<Vec<_>>()
                .join("\n");
            
            // Results should be consistent
            prop_assert_eq!(
                direct_result.clone(),
                streaming_result.clone(),
                "Streaming and direct obfuscation differ:\nDirect: {}\nStreaming: {}",
                direct_result,
                streaming_result
            );
        }
    }

    #[test]
    fn test_specific_edge_cases() {
        let rules = get_test_rules();
        
        // Test empty string
        let mut obfuscator = Obfuscator::new(&rules).unwrap();
        let obfuscated = obfuscator.obfuscate_text("");
        assert_eq!("", obfuscated);
        
        // Test only whitespace
        let obfuscated = obfuscator.obfuscate_text("   \n\t  ");
        assert_eq!("   \n\t  ", obfuscated);
        
        // Test only PII
        let obfuscated = obfuscator.obfuscate_text("test@example.com");
        assert!(obfuscated.contains("[EMAIL-"));
        
        // Test malformed email-like strings
        let obfuscated = obfuscator.obfuscate_text("not-an-email@");
        assert_eq!("not-an-email@", obfuscated);
        
        // Test Unicode
        let obfuscated = obfuscator.obfuscate_text("héllo wörld test@example.com");
        assert!(obfuscated.contains("héllo wörld"));
        assert!(obfuscated.contains("[EMAIL-"));
    }
}

#[cfg(test)]
mod performance_properties {
    use super::*;
    use std::time::{Duration, Instant};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))] // Fewer cases for performance tests

        #[test]
        fn prop_obfuscation_performance(text in "[a-zA-Z0-9 @.-]{100,1000}") {
            let rules = get_test_rules();
            let mut obfuscator = Obfuscator::new(&rules).unwrap();
            
            let start = Instant::now();
            let _obfuscated = obfuscator.obfuscate_text(&text);
            let duration = start.elapsed();
            
            // Should complete within reasonable time (10ms for small text)
            prop_assert!(
                duration < Duration::from_millis(10),
                "Obfuscation too slow: {:?} for {} chars",
                duration, text.len()
            );
        }

        #[test]
        fn prop_memory_bounded(large_text in "[a-zA-Z0-9 @.-]{1000,5000}") {
            let rules = get_test_rules();
            let mut obfuscator = Obfuscator::new(&rules).unwrap();
            
            // Should not crash or consume excessive memory
            let obfuscated = obfuscator.obfuscate_text(&large_text);
            
            // Obfuscated text should be reasonable size
            prop_assert!(
                obfuscated.len() <= large_text.len() * 3, // Allow for token expansion
                "Obfuscated text unexpectedly large: {} -> {}",
                large_text.len(), obfuscated.len()
            );
        }
    }
}