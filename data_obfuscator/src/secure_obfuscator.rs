use regex::{RegexSet, Regex};
use std::collections::HashMap;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt, AsyncRead, AsyncReadExt};
use crate::config::Rule;
use crate::obfuscator::StreamConfig;
use thiserror::Error;
use validator::ValidateEmail;
use once_cell::sync::Lazy;

// Unused - kept for potential future optimization
#[allow(dead_code)]
static EMAIL_CANDIDATE_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b[^\s@]{1,64}@[^\s@]{1,255}\.[^\s@]{2,}\b").unwrap()
});

static CREDIT_CARD_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b").unwrap()
});

#[derive(Debug, Error)]
pub enum SecureObfuscationError {
    #[error("regex compile error: {0}")]
    RegexCompile(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("tokio task error: {0}")]
    Tokio(#[from] tokio::task::JoinError),
}

pub struct SecureObfuscator {
    patterns: Vec<String>,
    labels: Vec<String>,
    placeholder_counter: usize,
    placeholder_map: HashMap<String, String>,
    reverse_map: HashMap<String, String>,
    email_label: String,
    credit_card_label: String,
    // Lazily compiled regexes
    regex_set: Option<RegexSet>,
    individual_regexes: Option<Vec<Regex>>,
}

impl SecureObfuscator {
    pub fn new(rule_cfg: &[Rule]) -> Result<Self, SecureObfuscationError> {
        let mut patterns = Vec::new();
        let mut labels = Vec::new();
        let mut email_label = "EMAIL".to_string();
        let mut credit_card_label = "CREDIT_CARD".to_string();
        
        for rule in rule_cfg {
            // Skip email patterns - we'll handle them with validator
            if rule.label.to_uppercase() == "EMAIL" {
                email_label = rule.label.clone();
                continue;
            }
            
            // Skip credit card patterns - we'll handle them with luhn
            if rule.label.to_uppercase().contains("CREDIT") || rule.label.to_uppercase().contains("CARD") {
                credit_card_label = rule.label.clone();
                continue;
            }
            
            patterns.push(rule.pattern.clone());
            labels.push(rule.label.clone());
        }
        
        Ok(Self {
            patterns,
            labels,
            placeholder_counter: 0,
            placeholder_map: HashMap::new(),
            reverse_map: HashMap::new(),
            email_label,
            credit_card_label,
            regex_set: None,
            individual_regexes: None,
        })
    }
    
    pub fn obfuscate_text(&mut self, input: &str) -> String {
        let mut result = input.to_string();
        
        // Process emails using validator
        result = self.obfuscate_emails(&result);
        
        // Process credit cards using luhn validation
        result = self.obfuscate_credit_cards(&result);
        
        // Process remaining patterns using RegexSet
        result = self.obfuscate_with_regex_set(&result);
        
        result
    }
    
    fn obfuscate_emails(&mut self, text: &str) -> String {
        let mut result = text.to_string();
        
        // Split by whitespace and check each word for potential emails
        // This avoids regex backtracking issues entirely
        let words: Vec<&str> = text.split_whitespace().collect();
        
        for word in words {
            // Quick checks before expensive validation
            if word.len() > 5 && word.len() <= 320 && word.contains('@') && word.contains('.') {
                // Remove common punctuation from the end
                let cleaned = word.trim_end_matches(|c: char| ".,;:!?".contains(c));
                
                if ValidateEmail::validate_email(&cleaned) {
                    let placeholder = self.get_or_create_placeholder(&self.email_label.clone(), cleaned);
                    result = result.replace(cleaned, &placeholder);
                }
            }
        }
        
        result
    }
    
    fn obfuscate_credit_cards(&mut self, text: &str) -> String {
        let mut result = text.to_string();
        
        for captures in CREDIT_CARD_REGEX.find_iter(text) {
            let matched = captures.as_str();
            // Extract just digits
            let digits: String = matched.chars().filter(|c| c.is_ascii_digit()).collect();
            
            if digits.len() >= 13 && digits.len() <= 19 && luhn::valid(&digits) {
                let placeholder = self.get_or_create_placeholder(&self.credit_card_label.clone(), matched);
                result = result.replace(matched, &placeholder);
            }
        }
        
        result
    }
    
    fn ensure_regexes_compiled(&mut self) -> Result<(), SecureObfuscationError> {
        if self.regex_set.is_none() || self.individual_regexes.is_none() {
            let mut individual_regexes = Vec::new();
            
            for pattern in &self.patterns {
                let compiled = Regex::new(pattern)
                    .map_err(|e| SecureObfuscationError::RegexCompile(e.to_string()))?;
                individual_regexes.push(compiled);
            }
            
            let regex_set = if self.patterns.is_empty() {
                RegexSet::new(&[r"$^"]).unwrap() // Never matches anything
            } else {
                RegexSet::new(&self.patterns)
                    .map_err(|e| SecureObfuscationError::RegexCompile(e.to_string()))?
            };
            
            self.regex_set = Some(regex_set);
            self.individual_regexes = Some(individual_regexes);
        }
        Ok(())
    }

    fn obfuscate_with_regex_set(&mut self, text: &str) -> String {
        let mut result = text.to_string();
        
        // Skip processing if text is too long to avoid ReDoS
        if text.len() > 100_000 {
            return result;
        }
        
        // Only compile regexes if we actually need them
        if self.patterns.is_empty() {
            return result;
        }
        
        if let Err(_) = self.ensure_regexes_compiled() {
            return result; // Skip regex processing on compilation error
        }
        
        let regex_set = self.regex_set.as_ref().unwrap();
        let individual_regexes = self.individual_regexes.as_ref().unwrap();
        
        // Use RegexSet to quickly identify which patterns match
        let matches: Vec<usize> = regex_set.matches(text).into_iter().collect();
        
        // Collect all replacements first to avoid borrowing conflicts
        let mut all_replacements = Vec::new();
        
        for &pattern_idx in &matches {
            if pattern_idx < individual_regexes.len() {
                let regex = &individual_regexes[pattern_idx];
                let label = self.labels[pattern_idx].clone();
                
                // Collect all matches first with timeout protection
                let matches_found: Vec<String> = regex.find_iter(&result)
                    .take(1000) // Limit matches to prevent excessive processing
                    .map(|mat| mat.as_str().to_string())
                    .collect();
                
                for matched in matches_found {
                    all_replacements.push((label.clone(), matched));
                }
            }
        }
        
        // Apply all replacements
        for (label, matched) in all_replacements {
            let placeholder = self.get_or_create_placeholder(&label, &matched);
            result = result.replace(&matched, &placeholder);
        }
        
        result
    }
    
    fn get_or_create_placeholder(&mut self, label: &str, matched: &str) -> String {
        if let Some(token) = self.reverse_map.get(matched) {
            token.clone()
        } else {
            let token = format!("[{}-{}]", label, self.placeholder_counter);
            self.placeholder_counter += 1;
            self.placeholder_map.insert(token.clone(), matched.to_string());
            self.reverse_map.insert(matched.to_string(), token.clone());
            token
        }
    }
    
    pub async fn obfuscate_stream<R, W>(
        &mut self,
        reader: R,
        mut writer: W,
    ) -> Result<(), SecureObfuscationError>
    where
        R: AsyncBufRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        let mut lines = reader.lines();
        while let Some(line) = lines.next_line().await? {
            let obfuscated = self.obfuscate_text(&line);
            writer.write_all(obfuscated.as_bytes()).await?;
            writer.write_all(b"\n").await?;
        }
        writer.flush().await?;
        Ok(())
    }

    pub async fn stream_file<R, W>(
        &mut self,
        mut reader: R,
        mut writer: W,
        config: &StreamConfig,
    ) -> Result<(), SecureObfuscationError>
    where
        R: AsyncRead + Unpin,
        W: AsyncWrite + Unpin,
    {
        let mut buffer = vec![0u8; config.chunk_size];
        let mut partial_line = String::new();

        loop {
            let bytes_read = reader.read(&mut buffer).await?;
            if bytes_read == 0 {
                break; // EOF reached
            }

            // Convert bytes to string and handle partial lines
            let chunk_str = String::from_utf8_lossy(&buffer[..bytes_read]);
            let full_text = format!("{}{}", partial_line, chunk_str);
            
            // Split into lines, keeping the last incomplete line for next iteration
            let mut lines: Vec<&str> = full_text.lines().collect();
            
            // Check if the last line is complete (ends with newline)
            let chunk_ends_with_newline = buffer[..bytes_read].ends_with(b"\n");
            
            if !chunk_ends_with_newline && !lines.is_empty() {
                // Last line is incomplete, save it for next iteration
                partial_line = lines.pop().unwrap_or("").to_string();
            } else {
                partial_line.clear();
            }

            // Process complete lines
            for line in lines {
                let obfuscated = self.obfuscate_text(line);
                writer.write_all(obfuscated.as_bytes()).await?;
                writer.write_all(b"\n").await?;
            }
        }

        // Process any remaining partial line
        if !partial_line.is_empty() {
            let obfuscated = self.obfuscate_text(&partial_line);
            writer.write_all(obfuscated.as_bytes()).await?;
            if !partial_line.ends_with('\n') {
                writer.write_all(b"\n").await?;
            }
        }

        writer.flush().await?;
        Ok(())
    }
    
    pub fn placeholder_map(&self) -> &HashMap<String, String> {
        &self.placeholder_map
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Rule;
    
    #[test]
    fn test_email_obfuscation() {
        let rules = vec![
            Rule {
                pattern: r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b".to_string(),
                label: "EMAIL".to_string(),
            }
        ];
        
        let mut obfuscator = SecureObfuscator::new(&rules).unwrap();
        let input = "Contact user@example.com for info";
        let result = obfuscator.obfuscate_text(input);
        
        assert!(result.contains("[EMAIL-0]"));
        assert!(!result.contains("user@example.com"));
    }
    
    #[test]
    fn test_credit_card_obfuscation() {
        let rules = vec![
            Rule {
                pattern: "unused".to_string(),
                label: "CREDIT_CARD".to_string(),
            }
        ];
        
        let mut obfuscator = SecureObfuscator::new(&rules).unwrap();
        // Valid test credit card number (Visa test number)
        let input = "Payment card: 4532015112830366";
        let result = obfuscator.obfuscate_text(input);
        
        assert!(result.contains("[CREDIT_CARD-0]"));
        assert!(!result.contains("4532015112830366"));
    }
    
    #[test]
    fn test_ssn_pattern_safe() {
        let rules = vec![
            Rule {
                pattern: r"\b\d{3}-\d{2}-\d{4}\b".to_string(),
                label: "SSN".to_string(),
            }
        ];
        
        let mut obfuscator = SecureObfuscator::new(&rules).unwrap();
        let input = "SSN: 123-45-6789";
        let result = obfuscator.obfuscate_text(input);
        
        assert!(result.contains("[SSN-0]"));
        assert!(!result.contains("123-45-6789"));
    }
    
    #[test] 
    fn test_invalid_email_not_obfuscated() {
        let rules = vec![
            Rule {
                pattern: r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b".to_string(),
                label: "EMAIL".to_string(),
            }
        ];
        
        let mut obfuscator = SecureObfuscator::new(&rules).unwrap();
        let input = "Invalid email: not-an-email@";
        let result = obfuscator.obfuscate_text(input);
        
        // Should not be obfuscated since it's not a valid email
        assert_eq!(result, input);
    }
    
    #[test]
    fn test_invalid_credit_card_not_obfuscated() {
        let rules = vec![
            Rule {
                pattern: "unused".to_string(),
                label: "CREDIT_CARD".to_string(),
            }
        ];
        
        let mut obfuscator = SecureObfuscator::new(&rules).unwrap();
        // Invalid credit card (fails luhn check)
        let input = "Invalid card: 1234 5678 9012 3456";
        let result = obfuscator.obfuscate_text(input);
        
        // Should not be obfuscated since it fails luhn validation
        assert_eq!(result, input);
    }
    
    #[test]
    fn test_email_validation_performance_various_long_strings() {
        use std::time::Instant;
        
        // Test various potentially problematic email patterns
        let test_cases = vec![
            // Basic long domain
            format!("a@{}", "a".repeat(50000)),
            // Long user part
            format!("{}@example.com", "a".repeat(50000)),
            // Complex pattern that might cause backtracking
            format!("{}@{}.com", "a+b".repeat(10000), "sub.domain".repeat(5000)),
            // Pattern with multiple dots (potential backtracking)
            format!("user@{}", "a.".repeat(25000)),
            // Long string with common ReDoS pattern
            format!("{}@domain.com", "a".repeat(30000) + "X"),
        ];
        
        for (i, test_email) in test_cases.iter().enumerate() {
            let start = Instant::now();
            let is_valid = ValidateEmail::validate_email(test_email);
            let duration = start.elapsed();
            
            println!("Test case {}: Email validation took: {:?} for length {}", i, duration, test_email.len());
            println!("  Email is valid: {}", is_valid);
            
            // Test should pass in under 1ms for security
            if duration.as_millis() >= 1 {
                panic!("Test case {} took {}ms (>=1ms) for string length {}: {}", 
                       i, duration.as_millis(), test_email.len(),
                       if test_email.len() > 100 { &test_email[..100] } else { test_email });
            }
        }
    }
}