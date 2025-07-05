//! Pattern definitions and types

use serde::{Deserialize, Serialize};
use std::fmt;

/// Types of PII patterns that can be detected and obfuscated
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatternType {
    Email,
    SSN,
    Phone,
    CreditCard,
    IPAddress,
    DateOfBirth,
    MedicalRecordNumber,
    DriversLicense,
    BankAccount,
    Passport,
    Name,
    Address,
    Custom(u32), // For user-defined patterns
}

impl PatternType {
    /// Get all standard pattern types (excluding custom)
    pub fn all() -> Vec<PatternType> {
        vec![
            PatternType::Email,
            PatternType::SSN,
            PatternType::Phone,
            PatternType::CreditCard,
            PatternType::IPAddress,
            PatternType::DateOfBirth,
            PatternType::MedicalRecordNumber,
            PatternType::DriversLicense,
            PatternType::BankAccount,
            PatternType::Passport,
            PatternType::Name,
            PatternType::Address,
        ]
    }
}

impl fmt::Display for PatternType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PatternType::Email => write!(f, "EMAIL"),
            PatternType::SSN => write!(f, "SSN"),
            PatternType::Phone => write!(f, "PHONE"),
            PatternType::CreditCard => write!(f, "CREDIT_CARD"),
            PatternType::IPAddress => write!(f, "IP_ADDRESS"),
            PatternType::DateOfBirth => write!(f, "DATE_OF_BIRTH"),
            PatternType::MedicalRecordNumber => write!(f, "MRN"),
            PatternType::DriversLicense => write!(f, "DRIVERS_LICENSE"),
            PatternType::BankAccount => write!(f, "BANK_ACCOUNT"),
            PatternType::Passport => write!(f, "PASSPORT"),
            PatternType::Name => write!(f, "NAME"),
            PatternType::Address => write!(f, "ADDRESS"),
            PatternType::Custom(id) => write!(f, "CUSTOM_{}", id),
        }
    }
}

/// A pattern configuration for obfuscation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    pub pattern_type: PatternType,
    pub regex: String,
    pub description: Option<String>,
    pub enabled: bool,
    pub priority: u32, // Higher priority patterns are applied first
}

impl Pattern {
    /// Create a new pattern
    pub fn new(pattern_type: PatternType, regex: String) -> Self {
        Self {
            pattern_type,
            regex,
            description: None,
            enabled: true,
            priority: 100,
        }
    }

    /// Create a pattern with description
    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }

    /// Set pattern priority
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }
}

/// Pattern matching result
#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub pattern_type: PatternType,
    pub start: usize,
    pub end: usize,
    pub matched_text: String,
    pub replacement_token: String,
}

/// Collection of patterns with utility methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternSet {
    patterns: Vec<Pattern>,
}

impl PatternSet {
    /// Create a new empty pattern set
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
        }
    }

    /// Create a pattern set with default PII patterns
    pub fn default_pii() -> Self {
        let mut set = Self::new();

        // Add common PII patterns
        set.add(
            Pattern::new(
                PatternType::Email,
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b".to_string(),
            )
            .with_description("Email addresses".to_string())
            .with_priority(100),
        );

        set.add(
            Pattern::new(PatternType::SSN, r"\b\d{3}-\d{2}-\d{4}\b".to_string())
                .with_description("Social Security Numbers".to_string())
                .with_priority(200),
        );

        set.add(
            Pattern::new(
                PatternType::Phone,
                r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b".to_string(),
            )
            .with_description("Phone numbers".to_string())
            .with_priority(90),
        );

        set.add(
            Pattern::new(
                PatternType::CreditCard,
                r"\b4[0-9]{12}(?:[0-9]{3})?\b".to_string(),
            )
            .with_description("Credit card numbers (Visa)".to_string())
            .with_priority(150),
        );

        set
    }

    /// Create a pattern set for HIPAA compliance
    pub fn hipaa_compliance() -> Self {
        let mut set = Self::default_pii();

        // Add HIPAA-specific patterns
        set.add(
            Pattern::new(
                PatternType::MedicalRecordNumber,
                r"\bMRN\s*:?\s*\d{6,10}\b".to_string(),
            )
            .with_description("Medical Record Numbers".to_string())
            .with_priority(180),
        );

        set.add(
            Pattern::new(
                PatternType::DateOfBirth,
                r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b".to_string(),
            )
            .with_description("Dates of Birth".to_string())
            .with_priority(120),
        );

        set
    }

    /// Add a pattern to the set
    pub fn add(&mut self, pattern: Pattern) {
        self.patterns.push(pattern);
        self.sort_by_priority();
    }

    /// Remove patterns of a specific type
    pub fn remove_type(&mut self, pattern_type: PatternType) {
        self.patterns.retain(|p| p.pattern_type != pattern_type);
    }

    /// Get all enabled patterns sorted by priority
    pub fn enabled_patterns(&self) -> Vec<&Pattern> {
        self.patterns.iter().filter(|p| p.enabled).collect()
    }

    /// Sort patterns by priority (descending)
    fn sort_by_priority(&mut self) {
        self.patterns.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Get patterns as a slice
    pub fn as_slice(&self) -> &[Pattern] {
        &self.patterns
    }

    /// Convert to a vector
    pub fn to_vec(self) -> Vec<Pattern> {
        self.patterns
    }
}

impl Default for PatternSet {
    fn default() -> Self {
        Self::new()
    }
}

impl From<Vec<Pattern>> for PatternSet {
    fn from(patterns: Vec<Pattern>) -> Self {
        let mut set = Self { patterns };
        set.sort_by_priority();
        set
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_creation() {
        let pattern = Pattern::new(
            PatternType::Email,
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b".to_string(),
        );

        assert_eq!(pattern.pattern_type, PatternType::Email);
        assert!(pattern.enabled);
        assert_eq!(pattern.priority, 100);
    }

    #[test]
    fn test_pattern_set() {
        let set = PatternSet::default_pii();
        let enabled = set.enabled_patterns();
        assert!(!enabled.is_empty());

        // Check that patterns are sorted by priority
        for i in 1..enabled.len() {
            assert!(enabled[i - 1].priority >= enabled[i].priority);
        }
    }
}
