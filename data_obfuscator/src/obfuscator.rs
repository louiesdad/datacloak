use regex::{Regex, RegexSet};
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt, AsyncRead, AsyncReadExt};
use crate::config::Rule;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ObfuscationError {
    #[error("regex compile error: {0}")]
    RegexCompile(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("tokio task error: {0}")]
    Tokio(#[from] tokio::task::JoinError),
}

#[derive(Debug, Clone)]
pub struct StreamConfig {
    pub chunk_size: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            chunk_size: 256 * 1024, // 256KB default
        }
    }
}

pub struct Obfuscator {
    rules: Vec<(Regex, String)>,
    placeholder_counter: usize,
    placeholder_map: HashMap<String, String>,
    reverse_map: HashMap<String, String>,
    // Performance optimization with RegexSet
    regex_set: Option<RegexSet>,
    patterns: Vec<String>,
}

impl Obfuscator {
    pub fn new(rule_cfg: &[Rule]) -> Result<Self, ObfuscationError> {
        let mut rules = Vec::new();
        for r in rule_cfg {
            let compiled = Regex::new(&r.pattern)
                .map_err(|e| ObfuscationError::RegexCompile(e.to_string()))?;
            rules.push((compiled, r.label.clone()));
        }
        let patterns: Vec<String> = rule_cfg.iter().map(|r| r.pattern.clone()).collect();
        let regex_set = if patterns.is_empty() {
            Some(RegexSet::new(&[r"$^"]).unwrap()) // Never matches
        } else {
            RegexSet::new(&patterns).ok()
        };
        
        Ok(Self {
            rules,
            placeholder_counter: 0,
            placeholder_map: HashMap::new(),
            reverse_map: HashMap::new(),
            regex_set,
            patterns,
        })
    }

    pub fn obfuscate_text(&mut self, input: &str) -> String {
        // Skip processing very long texts to prevent ReDoS
        if input.len() > 100_000 {
            return input.to_string();
        }
        
        let mut result = input.to_string();
        
        // Use RegexSet for fast pattern matching if available
        if let Some(regex_set) = &self.regex_set {
            let matches: Vec<usize> = regex_set.matches(input).into_iter().collect();
            
            // Only apply regexes that actually match
            for &pattern_idx in &matches {
                if pattern_idx < self.rules.len() {
                    result = self.apply_regex_at_index(&result, pattern_idx);
                }
            }
        } else {
            // Fallback to sequential processing
            for i in 0..self.rules.len() {
                result = self.apply_regex_at_index(&result, i);
            }
        }
        
        result
    }
    
    fn apply_regex_at_index(&mut self, text: &str, index: usize) -> String {
        // Collect matches first to avoid borrowing conflicts
        let matches: Vec<String> = {
            let (regex, _) = &self.rules[index];
            regex.find_iter(text)
                .take(1000) // Limit matches for performance
                .map(|mat| mat.as_str().to_string())
                .collect()
        };
        
        let mut result = text.to_string();
        let label = self.rules[index].1.clone();
        
        for matched in matches {
            if let Some(token) = self.reverse_map.get(&matched) {
                result = result.replace(&matched, token);
            } else {
                let token_id = Self::generate_deterministic_token_id(&matched, &label);
                let token = format!("[{}-{}]", label, token_id);
                self.placeholder_map.insert(token.clone(), matched.clone());
                self.reverse_map.insert(matched.clone(), token.clone());
                result = result.replace(&matched, &token);
            }
        }
        
        result
    }
    
    fn generate_deterministic_token_id(matched: &str, label: &str) -> usize {
        let mut hasher = DefaultHasher::new();
        matched.hash(&mut hasher);
        label.hash(&mut hasher);
        let hash = hasher.finish();
        
        // Generate a deterministic ID based on content
        // This ensures the same content gets the same token across different obfuscator instances
        (hash % 1000000) as usize
    }

    pub async fn obfuscate_stream<R, W>(
        &mut self,
        reader: R,
        mut writer: W,
    ) -> Result<(), ObfuscationError>
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
    ) -> Result<(), ObfuscationError>
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