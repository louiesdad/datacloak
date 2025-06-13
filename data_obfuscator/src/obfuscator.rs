use regex::{Captures, Regex};
use std::collections::HashMap;
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
}

impl Obfuscator {
    pub fn new(rule_cfg: &[Rule]) -> Result<Self, ObfuscationError> {
        let mut rules = Vec::new();
        for r in rule_cfg {
            let compiled = Regex::new(&r.pattern)
                .map_err(|e| ObfuscationError::RegexCompile(e.to_string()))?;
            rules.push((compiled, r.label.clone()));
        }
        Ok(Self {
            rules,
            placeholder_counter: 0,
            placeholder_map: HashMap::new(),
            reverse_map: HashMap::new(),
        })
    }

    pub fn obfuscate_text(&mut self, input: &str) -> String {
        let mut intermediate = input.to_string();
        for (regex, label) in &self.rules {
            let placeholder_counter = &mut self.placeholder_counter;
            let placeholder_map = &mut self.placeholder_map;
            let reverse_map = &mut self.reverse_map;
            intermediate = regex
                .replace_all(&intermediate, |caps: &Captures| {
                    let matched = caps.get(0).unwrap().as_str().to_string();
                    if let Some(token) = reverse_map.get(&matched) {
                        token.clone()
                    } else {
                        let token = format!("[{}-{}]", label, *placeholder_counter);
                        *placeholder_counter += 1;
                        placeholder_map.insert(token.clone(), matched.clone());
                        reverse_map.insert(matched, token.clone());
                        token
                    }
                })
                .into_owned();
        }
        intermediate
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