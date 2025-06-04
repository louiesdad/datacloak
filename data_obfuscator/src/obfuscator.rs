use regex::{Captures, Regex};
use std::collections::HashMap;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt};
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
            let mut counter = 0;
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
                        counter += 1;
                        token
                    }
                })
                .into_owned();
        }
        intermediate
    }

    pub fn placeholder_map(&self) -> &HashMap<String, String> {
        &self.placeholder_map
    }

    pub async fn obfuscate_stream<R, W>(
        &mut self,
        mut reader: R,
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
}
