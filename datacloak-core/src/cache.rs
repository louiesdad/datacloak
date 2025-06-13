//! Secure caching for obfuscation mappings

use crate::{crypto, DataCloakError, Result};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Cache for storing obfuscation mappings
pub struct ObfuscationCache {
    /// In-memory cache for fast lookups
    memory_cache: Arc<DashMap<String, CacheEntry>>,
    /// Optional persistent storage path
    storage_path: Option<PathBuf>,
    /// Encryption key for secure storage (in production, use proper key management)
    encryption_key: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheEntry {
    token: String,
    original: String,
    pattern_type: String,
    created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Serialize, Deserialize)]
struct CacheDump {
    version: u32,
    entries: Vec<CacheEntry>,
    created_at: chrono::DateTime<chrono::Utc>,
}

impl ObfuscationCache {
    /// Create a new cache
    pub fn new() -> Self {
        Self {
            memory_cache: Arc::new(DashMap::new()),
            storage_path: None,
            encryption_key: None,
        }
    }

    /// Create a cache with persistent storage
    pub fn with_storage(storage_path: PathBuf) -> Result<Self> {
        // Require encryption key from environment
        let key = env::var("DATACLOAK_CACHE_KEY")
            .map_err(|_| DataCloakError::Cache(
                "DATACLOAK_CACHE_KEY environment variable not set. Cache encryption requires a 32-byte hex key.".into()
            ))?;

        // Decode hex key
        let key_bytes = hex::decode(&key)
            .map_err(|e| DataCloakError::Cache(
                format!("Invalid DATACLOAK_CACHE_KEY format: {}. Expected 64 character hex string (32 bytes).", e)
            ))?;

        if key_bytes.len() != crypto::KEY_SIZE {
            return Err(DataCloakError::Cache(format!(
                "Invalid DATACLOAK_CACHE_KEY size: expected {} bytes, got {}",
                crypto::KEY_SIZE,
                key_bytes.len()
            )));
        }

        Ok(Self {
            memory_cache: Arc::new(DashMap::new()),
            storage_path: Some(storage_path),
            encryption_key: Some(key_bytes),
        })
    }

    /// Enable encryption for persistent storage
    pub fn with_encryption(mut self, key: Vec<u8>) -> Self {
        self.encryption_key = Some(key);
        self
    }

    /// Store a mapping
    pub fn store(&self, token: String, original: String, pattern_type: String) {
        let entry = CacheEntry {
            token: token.clone(),
            original,
            pattern_type,
            created_at: chrono::Utc::now(),
        };

        self.memory_cache.insert(token, entry);
    }

    /// Retrieve original value for a token
    pub fn get_original(&self, token: &str) -> Option<String> {
        self.memory_cache
            .get(token)
            .map(|entry| entry.original.clone())
    }

    /// Retrieve token for an original value
    pub fn get_token(&self, original: &str) -> Option<String> {
        // This is O(n) - in production, maintain a reverse index
        for entry in self.memory_cache.iter() {
            if entry.value().original == original {
                return Some(entry.key().clone());
            }
        }
        None
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            total_entries: self.memory_cache.len(),
            memory_size_estimate: self.estimate_memory_size(),
        }
    }

    /// Estimate memory usage
    fn estimate_memory_size(&self) -> usize {
        self.memory_cache
            .iter()
            .map(|entry| {
                entry.key().len()
                    + entry.value().original.len()
                    + entry.value().pattern_type.len()
                    + 32 // Overhead estimate
            })
            .sum()
    }

    /// Save cache to persistent storage
    pub async fn save(&self) -> Result<()> {
        let Some(ref path) = self.storage_path else {
            return Ok(());
        };

        // Collect all entries
        let entries: Vec<CacheEntry> = self
            .memory_cache
            .iter()
            .map(|entry| entry.value().clone())
            .collect();

        let dump = CacheDump {
            version: 1,
            entries,
            created_at: chrono::Utc::now(),
        };

        // Serialize to JSON
        let json = serde_json::to_vec(&dump)?;

        // Optionally encrypt
        let data = if let Some(ref key) = self.encryption_key {
            self.encrypt(&json, key)?
        } else {
            json
        };

        // Write to file
        let mut file = fs::File::create(path).await?;
        file.write_all(&data).await?;
        file.sync_all().await?;

        Ok(())
    }

    /// Load cache from persistent storage
    pub async fn load(&self) -> Result<()> {
        let Some(ref path) = self.storage_path else {
            return Ok(());
        };

        if !path.exists() {
            return Ok(());
        }

        // Read file
        let mut file = fs::File::open(path).await?;
        let mut data = Vec::new();
        file.read_to_end(&mut data).await?;

        // Optionally decrypt
        let json = if let Some(ref key) = self.encryption_key {
            self.decrypt(&data, key)?
        } else {
            data
        };

        // Deserialize
        let dump: CacheDump = serde_json::from_slice(&json)?;

        // Load into memory cache
        self.memory_cache.clear();
        for entry in dump.entries {
            self.memory_cache.insert(entry.token.clone(), entry);
        }

        Ok(())
    }

    /// Clear the cache
    pub fn clear(&self) {
        self.memory_cache.clear();
    }

    /// Encrypt data using AES-256-GCM
    fn encrypt(&self, data: &[u8], key: &[u8]) -> Result<Vec<u8>> {
        crypto::seal(key, data)
            .map_err(|e| DataCloakError::Cache(format!("Encryption error: {}", e)))
    }

    /// Decrypt data using AES-256-GCM
    fn decrypt(&self, data: &[u8], key: &[u8]) -> Result<Vec<u8>> {
        crypto::open(key, data)
            .map_err(|e| DataCloakError::Cache(format!("Decryption error: {}", e)))
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_entries: usize,
    pub memory_size_estimate: usize,
}

impl Default for ObfuscationCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_operations() {
        let cache = ObfuscationCache::new();

        cache.store(
            "TOKEN-1".to_string(),
            "john@example.com".to_string(),
            "EMAIL".to_string(),
        );

        assert_eq!(
            cache.get_original("TOKEN-1"),
            Some("john@example.com".to_string())
        );
        assert_eq!(
            cache.get_token("john@example.com"),
            Some("TOKEN-1".to_string())
        );

        let stats = cache.stats();
        assert_eq!(stats.total_entries, 1);
    }

    #[tokio::test]
    async fn test_persistence() {
        use tempfile::tempdir;

        // Set test encryption key
        env::set_var(
            "DATACLOAK_CACHE_KEY",
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        );

        let dir = tempdir().unwrap();
        let cache_file = dir.path().join("cache.bin");

        let cache = ObfuscationCache::with_storage(cache_file.clone()).unwrap();
        cache.store(
            "TOKEN-1".to_string(),
            "test@example.com".to_string(),
            "EMAIL".to_string(),
        );

        cache.save().await.unwrap();

        // Create new cache and load
        let cache2 = ObfuscationCache::with_storage(cache_file).unwrap();
        cache2.load().await.unwrap();

        assert_eq!(
            cache2.get_original("TOKEN-1"),
            Some("test@example.com".to_string())
        );
    }
}
