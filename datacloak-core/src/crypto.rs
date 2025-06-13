use aes_gcm::{
    aead::{Aead, AeadCore, KeyInit, OsRng},
    Aes256Gcm, Key, Nonce,
};
use anyhow::{anyhow, Result};

pub const KEY_SIZE: usize = 32; // 256 bits
pub const NONCE_SIZE: usize = 12; // 96 bits

/// Encrypts data using AES-256-GCM
pub fn seal(key: &[u8], plaintext: &[u8]) -> Result<Vec<u8>> {
    if key.len() != KEY_SIZE {
        anyhow::bail!(
            "Invalid key size: expected {} bytes, got {}",
            KEY_SIZE,
            key.len()
        );
    }

    let key = Key::<Aes256Gcm>::from_slice(key);
    let cipher = Aes256Gcm::new(key);

    // Generate a random nonce
    let nonce = Aes256Gcm::generate_nonce(&mut OsRng);

    // Encrypt the plaintext
    let ciphertext = cipher
        .encrypt(&nonce, plaintext)
        .map_err(|_| anyhow!("Encryption failed"))?;

    // Prepend nonce to ciphertext for storage
    let mut result = Vec::with_capacity(NONCE_SIZE + ciphertext.len());
    result.extend_from_slice(&nonce);
    result.extend_from_slice(&ciphertext);

    Ok(result)
}

/// Decrypts data using AES-256-GCM
pub fn open(key: &[u8], ciphertext: &[u8]) -> Result<Vec<u8>> {
    if key.len() != KEY_SIZE {
        anyhow::bail!(
            "Invalid key size: expected {} bytes, got {}",
            KEY_SIZE,
            key.len()
        );
    }

    if ciphertext.len() < NONCE_SIZE {
        anyhow::bail!("Ciphertext too short to contain nonce");
    }

    let key = Key::<Aes256Gcm>::from_slice(key);
    let cipher = Aes256Gcm::new(key);

    // Extract nonce and actual ciphertext
    let (nonce_bytes, encrypted_data) = ciphertext.split_at(NONCE_SIZE);
    let nonce = Nonce::from_slice(nonce_bytes);

    // Decrypt the ciphertext
    let plaintext = cipher
        .decrypt(nonce, encrypted_data)
        .map_err(|_| anyhow!("Decryption failed"))?;

    Ok(plaintext)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_seal_open_roundtrip() {
        let mut rng = rand::rng();
        let key: Vec<u8> = (0..KEY_SIZE).map(|_| rng.random::<u8>()).collect();

        let plaintext = b"Hello, World!";

        let encrypted = seal(&key, plaintext).unwrap();
        let decrypted = open(&key, &encrypted).unwrap();

        assert_eq!(plaintext, &decrypted[..]);
    }

    #[test]
    fn test_invalid_key_size() {
        let short_key = vec![0u8; 16];
        let plaintext = b"test";

        assert!(seal(&short_key, plaintext).is_err());
        assert!(open(&short_key, &[0u8; 32]).is_err());
    }

    #[test]
    fn test_corrupted_ciphertext() {
        let mut rng = rand::rng();
        let key: Vec<u8> = (0..KEY_SIZE).map(|_| rng.random::<u8>()).collect();

        let plaintext = b"test data";
        let mut encrypted = seal(&key, plaintext).unwrap();

        // Corrupt the ciphertext
        encrypted[NONCE_SIZE + 5] ^= 0xFF;

        assert!(open(&key, &encrypted).is_err());
    }
}
