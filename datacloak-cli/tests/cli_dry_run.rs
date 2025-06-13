use std::path::PathBuf;
use std::io::Write;
use tempfile::NamedTempFile;
use serde_json::Value;
use anyhow::Result;

#[tokio::test]
async fn test_obfuscate_dry_run_mode() -> Result<()> {
    // Create a temporary CSV file with test data
    let mut temp_file = NamedTempFile::new()?;
    writeln!(temp_file, "name,email,phone,ssn")?;
    writeln!(temp_file, "John Doe,john@example.com,555-1234,123-45-6789")?;
    writeln!(temp_file, "Jane Smith,jane@test.org,555-5678,987-65-4321")?;
    writeln!(temp_file, "Bob Wilson,bob@company.net,555-9012,456-78-9012")?;
    temp_file.flush()?;
    
    let temp_path = temp_file.path().to_path_buf();
    
    // Capture stdout
    let output = std::process::Command::new("cargo")
        .args(&[
            "run", 
            "--package", "datacloak-cli",
            "--",
            "obfuscate",
            "--file", temp_path.to_str().unwrap(),
            "--rows", "10",
            "--dry-run",
        ])
        .output()
        .expect("Failed to execute command");
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Verify dry run header is printed
    assert!(stdout.contains("DRY RUN MODE"));
    
    // Parse the JSON output
    let json_start = stdout.find('{').expect("JSON output not found");
    let json_str = &stdout[json_start..];
    let summary: Value = serde_json::from_str(json_str)?;
    
    // Verify the JSON structure
    assert_eq!(summary["mode"], "dry-run");
    assert!(summary["input_file"].as_str().unwrap().contains(temp_path.file_name().unwrap().to_str().unwrap()));
    assert_eq!(summary["records_processed"], 3);
    assert!(summary["tokens_generated"].as_u64().unwrap() > 0);
    assert!(summary["patterns_loaded"].as_u64().unwrap() > 0);
    assert!(summary["sample_obfuscation"].is_object());
    assert!(summary["pattern_types"].is_array());
    assert_eq!(summary["output_would_be_written_to"], "stdout");
    
    Ok(())
}

#[tokio::test]
async fn test_obfuscate_dry_run_with_output_path() -> Result<()> {
    // Create a temporary CSV file
    let mut temp_file = NamedTempFile::new()?;
    writeln!(temp_file, "email")?;
    writeln!(temp_file, "test@example.com")?;
    temp_file.flush()?;
    
    let temp_path = temp_file.path().to_path_buf();
    let output_path = PathBuf::from("/tmp/output.json");
    
    // Run with dry-run and output path
    let output = std::process::Command::new("cargo")
        .args(&[
            "run", 
            "--package", "datacloak-cli",
            "--",
            "obfuscate",
            "--file", temp_path.to_str().unwrap(),
            "--output", output_path.to_str().unwrap(),
            "--dry-run",
        ])
        .output()
        .expect("Failed to execute command");
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Parse JSON
    let json_start = stdout.find('{').expect("JSON output not found");
    let json_str = &stdout[json_start..];
    let summary: Value = serde_json::from_str(json_str)?;
    
    // Verify output path is included but file is not created
    assert_eq!(summary["output_would_be_written_to"], output_path.to_str().unwrap());
    assert!(!output_path.exists(), "Output file should not be created in dry-run mode");
    
    Ok(())
}

#[tokio::test]
async fn test_obfuscate_normal_mode_creates_file() -> Result<()> {
    // Create a temporary CSV file
    let mut temp_file = NamedTempFile::new()?;
    writeln!(temp_file, "name,email")?;
    writeln!(temp_file, "Test User,user@test.com")?;
    temp_file.flush()?;
    
    let temp_path = temp_file.path().to_path_buf();
    let output_file = NamedTempFile::new()?;
    let output_path = output_file.path().to_path_buf();
    
    // Run without dry-run
    let _output = std::process::Command::new("cargo")
        .args(&[
            "run", 
            "--package", "datacloak-cli",
            "--",
            "obfuscate",
            "--file", temp_path.to_str().unwrap(),
            "--output", output_path.to_str().unwrap(),
            "--rows", "1",
        ])
        .output()
        .expect("Failed to execute command");
    
    // Verify file was created
    assert!(output_path.exists(), "Output file should be created in normal mode");
    
    // Read and verify content
    let content = tokio::fs::read_to_string(&output_path).await?;
    let data: Value = serde_json::from_str(&content)?;
    assert!(data.is_array());
    
    Ok(())
}

#[tokio::test]
async fn test_obfuscate_dry_run_sample_data() -> Result<()> {
    // Create test data with known patterns
    let mut temp_file = NamedTempFile::new()?;
    writeln!(temp_file, "data")?;
    writeln!(temp_file, "My email is contact@example.com and SSN is 111-22-3333")?;
    temp_file.flush()?;
    
    let temp_path = temp_file.path().to_path_buf();
    
    // Run dry-run
    let output = std::process::Command::new("cargo")
        .args(&[
            "run", 
            "--package", "datacloak-cli",
            "--",
            "obfuscate",
            "--file", temp_path.to_str().unwrap(),
            "--dry-run",
        ])
        .output()
        .expect("Failed to execute command");
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Parse JSON
    let json_start = stdout.find('{').expect("JSON output not found");
    let json_str = &stdout[json_start..];
    let summary: Value = serde_json::from_str(json_str)?;
    
    // Verify sample obfuscation contains tokens
    let sample = &summary["sample_obfuscation"];
    assert!(sample.is_object());
    
    // The obfuscated data should have placeholder tokens
    let data_str = sample["data"]["data"].as_str().unwrap_or("");
    assert!(!data_str.contains("contact@example.com"), "Email should be obfuscated");
    assert!(!data_str.contains("111-22-3333"), "SSN should be obfuscated");
    
    Ok(())
}

