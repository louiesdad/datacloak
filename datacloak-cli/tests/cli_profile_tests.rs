use assert_cmd::Command;
use predicates::prelude::*;
use serde_json;
use tempfile::NamedTempFile;
use std::io::Write;

#[derive(Debug, serde::Deserialize)]
struct ProfileOutput {
    candidates: Vec<ColumnCandidate>,
}

#[derive(Debug, serde::Deserialize)]
struct ColumnCandidate {
    name: String,
    score: f32,
    ml_probability: f32,
    graph_score: f32,
    column_type: String,
}

#[test]
fn test_cli_profile_command() {
    // Create test CSV file
    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, "id,description,price,category").unwrap();
    writeln!(file, "1,This is a great product with many features,99.99,electronics").unwrap();
    writeln!(file, "2,Another excellent item that customers love,149.99,home").unwrap();
    writeln!(file, "3,High quality and durable construction,79.99,tools").unwrap();
    file.flush().unwrap();
    
    let mut cmd = Command::cargo_bin("datacloak-cli").unwrap();
    cmd.arg("profile")
       .arg("--file").arg(file.path())
       .arg("--output").arg("json");
    
    let output = cmd.output().unwrap();
    assert!(output.status.success());
    
    let result: ProfileOutput = serde_json::from_slice(&output.stdout).unwrap();
    assert!(result.candidates.len() > 0);
    assert!(result.candidates[0].score >= result.candidates[1].score);
    
    // Description should be identified as text-heavy
    let description_col = result.candidates.iter()
        .find(|c| c.name == "description")
        .expect("Should find description column");
    assert!(description_col.score > 0.7);
}

#[test]
fn test_cli_multi_field_analyze() {
    // Create test CSV file with multiple text fields
    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, "id,description,comments,feedback,price").unwrap();
    writeln!(file, "1,Great product,Love it,Very satisfied,99.99").unwrap();
    writeln!(file, "2,Could be better,Has issues,Not happy,149.99").unwrap();
    writeln!(file, "3,Excellent quality,Recommend,Will buy again,79.99").unwrap();
    file.flush().unwrap();
    
    let mut cmd = Command::cargo_bin("datacloak-cli").unwrap();
    cmd.arg("analyze")
       .arg("--file").arg(file.path())
       .arg("--columns").arg("description,comments,feedback")
       .arg("--output").arg("csv");
    
    let output = cmd.output().unwrap();
    assert!(output.status.success());
    
    // Verify CSV has all columns
    let csv_output = String::from_utf8(output.stdout).unwrap();
    assert!(csv_output.contains("description_sentiment"));
    assert!(csv_output.contains("comments_sentiment"));
    assert!(csv_output.contains("feedback_sentiment"));
}

#[test]
fn test_cli_auto_discovery_mode() {
    // Create test CSV file
    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, "id,customer_feedback,order_notes,sku,quantity").unwrap();
    writeln!(file, "1,The product exceeded my expectations,Please ship ASAP,SKU123,5").unwrap();
    writeln!(file, "2,Poor quality and slow delivery,Handle with care,SKU456,2").unwrap();
    writeln!(file, "3,Amazing service and great value,Gift wrap please,SKU789,1").unwrap();
    file.flush().unwrap();
    
    let mut cmd = Command::cargo_bin("datacloak-cli").unwrap();
    cmd.arg("analyze")
       .arg("--file").arg(file.path())
       .arg("--auto-discover")
       .arg("--threshold").arg("0.7");
    
    let output = cmd.output().unwrap();
    assert!(output.status.success());
    
    // Should analyze high-scoring columns automatically
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("Auto-discovered"));
    assert!(stderr.contains("columns for analysis"));
}

#[test]
fn test_cli_dry_run_with_estimate() {
    // Create test CSV file
    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, "col1,col2,col3,col4,col5").unwrap();
    for i in 0..100 {
        writeln!(file, "Text {},More text {},Data {},Info {},Value {}", i, i, i, i, i).unwrap();
    }
    file.flush().unwrap();
    
    let mut cmd = Command::cargo_bin("datacloak-cli").unwrap();
    cmd.arg("analyze")
       .arg("--file").arg(file.path())
       .arg("--columns").arg("col1,col2,col3")
       .arg("--dry-run");
    
    let output = cmd.output().unwrap();
    assert!(output.status.success());
    
    #[derive(Debug, serde::Deserialize)]
    struct DryRunOutput {
        estimated_time_seconds: u64,
        estimated_cost_usd: f64,
        selected_columns: Vec<String>,
    }
    
    let result: DryRunOutput = serde_json::from_slice(&output.stdout).unwrap();
    assert!(result.estimated_time_seconds > 0);
    assert!(result.estimated_cost_usd > 0.0);
    assert_eq!(result.selected_columns.len(), 3);
}

#[test]
fn test_profile_with_ml_only_flag() {
    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, "id,text_field,numeric_field").unwrap();
    writeln!(file, "1,Long descriptive text here,100").unwrap();
    writeln!(file, "2,Another text sample,200").unwrap();
    file.flush().unwrap();
    
    let mut cmd = Command::cargo_bin("datacloak-cli").unwrap();
    cmd.arg("profile")
       .arg("--file").arg(file.path())
       .arg("--ml-only");
    
    let output = cmd.output().unwrap();
    assert!(output.status.success());
    
    let result: ProfileOutput = serde_json::from_slice(&output.stdout).unwrap();
    // When ML-only, graph scores should be 0
    for candidate in &result.candidates {
        assert_eq!(candidate.graph_score, 0.0);
    }
}

#[test]
fn test_profile_with_graph_only_flag() {
    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, "id,comments,reviews").unwrap();
    writeln!(file, "1,Great feedback,Excellent review").unwrap();
    writeln!(file, "2,More comments,Another review").unwrap();
    file.flush().unwrap();
    
    let mut cmd = Command::cargo_bin("datacloak-cli").unwrap();
    cmd.arg("profile")
       .arg("--file").arg(file.path())
       .arg("--graph-only");
    
    let output = cmd.output().unwrap();
    assert!(output.status.success());
    
    let result: ProfileOutput = serde_json::from_slice(&output.stdout).unwrap();
    // When graph-only, ML probabilities should be 0
    for candidate in &result.candidates {
        assert_eq!(candidate.ml_probability, 0.0);
    }
}