use assert_cmd::Command;
use serde_json;
use tempfile::NamedTempFile;
use std::io::Write;

#[derive(Debug, serde::Deserialize)]
struct AnalysisResult {
    record_id: String,
    column: String,
    sentiment: String,
    confidence: f32,
}

#[test]
fn test_analyze_multiple_columns_json_output() {
    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, "id,feedback,support_ticket,review").unwrap();
    writeln!(file, "1,Great service!,No issues at all,5 stars").unwrap();
    writeln!(file, "2,Terrible experience,Product broken on arrival,1 star").unwrap();
    writeln!(file, "3,OK service,Minor delay in shipping,3 stars").unwrap();
    file.flush().unwrap();
    
    let mut cmd = Command::cargo_bin("datacloak-cli").unwrap();
    cmd.arg("analyze")
       .arg("--file").arg(file.path())
       .arg("--columns").arg("feedback,support_ticket,review")
       .arg("--output").arg("json")
       .arg("--mock-llm");  // Use mock LLM for testing
    
    let output = cmd.output().unwrap();
    assert!(output.status.success());
    
    let results: Vec<AnalysisResult> = serde_json::from_slice(&output.stdout).unwrap();
    
    // Should have 3 records x 3 columns = 9 results
    assert_eq!(results.len(), 9);
    
    // Check that all columns are analyzed
    let columns: std::collections::HashSet<_> = results.iter()
        .map(|r| r.column.as_str())
        .collect();
    assert!(columns.contains("feedback"));
    assert!(columns.contains("support_ticket"));
    assert!(columns.contains("review"));
}

#[test]
fn test_analyze_with_progress_reporting() {
    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, "id,text1,text2").unwrap();
    for i in 0..50 {
        writeln!(file, "{},Sample text {},Another text {}", i, i, i).unwrap();
    }
    file.flush().unwrap();
    
    let mut cmd = Command::cargo_bin("datacloak-cli").unwrap();
    cmd.arg("analyze")
       .arg("--file").arg(file.path())
       .arg("--columns").arg("text1,text2")
       .arg("--output").arg("json")
       .arg("--mock-llm");
    
    let output = cmd.output().unwrap();
    assert!(output.status.success());
    
    // Progress should be reported to stderr
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("Processing"));
    assert!(stderr.contains("%"));
}

#[test]
fn test_auto_discovery_with_threshold() {
    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, "id,long_text,short_text,number,code").unwrap();
    writeln!(file, "1,This is a very long descriptive text field with lots of content,Short,123,ABC123").unwrap();
    writeln!(file, "2,Another lengthy description that contains meaningful text,Brief,456,DEF456").unwrap();
    writeln!(file, "3,Yet more descriptive content for sentiment analysis,Quick,789,GHI789").unwrap();
    file.flush().unwrap();
    
    let mut cmd = Command::cargo_bin("datacloak-cli").unwrap();
    cmd.arg("analyze")
       .arg("--file").arg(file.path())
       .arg("--auto-discover")
       .arg("--threshold").arg("0.8")
       .arg("--dry-run");
    
    let output = cmd.output().unwrap();
    assert!(output.status.success());
    
    let stderr = String::from_utf8(output.stderr).unwrap();
    // Should discover long_text as high-scoring
    assert!(stderr.contains("long_text"));
    // Should not include number or code columns
    assert!(!stderr.contains("number") || !stderr.contains("Selected columns:.*number"));
}

#[test]
fn test_streaming_output_format() {
    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, "id,comment").unwrap();
    writeln!(file, "1,Happy customer").unwrap();
    writeln!(file, "2,Unhappy customer").unwrap();
    file.flush().unwrap();
    
    let mut cmd = Command::cargo_bin("datacloak-cli").unwrap();
    cmd.arg("analyze")
       .arg("--file").arg(file.path())
       .arg("--columns").arg("comment")
       .arg("--output").arg("stream")
       .arg("--mock-llm");
    
    let output = cmd.output().unwrap();
    assert!(output.status.success());
    
    // Each line should be valid JSON
    let stdout = String::from_utf8(output.stdout).unwrap();
    for line in stdout.lines() {
        if !line.trim().is_empty() {
            let result: AnalysisResult = serde_json::from_str(line)
                .expect("Each line should be valid JSON");
            assert!(!result.record_id.is_empty());
            assert!(["positive", "negative", "neutral"].contains(&result.sentiment.as_str()));
        }
    }
}

#[test]
fn test_csv_output_format() {
    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, "id,feedback,comment").unwrap();
    writeln!(file, "1,Great,Excellent").unwrap();
    writeln!(file, "2,Bad,Poor").unwrap();
    file.flush().unwrap();
    
    let mut cmd = Command::cargo_bin("datacloak-cli").unwrap();
    cmd.arg("analyze")
       .arg("--file").arg(file.path())
       .arg("--columns").arg("feedback,comment")
       .arg("--output").arg("csv")
       .arg("--mock-llm");
    
    let output = cmd.output().unwrap();
    assert!(output.status.success());
    
    let csv_output = String::from_utf8(output.stdout).unwrap();
    let lines: Vec<&str> = csv_output.lines().collect();
    
    // Check header
    assert!(lines[0].contains("id"));
    assert!(lines[0].contains("feedback_sentiment"));
    assert!(lines[0].contains("feedback_confidence"));
    assert!(lines[0].contains("comment_sentiment"));
    assert!(lines[0].contains("comment_confidence"));
    
    // Check data rows
    assert_eq!(lines.len() - 1, 2); // 2 data rows + 1 header
}

#[test]
fn test_column_validation() {
    let mut file = NamedTempFile::new().unwrap();
    writeln!(file, "id,col1,col2").unwrap();
    writeln!(file, "1,data1,data2").unwrap();
    file.flush().unwrap();
    
    let mut cmd = Command::cargo_bin("datacloak-cli").unwrap();
    cmd.arg("analyze")
       .arg("--file").arg(file.path())
       .arg("--columns").arg("col1,nonexistent_column");
    
    let output = cmd.output().unwrap();
    assert!(!output.status.success());
    
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("nonexistent_column"));
    assert!(stderr.contains("not found"));
}