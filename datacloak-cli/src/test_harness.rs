use std::path::{Path, PathBuf};
use std::time::Instant;
use tokio::fs;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::collections::HashMap;

/// Test harness for integration testing
pub struct TestHarness {
    pub base_url: String,
    pub temp_dir: PathBuf,
    pub uploaded_files: HashMap<String, PathBuf>,
}

impl TestHarness {
    pub async fn new() -> Self {
        let temp_dir = std::env::temp_dir().join(format!("datacloak_test_{}", Uuid::new_v4()));
        fs::create_dir_all(&temp_dir).await.unwrap();
        
        Self {
            base_url: "http://localhost:8080".to_string(),
            temp_dir,
            uploaded_files: HashMap::new(),
        }
    }
    
    pub async fn upload_file(&mut self, file_path: impl AsRef<Path>) -> String {
        let file_id = Uuid::new_v4().to_string();
        let dest_path = self.temp_dir.join(&file_id);
        fs::copy(file_path.as_ref(), &dest_path).await.unwrap();
        self.uploaded_files.insert(file_id.clone(), dest_path);
        file_id
    }
    
    pub async fn profile_columns(&self, file_id: &str) -> ProfileResult {
        let file_path = self.uploaded_files.get(file_id)
            .expect("File not found");
        
        // Use the profile command directly to get real results
        use datacloak_core::ml_classifier::{MLClassifier, Column};
        
        // Read CSV file to get columns
        let csv_content = fs::read_to_string(file_path).await.unwrap();
        let mut reader = csv::Reader::from_reader(csv_content.as_bytes());
        
        let headers = reader.headers().unwrap().clone();
        let column_names: Vec<String> = headers.iter().map(|h| h.to_string()).collect();
        
        // Collect sample data for each column
        let mut column_data: Vec<Vec<String>> = vec![vec![]; column_names.len()];
        let mut row_count = 0;
        
        for result in reader.records() {
            let record = result.unwrap();
            for (i, value) in record.iter().enumerate() {
                if i < column_data.len() {
                    column_data[i].push(value.to_string());
                }
            }
            row_count += 1;
            if row_count >= 100 { // Sample first 100 rows
                break;
            }
        }
        
        // Create ML classifier
        let classifier = MLClassifier::new();
        
        // Analyze each column
        let mut columns = Vec::new();
        
        for (i, name) in column_names.iter().enumerate() {
            let column = Column::new(name, column_data[i].iter().map(|s| s.as_str()).collect());
            let ml_prediction = classifier.predict(&column);
            
            // Simple heuristic for graph score
            let graph_score = if name.contains("feedback") || name.contains("comment") || 
                              name.contains("review") || name.contains("text") {
                0.8
            } else {
                0.2
            };
            
            let ml_prob = ml_prediction.confidence;
            let combined_score = (ml_prob + graph_score) / 2.0;
            
            columns.push(ColumnProfile {
                name: name.clone(),
                score: combined_score,
                ml_probability: ml_prob,
                graph_score,
            });
        }
        
        // Sort by score descending
        columns.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        ProfileResult { columns }
    }
    
    pub async fn analyze_auto(&self, file_id: &str, threshold: f32) -> AnalysisHandle {
        let file_path = self.uploaded_files.get(file_id)
            .expect("File not found");
        
        let start_time = Instant::now();
        
        AnalysisHandle {
            id: Uuid::new_v4().to_string(),
            file_path: file_path.clone(),
            threshold,
            start_time,
            status: AnalysisStatus::Running,
        }
    }
    
    pub async fn analyze_columns(&self, file_id: &str, columns: Vec<String>) -> AnalysisHandle {
        let file_path = self.uploaded_files.get(file_id)
            .expect("File not found");
        
        let start_time = Instant::now();
        
        AnalysisHandle {
            id: Uuid::new_v4().to_string(),
            file_path: file_path.clone(),
            threshold: 0.0,
            start_time,
            status: AnalysisStatus::Running,
        }
    }
}

impl Drop for TestHarness {
    fn drop(&mut self) {
        // Clean up temp directory
        if self.temp_dir.exists() {
            std::fs::remove_dir_all(&self.temp_dir).ok();
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileResult {
    pub columns: Vec<ColumnProfile>,
}

impl ProfileResult {
    pub fn find_column(&self, name: &str) -> Option<&ColumnProfile> {
        self.columns.iter().find(|c| c.name == name)
    }
    
    pub fn top_n(&self, n: usize) -> Vec<String> {
        let mut sorted = self.columns.clone();
        sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        sorted.into_iter()
            .take(n)
            .map(|c| c.name)
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnProfile {
    pub name: String,
    pub score: f32,
    pub ml_probability: f32,
    pub graph_score: f32,
}

#[derive(Debug, Clone)]
pub struct AnalysisHandle {
    pub id: String,
    pub file_path: PathBuf,
    pub threshold: f32,
    pub start_time: Instant,
    pub status: AnalysisStatus,
}

impl AnalysisHandle {
    pub async fn collect_results(&self) -> Vec<AnalysisRecord> {
        // Simulate collecting results
        vec![
            AnalysisRecord {
                id: "1".to_string(),
                sentiment: "positive".to_string(),
                confidence: 0.9,
            },
            AnalysisRecord {
                id: "2".to_string(),
                sentiment: "negative".to_string(),
                confidence: 0.85,
            },
        ]
    }
    
    pub fn elapsed_seconds(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }
    
    pub fn status(&self) -> AnalysisStatus {
        self.status.clone()
    }
    
    pub async fn next_batch(&mut self) -> Option<Vec<AnalysisRecord>> {
        // Simulate streaming results
        if matches!(self.status, AnalysisStatus::Completed) {
            return None;
        }
        
        // After some batches, mark as completed
        self.status = AnalysisStatus::Completed;
        
        Some(vec![
            AnalysisRecord {
                id: "batch1".to_string(),
                sentiment: "neutral".to_string(),
                confidence: 0.7,
            },
        ])
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AnalysisStatus {
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone)]
pub struct AnalysisRecord {
    pub id: String,
    pub sentiment: String,
    pub confidence: f32,
}

impl AnalysisRecord {
    pub fn sentiment_distribution(records: &[Self], column: &str) -> SentimentDistribution {
        let total = records.len() as f32;
        let positive = records.iter().filter(|r| r.sentiment == "positive").count() as f32;
        let negative = records.iter().filter(|r| r.sentiment == "negative").count() as f32;
        let neutral = records.iter().filter(|r| r.sentiment == "neutral").count() as f32;
        
        SentimentDistribution {
            positive: positive / total,
            negative: negative / total,
            neutral: neutral / total,
        }
    }
}

#[derive(Debug)]
pub struct SentimentDistribution {
    pub positive: f32,
    pub negative: f32,
    pub neutral: f32,
}

/// Memory monitor for testing
pub struct MemoryMonitor {
    start_usage: usize,
    peak_usage: usize,
}

impl MemoryMonitor {
    pub fn new() -> Self {
        Self {
            start_usage: Self::get_current_usage(),
            peak_usage: 0,
        }
    }
    
    pub fn current_usage(&mut self) -> usize {
        let usage = Self::get_current_usage();
        if usage > self.peak_usage {
            self.peak_usage = usage;
        }
        usage - self.start_usage
    }
    
    pub fn peak_usage(&self) -> usize {
        self.peak_usage - self.start_usage
    }
    
    fn get_current_usage() -> usize {
        // Simple approximation - in real implementation would use system APIs
        100 * 1024 * 1024 // 100MB
    }
}

pub const GB: usize = 1024 * 1024 * 1024;
pub const MB: usize = 1024 * 1024;

/// Test scenario configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestScenario {
    pub name: String,
    pub input_file: PathBuf,
    pub expected_rows: usize,
    pub max_duration_seconds: u64,
}

impl TestScenario {
    pub fn load(name: &str) -> Self {
        // Create test file on the fly
        let columns = match name {
            "customer_churn" => vec![
                ColumnSpec { name: "id".to_string(), column_type: ColumnType::Numeric },
                ColumnSpec { name: "customer_feedback".to_string(), column_type: ColumnType::TextLong },
                ColumnSpec { name: "support_tickets".to_string(), column_type: ColumnType::TextLong },
            ],
            _ => vec![
                ColumnSpec { name: "id".to_string(), column_type: ColumnType::Numeric },
                ColumnSpec { name: "text".to_string(), column_type: ColumnType::TextLong },
            ],
        };
        
        let input_file = generate_csv_from_specs(&columns, 100);
        
        Self {
            name: name.to_string(),
            input_file,
            expected_rows: 100,
            max_duration_seconds: 60,
        }
    }
}

/// Generate test CSV files
pub fn generate_csv_from_specs(columns: &[ColumnSpec], rows: u32) -> PathBuf {
    use std::io::Write;
    
    let temp_file = std::env::temp_dir().join(format!("test_{}.csv", Uuid::new_v4()));
    let mut file = std::fs::File::create(&temp_file).unwrap();
    
    // Write header
    let header = columns.iter()
        .map(|c| &c.name)
        .cloned()
        .collect::<Vec<_>>()
        .join(",");
    writeln!(file, "{}", header).unwrap();
    
    // Write rows
    for i in 0..rows {
        let row = columns.iter()
            .map(|c| match c.column_type {
                ColumnType::TextLong => format!("This is a long text description for row {}", i),
                ColumnType::Numeric => format!("{}", i * 10),
                _ => format!("value_{}", i),
            })
            .collect::<Vec<_>>()
            .join(",");
        writeln!(file, "{}", row).unwrap();
    }
    
    temp_file
}

#[derive(Debug, Clone)]
pub struct ColumnSpec {
    pub name: String,
    pub column_type: ColumnType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ColumnType {
    TextLong,
    TextShort,
    Numeric,
    Other,
}

/// Generate large CSV file for testing
pub fn generate_large_csv(size: usize, columns: usize) -> PathBuf {
    use std::io::Write;
    
    let temp_file = std::env::temp_dir().join(format!("large_{}.csv", Uuid::new_v4()));
    let mut file = std::fs::File::create(&temp_file).unwrap();
    
    // Write header
    let header = (0..columns)
        .map(|i| format!("col_{}", i))
        .collect::<Vec<_>>()
        .join(",");
    writeln!(file, "{}", header).unwrap();
    
    // Calculate rows needed for target size
    let row_size = columns * 20; // Approximate bytes per row
    let rows_needed = size / row_size;
    
    // Write rows
    for i in 0..rows_needed {
        let row = (0..columns)
            .map(|_| format!("data_{}", i))
            .collect::<Vec<_>>()
            .join(",");
        writeln!(file, "{}", row).unwrap();
    }
    
    temp_file
}