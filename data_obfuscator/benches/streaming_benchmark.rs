use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use data_obfuscator::config::Rule;
use data_obfuscator::obfuscator::{Obfuscator, StreamConfig};
use std::io::Cursor;
use tokio::runtime::Runtime;
use std::time::Duration;

// Create test data with email patterns to obfuscate
fn create_test_data(size_mb: usize) -> Vec<u8> {
    let line = "This is a test line with email test@example.com and data.\n";
    let line_bytes = line.as_bytes();
    let target_size = size_mb * 1024 * 1024;
    let lines_needed = target_size / line_bytes.len();
    
    let mut data = Vec::with_capacity(target_size);
    for i in 0..lines_needed {
        // Vary the email to create realistic data
        let varied_line = format!("This is test line {} with email user{}@domain{}.com and more data.\n", i, i % 1000, i % 50);
        data.extend_from_slice(varied_line.as_bytes());
    }
    
    // Fill to exact size
    while data.len() < target_size {
        data.push(b' ');
    }
    data.truncate(target_size);
    
    data
}

fn bench_streaming_chunk_sizes(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    // Create rules for obfuscation
    let rules = vec![
        Rule {
            pattern: r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b".to_string(),
            label: "EMAIL".to_string(),
        },
        Rule {
            pattern: r"\b\d{3}-\d{2}-\d{4}\b".to_string(),
            label: "SSN".to_string(),
        },
    ];
    
    // Test with different file sizes first (smaller for CI)
    let test_sizes = vec![
        (1, "1MB"),
        (10, "10MB"), 
        (100, "100MB"),
        (500, "500MB"), // Use 500MB instead of 1GB for faster benchmarks
    ];
    
    // Different chunk sizes to test
    let chunk_sizes = vec![
        (8 * 1024, "8KB"),
        (64 * 1024, "64KB"),  
        (256 * 1024, "256KB"),
        (1024 * 1024, "1MB"),
        (4 * 1024 * 1024, "4MB"),
    ];
    
    for (size_mb, size_name) in test_sizes {
        let test_data = create_test_data(size_mb);
        println!("Created test data: {} ({} bytes)", size_name, test_data.len());
        
        let mut group = c.benchmark_group(format!("streaming_{}", size_name));
        group.sample_size(10); // Reduce sample size for large data
        group.measurement_time(Duration::from_secs(30)); // Longer measurement time
        
        for (chunk_size, chunk_name) in &chunk_sizes {
            group.bench_with_input(
                BenchmarkId::new("chunk_size", chunk_name),
                chunk_size,
                |b, &chunk_size| {
                    b.iter(|| {
                        rt.block_on(async {
                            let mut obfuscator = Obfuscator::new(&rules).unwrap();
                            let reader = Cursor::new(&test_data);
                            let mut output = Vec::new();
                            let config = StreamConfig { chunk_size };
                            
                            obfuscator.stream_file(reader, &mut output, &config).await.unwrap();
                            output.len() // Return something to prevent optimization
                        })
                    });
                },
            );
        }
        group.finish();
    }
}

fn bench_throughput_comparison(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let rules = vec![
        Rule {
            pattern: r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b".to_string(),
            label: "EMAIL".to_string(),
        },
    ];
    
    // Fixed 100MB test for throughput measurement
    let test_data = create_test_data(100);
    
    let mut group = c.benchmark_group("throughput_100MB");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));
    
    // Test key chunk sizes for throughput analysis
    let key_chunk_sizes = vec![
        (8 * 1024, "8KB"),
        (256 * 1024, "256KB"), // Default
        (1024 * 1024, "1MB"),
        (4 * 1024 * 1024, "4MB"),
    ];
    
    for (chunk_size, chunk_name) in key_chunk_sizes {
        group.bench_function(
            chunk_name,
            |b| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut obfuscator = Obfuscator::new(&rules).unwrap();
                        let reader = Cursor::new(&test_data);
                        let mut output = Vec::new();
                        let config = StreamConfig { chunk_size };
                        
                        let start = std::time::Instant::now();
                        obfuscator.stream_file(reader, &mut output, &config).await.unwrap();
                        let duration = start.elapsed();
                        
                        // Calculate throughput in MB/s
                        let mb_processed = test_data.len() as f64 / (1024.0 * 1024.0);
                        let throughput = mb_processed / duration.as_secs_f64();
                        
                        throughput
                    })
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .significance_level(0.1)
        .sample_size(10);
    targets = bench_streaming_chunk_sizes, bench_throughput_comparison
);
criterion_main!(benches);