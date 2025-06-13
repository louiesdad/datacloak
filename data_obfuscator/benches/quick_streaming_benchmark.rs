use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use data_obfuscator::config::Rule;
use data_obfuscator::obfuscator::{Obfuscator, StreamConfig};
use std::io::Cursor;
use tokio::runtime::Runtime;
use std::time::Duration;

fn create_test_data(size_mb: usize) -> Vec<u8> {
    let line = "Line with email user@test.com and ssn 123-45-6789 and more data.\n";
    let line_bytes = line.as_bytes();
    let target_size = size_mb * 1024 * 1024;
    let lines_needed = target_size / line_bytes.len();
    
    let mut data = Vec::with_capacity(target_size);
    for i in 0..lines_needed {
        let varied_line = format!("Line {} with email user{}@test{}.com and ssn {}-{}-{} and more data.\n", 
                                 i, i % 100, i % 10, 100 + (i % 900), 10 + (i % 90), 1000 + (i % 9000));
        data.extend_from_slice(varied_line.as_bytes());
    }
    
    data.truncate(target_size);
    data
}

fn quick_streaming_benchmark(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
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
    
    // Test with 10MB for quick results
    let test_data = create_test_data(10);
    
    let chunk_sizes = vec![
        (8 * 1024, "8KB"),
        (256 * 1024, "256KB"),
        (1024 * 1024, "1MB"),
    ];
    
    let mut group = c.benchmark_group("streaming_10MB_quick");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));
    
    for (chunk_size, chunk_name) in chunk_sizes {
        group.bench_with_input(
            BenchmarkId::new("chunk", chunk_name),
            &chunk_size,
            |b, &chunk_size| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut obfuscator = Obfuscator::new(&rules).unwrap();
                        let reader = Cursor::new(&test_data);
                        let mut output = Vec::new();
                        let config = StreamConfig { chunk_size };
                        
                        obfuscator.stream_file(reader, &mut output, &config).await.unwrap();
                        output.len()
                    })
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = quick_streaming_benchmark
);
criterion_main!(benches);