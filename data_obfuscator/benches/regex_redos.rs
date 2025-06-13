use criterion::{criterion_group, criterion_main, Criterion};
use regex::Regex;
use std::hint::black_box;
use std::time::Duration;

// OWASP ReDoS attack patterns designed to test regex performance
fn get_redos_corpus() -> Vec<String> {
    vec![
        // Email ReDoS patterns
        format!("a@{}", "a".repeat(50000)),
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa@b.com".to_string(),
        format!("user@{}.com", "a".repeat(10000)),
        
        // Nested quantifier attacks
        format!("{}X", "a".repeat(20000)),
        format!("{}b{}", "a".repeat(15000), "a".repeat(15000)),
        
        // Alternation attacks
        format!("{}|{}", "a".repeat(10000), "b".repeat(10000)),
        
        // Catastrophic backtracking patterns
        format!("{}{}", "a?".repeat(30), "a".repeat(30)),
        
        // SSN pattern attacks
        format!("{}-{}-{}", "1".repeat(10), "2".repeat(10), "3".repeat(10)),
        format!("123-45-678{}", "9".repeat(5000)),
        
        // Long valid patterns to ensure normal operation isn't affected
        "user.name+tag@example-domain.co.uk".to_string(),
        "123-45-6789".to_string(),
        "normal text with no patterns".to_string(),
        format!("{}user@example.com more words", "word ".repeat(1000)),
    ]
}

fn benchmark_email_regex(c: &mut Criterion) {
    let email_pattern = r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b";
    let regex = Regex::new(email_pattern).unwrap();
    let corpus = get_redos_corpus();
    
    let mut group = c.benchmark_group("email_regex");
    // Set timeout to 1ms as required
    group.measurement_time(Duration::from_millis(1000));
    group.sample_size(100);
    
    for (i, input) in corpus.iter().enumerate() {
        group.bench_function(&format!("email_pattern_{}", i), |b| {
            b.iter(|| {
                let start = std::time::Instant::now();
                let _ = regex.find_iter(black_box(input)).count();
                let elapsed = start.elapsed();
                
                // Fail if any single pattern takes longer than 1ms
                if elapsed > Duration::from_millis(1) {
                    panic!("Pattern took {}ms (>1ms) for input length {}: {}", 
                           elapsed.as_millis(), input.len(),
                           if input.len() > 100 { &input[..100] } else { input });
                }
            });
        });
    }
    group.finish();
}

fn benchmark_ssn_regex(c: &mut Criterion) {
    let ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b";
    let regex = Regex::new(ssn_pattern).unwrap();
    let corpus = get_redos_corpus();
    
    let mut group = c.benchmark_group("ssn_regex");
    group.measurement_time(Duration::from_millis(1000));
    group.sample_size(100);
    
    for (i, input) in corpus.iter().enumerate() {
        group.bench_function(&format!("ssn_pattern_{}", i), |b| {
            b.iter(|| {
                let start = std::time::Instant::now();
                let _ = regex.find_iter(black_box(input)).count();
                let elapsed = start.elapsed();
                
                // Fail if any single pattern takes longer than 1ms
                if elapsed > Duration::from_millis(1) {
                    panic!("SSN pattern took {}ms (>1ms) for input length {}: {}", 
                           elapsed.as_millis(), input.len(),
                           if input.len() > 100 { &input[..100] } else { input });
                }
            });
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_email_regex, benchmark_ssn_regex);
criterion_main!(benches);