use criterion::{criterion_group, criterion_main, Criterion};
use data_obfuscator::secure_obfuscator::SecureObfuscator;
use data_obfuscator::config::Rule;
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

fn benchmark_secure_obfuscator(c: &mut Criterion) {
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
    
    let corpus = get_redos_corpus();
    
    let mut group = c.benchmark_group("secure_obfuscator");
    group.measurement_time(Duration::from_millis(1000));
    group.sample_size(100);
    
    for (i, input) in corpus.iter().enumerate() {
        group.bench_function(&format!("secure_pattern_{}", i), |b| {
            b.iter(|| {
                let start = std::time::Instant::now();
                let mut obfuscator = SecureObfuscator::new(&rules).unwrap();
                let _ = obfuscator.obfuscate_text(black_box(input));
                let elapsed = start.elapsed();
                
                // Fail if any single pattern takes longer than 1ms
                if elapsed > Duration::from_millis(1) {
                    panic!("Secure obfuscator took {}ms (>1ms) for input length {}: {}", 
                           elapsed.as_millis(), input.len(),
                           if input.len() > 100 { &input[..100] } else { input });
                }
            });
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_secure_obfuscator);
criterion_main!(benches);