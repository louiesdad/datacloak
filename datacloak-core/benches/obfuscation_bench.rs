//! Performance benchmarks for DataCloak obfuscation

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use datacloak_core::{Obfuscator, Pattern, PatternType};
use serde_json::json;

fn create_test_data(size: usize) -> Vec<serde_json::Value> {
    (0..size)
        .map(|i| {
            json!({
                "id": i,
                "email": format!("user{}@example.com", i),
                "phone": format!("555-{:03}-{:04}", i % 1000, i % 10000),
                "ssn": format!("{:03}-{:02}-{:04}", i % 1000, i % 100, i % 10000),
                "credit_card": format!("4532{:012}", i),
                "ip_address": format!("192.168.{}.{}", i % 256, (i * 7) % 256),
                "notes": format!("Customer {} can be reached at user{}@example.com or 555-{:03}-{:04}", 
                    i, i, i % 1000, i % 10000),
            })
        })
        .collect()
}

fn bench_obfuscation(c: &mut Criterion) {
    let obfuscator = Obfuscator::new();
    
    // Set up patterns
    let patterns = vec![
        Pattern::new(PatternType::Email, r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b".to_string()),
        Pattern::new(PatternType::Phone, r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b".to_string()),
        Pattern::new(PatternType::SSN, r"\b\d{3}-\d{2}-\d{4}\b".to_string()),
        Pattern::new(PatternType::CreditCard, r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b".to_string()),
        Pattern::new(PatternType::IPAddress, r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b".to_string()),
    ];
    
    obfuscator.set_patterns(patterns).unwrap();
    
    let mut group = c.benchmark_group("obfuscation");
    
    // Benchmark different batch sizes
    for size in [10, 100, 1000, 10000].iter() {
        let data = create_test_data(*size);
        
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let result = obfuscator.obfuscate_batch(black_box(&data)).unwrap();
                black_box(result);
            });
        });
    }
    
    group.finish();
}

fn bench_pattern_matching(c: &mut Criterion) {
    use regex::Regex;
    
    let email_pattern = Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b").unwrap();
    let text = "Contact us at support@example.com, sales@example.com, or info@example.com for more information.";
    
    c.bench_function("regex_find_all", |b| {
        b.iter(|| {
            let matches: Vec<_> = email_pattern.find_iter(black_box(text)).collect();
            black_box(matches);
        });
    });
}

fn bench_json_processing(c: &mut Criterion) {
    let obfuscator = Obfuscator::new();
    let patterns = vec![
        Pattern::new(PatternType::Email, r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b".to_string()),
    ];
    obfuscator.set_patterns(patterns).unwrap();
    
    let nested_json = json!({
        "user": {
            "profile": {
                "email": "user@example.com",
                "contacts": [
                    {"type": "work", "email": "work@example.com"},
                    {"type": "personal", "email": "personal@example.com"}
                ]
            }
        }
    });
    
    c.bench_function("nested_json_obfuscation", |b| {
        b.iter(|| {
            let batch = vec![nested_json.clone()];
            let result = obfuscator.obfuscate_batch(black_box(&batch)).unwrap();
            black_box(result);
        });
    });
}

fn bench_cache_operations(c: &mut Criterion) {
    use datacloak_core::ObfuscationCache;
    
    let cache = ObfuscationCache::new();
    
    // Pre-populate cache
    for i in 0..1000 {
        cache.store(
            format!("TOKEN-{}", i),
            format!("original-{}", i),
            "EMAIL".to_string(),
        );
    }
    
    c.bench_function("cache_lookup", |b| {
        b.iter(|| {
            let token = black_box("TOKEN-500");
            let result = cache.get_original(token);
            black_box(result);
        });
    });
    
    c.bench_function("cache_store", |b| {
        let mut counter = 1000;
        b.iter(|| {
            cache.store(
                format!("TOKEN-{}", counter),
                format!("original-{}", counter),
                "EMAIL".to_string(),
            );
            counter += 1;
        });
    });
}

criterion_group!(
    benches,
    bench_obfuscation,
    bench_pattern_matching,
    bench_json_processing,
    bench_cache_operations
);
criterion_main!(benches);
