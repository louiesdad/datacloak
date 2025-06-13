use criterion::{black_box, criterion_group, criterion_main, Criterion};
use datacloak_core::{
    obfuscator::Obfuscator,
    patterns::{Pattern, PatternType},
};
use serde_json::json;

fn bench_obfuscate(c: &mut Criterion) {
    let obfuscator = Obfuscator::new();

    // Set up patterns
    let patterns = vec![
        Pattern::new(
            PatternType::Email,
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}".to_string(),
        ),
        Pattern::new(
            PatternType::CreditCard,
            r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b".to_string(),
        ),
        Pattern::new(PatternType::SSN, r"\b\d{3}-\d{2}-\d{4}\b".to_string()),
        Pattern::new(
            PatternType::Phone,
            r"\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b".to_string(),
        ),
    ];

    obfuscator.set_patterns(patterns).unwrap();

    let test_data = vec![
        json!({
            "text": "My email is test@example.com and my SSN is 123-45-6789"
        }),
        json!({
            "text": "Call me at (555) 123-4567 or use my credit card 4111-1111-1111-1111"
        }),
    ];

    c.bench_function("obfuscate_small_texts", |b| {
        b.iter(|| {
            let _ = obfuscator.obfuscate_batch(black_box(&test_data));
        })
    });
}

criterion_group!(benches, bench_obfuscate);
criterion_main!(benches);
