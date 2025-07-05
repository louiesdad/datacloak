use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use datacloak_core::graph::{ColumnData, SimilarityCalculator};
use std::time::Instant;

fn generate_random_vector(size: usize) -> Vec<f32> {
    (0..size).map(|i| ((i * 17) % 100) as f32 / 100.0).collect()
}

fn generate_random_tokens(count: usize) -> Vec<String> {
    (0..count).map(|i| format!("token_{}", i % 20)).collect()
}

fn bench_cosine_similarity(c: &mut Criterion) {
    let calc = SimilarityCalculator::new();

    c.bench_function("cosine_similarity_small", |b| {
        let vec1 = generate_random_vector(10);
        let vec2 = generate_random_vector(10);
        b.iter(|| calc.cosine_similarity(black_box(&vec1), black_box(&vec2)));
    });

    c.bench_function("cosine_similarity_medium", |b| {
        let vec1 = generate_random_vector(300);
        let vec2 = generate_random_vector(300);
        b.iter(|| calc.cosine_similarity(black_box(&vec1), black_box(&vec2)));
    });

    c.bench_function("cosine_similarity_large", |b| {
        let vec1 = generate_random_vector(1024);
        let vec2 = generate_random_vector(1024);
        b.iter(|| calc.cosine_similarity(black_box(&vec1), black_box(&vec2)));
    });
}

fn bench_jaccard_similarity(c: &mut Criterion) {
    let calc = SimilarityCalculator::new();

    c.bench_function("jaccard_similarity_small", |b| {
        let set1 = vec!["hello", "world", "rust"];
        let set2 = vec!["hello", "programming", "rust"];
        b.iter(|| calc.jaccard_similarity(black_box(&set1), black_box(&set2)));
    });

    c.bench_function("jaccard_similarity_large", |b| {
        let tokens1: Vec<&str> = (0..100)
            .map(|i| match i % 20 {
                0 => "apple",
                1 => "banana",
                2 => "cherry",
                3 => "date",
                4 => "elderberry",
                5 => "fig",
                6 => "grape",
                7 => "honeydew",
                8 => "kiwi",
                9 => "lemon",
                10 => "mango",
                11 => "nectarine",
                12 => "orange",
                13 => "papaya",
                14 => "quince",
                15 => "raspberry",
                16 => "strawberry",
                17 => "tangerine",
                18 => "ugli",
                _ => "vanilla",
            })
            .collect();

        let tokens2: Vec<&str> = (0..100)
            .map(|i| match (i + 10) % 20 {
                0 => "apple",
                1 => "banana",
                2 => "cherry",
                3 => "date",
                4 => "elderberry",
                5 => "fig",
                6 => "grape",
                7 => "honeydew",
                8 => "kiwi",
                9 => "lemon",
                10 => "mango",
                11 => "nectarine",
                12 => "orange",
                13 => "papaya",
                14 => "quince",
                15 => "raspberry",
                16 => "strawberry",
                17 => "tangerine",
                18 => "ugli",
                _ => "vanilla",
            })
            .collect();

        b.iter(|| calc.jaccard_similarity(black_box(&tokens1), black_box(&tokens2)));
    });
}

fn bench_combined_similarity(c: &mut Criterion) {
    let calc = SimilarityCalculator::new();

    c.bench_function("combined_similarity", |b| {
        b.iter_batched(
            || {
                let col1 = ColumnData {
                    embedding: generate_random_vector(300),
                    tokens: generate_random_tokens(50),
                };
                let col2 = ColumnData {
                    embedding: generate_random_vector(300),
                    tokens: generate_random_tokens(50),
                };
                (col1, col2)
            },
            |(col1, col2)| calc.combined_similarity(black_box(&col1), black_box(&col2), 0.6, 0.4),
            BatchSize::SmallInput,
        );
    });
}

fn measure_similarity_performance() {
    println!("\n=== Similarity Performance Test ===");
    let calc = SimilarityCalculator::new();

    // Test cosine similarity at various scales
    for size in [100, 300, 1000, 3000].iter() {
        let vec1 = generate_random_vector(*size);
        let vec2 = generate_random_vector(*size);

        let start = Instant::now();
        let iterations = 10000;
        for _ in 0..iterations {
            calc.cosine_similarity(&vec1, &vec2);
        }
        let elapsed = start.elapsed();
        let per_call = elapsed.as_micros() as f64 / iterations as f64;

        println!("Cosine similarity ({}D): {:.3}µs per call", size, per_call);
        assert!(
            per_call < 100.0,
            "Cosine similarity too slow for {}D vectors",
            size
        );
    }

    // Test combined similarity
    let col1 = ColumnData {
        embedding: generate_random_vector(300),
        tokens: generate_random_tokens(100),
    };
    let col2 = ColumnData {
        embedding: generate_random_vector(300),
        tokens: generate_random_tokens(100),
    };

    let start = Instant::now();
    let iterations = 10000;
    for _ in 0..iterations {
        calc.combined_similarity(&col1, &col2, 0.6, 0.4);
    }
    let elapsed = start.elapsed();
    let per_call = elapsed.as_micros() as f64 / iterations as f64;

    println!("Combined similarity: {:.3}µs per call", per_call);
    assert!(per_call < 100.0, "Combined similarity calculation too slow");
}

#[cfg(feature = "similarity-search")]
fn bench_simd_comparison(c: &mut Criterion) {
    let calc = SimilarityCalculator::new();

    c.bench_function("cosine_similarity_scalar_1024", |b| {
        let vec1 = generate_random_vector(1024);
        let vec2 = generate_random_vector(1024);
        b.iter(|| calc.cosine_similarity(black_box(&vec1), black_box(&vec2)));
    });

    c.bench_function("cosine_similarity_simd_1024", |b| {
        let vec1 = generate_random_vector(1024);
        let vec2 = generate_random_vector(1024);
        b.iter(|| calc.cosine_similarity_simd(black_box(&vec1), black_box(&vec2)));
    });
}

criterion_group!(
    benches,
    bench_cosine_similarity,
    bench_jaccard_similarity,
    bench_combined_similarity
);

#[cfg(feature = "similarity-search")]
criterion_group!(simd_benches, bench_simd_comparison);

#[cfg(not(feature = "similarity-search"))]
criterion_main!(benches);

#[cfg(feature = "similarity-search")]
criterion_main!(benches, simd_benches);
