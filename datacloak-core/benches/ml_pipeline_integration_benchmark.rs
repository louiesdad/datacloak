use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use datacloak_core::{
    feature_extractor::FeatureExtractor,
    ml_classifier::{Column, MLClassifier},
    ml_graph_integration::MLGraphRanker,
};

fn benchmark_full_pipeline(c: &mut Criterion) {
    let ranker = MLGraphRanker::new();

    let mut group = c.benchmark_group("ml_pipeline");

    for column_count in [10, 50, 100, 500].iter() {
        let columns: Vec<Column> = (0..*column_count)
            .map(|i| {
                if i % 3 == 0 {
                    Column::new(
                        &format!("text_col_{}", i),
                        vec![
                            "This is a comprehensive product description with detailed information",
                            "Another detailed review with extensive analysis and recommendations",
                            "More descriptive text content for accurate classification testing",
                        ],
                    )
                } else if i % 3 == 1 {
                    Column::new(
                        &format!("num_col_{}", i),
                        vec!["123.45", "67.89", "999.00", "12.34"],
                    )
                } else {
                    Column::new(
                        &format!("id_col_{}", i),
                        vec!["ID12345", "USR67890", "ORD11111", "SKU22222"],
                    )
                }
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("profile_columns", column_count),
            &columns,
            |b, columns| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                b.to_async(rt)
                    .iter(|| async { ranker.profile_columns(black_box(columns)).await.unwrap() })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("build_similarity_graph", column_count),
            &columns,
            |b, columns| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                b.to_async(rt).iter(|| async {
                    ranker
                        .build_similarity_graph(black_box(columns))
                        .await
                        .unwrap()
                })
            },
        );
    }

    group.finish();
}

fn benchmark_feature_extraction_scalability(c: &mut Criterion) {
    let extractor = FeatureExtractor::new();

    let mut group = c.benchmark_group("feature_extraction_scalability");

    for data_size in [100, 500, 1000, 2000].iter() {
        let text_samples: Vec<String> = (0..*data_size)
            .map(|i| format!("Sample text data for testing number {}", i))
            .collect();
        let text_refs: Vec<&str> = text_samples.iter().map(|s| s.as_str()).collect();
        let column = Column::new("test_column", text_refs);

        group.bench_with_input(
            BenchmarkId::new("extract_all_features", data_size),
            &column,
            |b, column| b.iter(|| extractor.extract_all_features(black_box(column))),
        );
    }

    group.finish();
}

fn benchmark_ml_classifier_batch_processing(c: &mut Criterion) {
    let classifier = MLClassifier::new();

    let mut group = c.benchmark_group("ml_classifier_batch");

    for batch_size in [50, 100, 500, 1000].iter() {
        let batch: Vec<Column> = (0..*batch_size)
            .map(|i| match i % 4 {
                0 => Column::new(
                    &format!("text_col_{}", i),
                    vec![
                        "Comprehensive product description with detailed specifications",
                        "Extensive user review with pros and cons analysis",
                    ],
                ),
                1 => Column::new(
                    &format!("short_text_{}", i),
                    vec!["Good", "Bad", "Excellent", "Poor", "Average"],
                ),
                2 => Column::new(
                    &format!("numeric_col_{}", i),
                    vec!["123.45", "67.89", "999.00"],
                ),
                _ => Column::new(
                    &format!("id_col_{}", i),
                    vec!["USR12345", "ORD67890", "SKU11111"],
                ),
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("predict_batch", batch_size),
            &batch,
            |b, batch| b.iter(|| classifier.predict_batch(black_box(batch))),
        );
    }

    group.finish();
}

fn benchmark_graph_construction_performance(c: &mut Criterion) {
    let ranker = MLGraphRanker::new();

    c.bench_function("graph_construction_similar_columns", |b| {
        let similar_columns = vec![
            Column::new(
                "product_description",
                vec![
                    "High-quality wireless headphones with excellent sound",
                    "Premium audio device with noise cancellation features",
                ],
            ),
            Column::new(
                "product_review",
                vec![
                    "Amazing sound quality and comfortable to wear",
                    "Great product with fast shipping and good packaging",
                ],
            ),
            Column::new(
                "customer_feedback",
                vec![
                    "Very satisfied with the purchase experience",
                    "Excellent customer service and product quality",
                ],
            ),
            Column::new(
                "item_details",
                vec![
                    "Professional-grade equipment for audio enthusiasts",
                    "High-end specifications with advanced technology",
                ],
            ),
        ];

        let rt = tokio::runtime::Runtime::new().unwrap();
        b.to_async(rt).iter(|| async {
            ranker
                .build_similarity_graph(black_box(&similar_columns))
                .await
                .unwrap()
        })
    });

    c.bench_function("graph_construction_diverse_columns", |b| {
        let diverse_columns = vec![
            Column::new("text_content", vec!["Long text description", "More text"]),
            Column::new("numeric_values", vec!["123.45", "67.89"]),
            Column::new("identifiers", vec!["ID123", "USR456"]),
            Column::new("categories", vec!["A", "B", "C"]),
            Column::new("dates", vec!["2024-01-01", "2024-01-02"]),
            Column::new("booleans", vec!["true", "false"]),
        ];

        let rt = tokio::runtime::Runtime::new().unwrap();
        b.to_async(rt).iter(|| async {
            ranker
                .build_similarity_graph(black_box(&diverse_columns))
                .await
                .unwrap()
        })
    });
}

criterion_group!(
    benches,
    benchmark_full_pipeline,
    benchmark_feature_extraction_scalability,
    benchmark_ml_classifier_batch_processing,
    benchmark_graph_construction_performance
);
criterion_main!(benches);
