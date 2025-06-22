use criterion::{black_box, criterion_group, criterion_main, Criterion};
use datacloak_core::ml_classifier::Column;
use datacloak_core::ml_graph_integration::MLGraphIntegration;

fn benchmark_column_ranking(c: &mut Criterion) {
    let integration = MLGraphIntegration::new();
    
    // Create test columns
    let columns: Vec<Column> = (0..20)
        .map(|i| {
            if i % 3 == 0 {
                Column::new(
                    &format!("text_column_{}", i),
                    vec!["This is a long text description", "Another paragraph of text"],
                )
            } else if i % 3 == 1 {
                Column::new(
                    &format!("numeric_column_{}", i),
                    vec!["123.45", "678.90", "234.56"],
                )
            } else {
                Column::new(
                    &format!("mixed_column_{}", i),
                    vec!["ABC123", "XYZ789", "123ABC"],
                )
            }
        })
        .collect();
    
    c.bench_function("rank_20_columns_ml_only", |b| {
        b.iter(|| {
            integration.rank_columns_with_graph(black_box(&columns), 1.0)
        })
    });
    
    c.bench_function("rank_20_columns_hybrid", |b| {
        b.iter(|| {
            integration.rank_columns_with_graph(black_box(&columns), 0.7)
        })
    });
    
    c.bench_function("rank_20_columns_graph_heavy", |b| {
        b.iter(|| {
            integration.rank_columns_with_graph(black_box(&columns), 0.3)
        })
    });
}

fn benchmark_graph_building(c: &mut Criterion) {
    let integration = MLGraphIntegration::new();
    
    let columns: Vec<Column> = (0..50)
        .map(|i| {
            Column::new(
                &format!("column_{}", i),
                vec!["sample", "data", "values"],
            )
        })
        .collect();
    
    c.bench_function("build_graph_50_columns", |b| {
        b.iter(|| {
            integration.build_column_graph(black_box(&columns))
        })
    });
}

criterion_group!(benches, benchmark_column_ranking, benchmark_graph_building);
criterion_main!(benches);