use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use datacloak_core::model_optimization::{ModelCache, ModelOptimizer, QuantizationLevel};
use datacloak_core::onnx_model::OnnxModel;

fn benchmark_quantization_inference(c: &mut Criterion) {
    let model = OnnxModel::mock();
    let optimizer = ModelOptimizer::new();

    let quantized_int8 = optimizer.quantize(&model, QuantizationLevel::Int8).unwrap();
    let quantized_int16 = optimizer
        .quantize(&model, QuantizationLevel::Int16)
        .unwrap();
    let quantized_dynamic = optimizer
        .quantize(&model, QuantizationLevel::Dynamic)
        .unwrap();

    let test_features = vec![0.5; 377];

    let mut group = c.benchmark_group("model_inference");

    group.bench_function("original", |b| {
        b.iter(|| model.predict(black_box(&test_features)).unwrap())
    });

    group.bench_function("quantized_int8", |b| {
        b.iter(|| quantized_int8.predict(black_box(&test_features)).unwrap())
    });

    group.bench_function("quantized_int16", |b| {
        b.iter(|| quantized_int16.predict(black_box(&test_features)).unwrap())
    });

    group.bench_function("quantized_dynamic", |b| {
        b.iter(|| {
            quantized_dynamic
                .predict(black_box(&test_features))
                .unwrap()
        })
    });

    group.finish();
}

fn benchmark_model_caching(c: &mut Criterion) {
    let cache = ModelCache::new(100); // 100MB cache

    c.bench_function("model_load_no_cache", |b| {
        b.iter(|| {
            let _model = OnnxModel::mock();
        })
    });

    c.bench_function("model_load_with_cache", |b| {
        b.iter(|| {
            let _model = cache.get_or_load(black_box("test_model")).unwrap();
        })
    });
}

fn benchmark_batch_inference(c: &mut Criterion) {
    let model = OnnxModel::mock();
    let optimizer = ModelOptimizer::new();
    let quantized = optimizer.quantize(&model, QuantizationLevel::Int8).unwrap();

    let mut group = c.benchmark_group("batch_inference");

    for batch_size in [10, 100, 1000].iter() {
        let batch: Vec<_> = (0..*batch_size).map(|_| vec![0.5; 377]).collect();

        group.bench_with_input(
            BenchmarkId::new("original", batch_size),
            &batch,
            |b, batch| {
                b.iter(|| {
                    for features in batch {
                        let _ = model.predict(black_box(features)).unwrap();
                    }
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("quantized", batch_size),
            &batch,
            |b, batch| {
                b.iter(|| {
                    for features in batch {
                        let _ = quantized.predict(black_box(features)).unwrap();
                    }
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_quantization_inference,
    benchmark_model_caching,
    benchmark_batch_inference
);
criterion_main!(benches);
