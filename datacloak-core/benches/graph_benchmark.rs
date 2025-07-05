use criterion::{black_box, criterion_group, criterion_main, Criterion};
use datacloak_core::graph::{ColumnGraph, ColumnNode};
use std::time::Instant;

fn generate_test_columns(count: usize) -> Vec<ColumnNode> {
    (0..count)
        .map(|i| {
            let embedding = vec![i as f32 / count as f32; 300]; // 300-dim embeddings
            ColumnNode::new(&format!("column_{}", i), embedding)
        })
        .collect()
}

fn bench_graph_construction(c: &mut Criterion) {
    c.bench_function("graph_construction_1000_columns", |b| {
        let columns = generate_test_columns(1000);
        b.iter(|| {
            let mut graph = ColumnGraph::new();
            for col in &columns {
                graph.add_node(black_box(col.clone()));
            }
            black_box(graph);
        });
    });

    c.bench_function("graph_with_edges_100_nodes", |b| {
        let columns = generate_test_columns(100);
        b.iter(|| {
            let mut graph = ColumnGraph::new();
            let node_indices: Vec<_> = columns
                .iter()
                .map(|col| graph.add_node(col.clone()))
                .collect();

            // Add edges with decreasing similarity
            for i in 0..node_indices.len() {
                for j in i + 1..node_indices.len() {
                    let similarity = 1.0 - ((j - i) as f32 / node_indices.len() as f32);
                    if similarity > 0.5 {
                        graph.add_edge(node_indices[i], node_indices[j], similarity);
                    }
                }
            }
            black_box(graph);
        });
    });
}

fn bench_graph_operations(c: &mut Criterion) {
    let mut graph = ColumnGraph::new();
    let columns = generate_test_columns(500);
    let node_indices: Vec<_> = columns
        .iter()
        .map(|col| graph.add_node(col.clone()))
        .collect();

    // Add edges
    for i in 0..node_indices.len() {
        for j in i + 1..((i + 10).min(node_indices.len())) {
            graph.add_edge(node_indices[i], node_indices[j], 0.8);
        }
    }

    c.bench_function("get_neighbors", |b| {
        b.iter(|| {
            let neighbors = graph.get_neighbors(black_box(node_indices[0]));
            black_box(neighbors);
        });
    });

    c.bench_function("get_metrics", |b| {
        b.iter(|| {
            let metrics = graph.get_metrics();
            black_box(metrics);
        });
    });

    c.bench_function("remove_edges_below_threshold", |b| {
        let graph_clone = graph.clone();
        b.iter_batched(
            || graph_clone.clone(),
            |mut g| {
                g.remove_edges_below_threshold(0.7);
                black_box(g);
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

fn measure_large_graph_performance() {
    println!("\n=== Large Graph Performance Test ===");

    let start = Instant::now();
    let columns = generate_test_columns(1000);
    println!("Generated 1000 columns in {:?}", start.elapsed());

    let start = Instant::now();
    let mut graph = ColumnGraph::new();
    let node_indices: Vec<_> = columns
        .iter()
        .map(|col| graph.add_node(col.clone()))
        .collect();
    println!("Added 1000 nodes in {:?}", start.elapsed());

    let start = Instant::now();
    let mut edge_count = 0;
    for i in 0..node_indices.len() {
        for j in i + 1..((i + 50).min(node_indices.len())) {
            let similarity = 1.0 - ((j - i) as f32 / 50.0);
            if similarity > 0.3 {
                graph.add_edge(node_indices[i], node_indices[j], similarity);
                edge_count += 1;
            }
        }
    }
    println!("Added {} edges in {:?}", edge_count, start.elapsed());

    let start = Instant::now();
    let metrics = graph.get_metrics();
    println!("Calculated metrics in {:?}", start.elapsed());
    println!("Graph metrics: {:?}", metrics);

    assert!(
        start.elapsed().as_secs() < 5,
        "Graph construction took too long!"
    );
}

criterion_group!(benches, bench_graph_construction, bench_graph_operations);
criterion_main!(benches);
