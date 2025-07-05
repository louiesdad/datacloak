use datacloak_core::graph::knn::KnnIndex;

#[test]
fn test_build_faiss_index() {
    let mut index = KnnIndex::new(300); // 300-dim vectors
    let vectors = vec![vec![1.0; 300], vec![0.5; 300], vec![0.0; 300]];

    index.add_batch(&vectors).unwrap();
    assert_eq!(index.size(), 3);
}

#[test]
fn test_knn_search() {
    let mut index = KnnIndex::new(2);
    index.add(&vec![0.0, 0.0]).unwrap();
    index.add(&vec![1.0, 0.0]).unwrap();
    index.add(&vec![0.0, 1.0]).unwrap();

    let query = vec![0.1, 0.1];
    let neighbors = index.search(&query, 2).unwrap();

    assert_eq!(neighbors.len(), 2);
    assert_eq!(neighbors[0].index, 0); // Closest is first point
    assert!(neighbors[0].distance < neighbors[1].distance);
}

#[test]
fn test_knn_search_with_ids() {
    let mut index = KnnIndex::new(3);

    // Add vectors with custom IDs
    index.add_with_id(&vec![1.0, 0.0, 0.0], "col_a").unwrap();
    index.add_with_id(&vec![0.0, 1.0, 0.0], "col_b").unwrap();
    index.add_with_id(&vec![0.0, 0.0, 1.0], "col_c").unwrap();

    let query = vec![0.9, 0.1, 0.0];
    let neighbors = index.search(&query, 2).unwrap();

    assert_eq!(neighbors[0].id.as_ref().unwrap(), "col_a");
}

#[test]
fn test_batch_search() {
    let mut index = KnnIndex::new(2);
    let vectors = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];
    index.add_batch(&vectors).unwrap();

    let queries = vec![vec![0.1, 0.0], vec![0.9, 0.0]];

    let results = index.batch_search(&queries, 2).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0][0].index, 0); // First query closest to first point
    assert_eq!(results[1][0].index, 1); // Second query closest to second point
}

#[test]
fn test_knn_performance() {
    use std::time::Instant;

    let mut index = KnnIndex::new(300);
    let vectors: Vec<Vec<f32>> = (0..10000)
        .map(|_| (0..300).map(|_| rand::random()).collect())
        .collect();

    index.add_batch(&vectors).unwrap();

    let start = Instant::now();
    let _ = index.search(&vectors[0], 10);
    let elapsed = start.elapsed();
    println!("KNN search time: {:?}", elapsed);

    // In debug mode, this will be slower. Performance should be tested in release mode.
    #[cfg(debug_assertions)]
    assert!(elapsed.as_millis() < 100); // <100ms in debug mode

    #[cfg(not(debug_assertions))]
    assert!(elapsed.as_millis() < 10); // <10ms in release mode
}

#[test]
fn test_save_and_load_index() {
    let mut index = KnnIndex::new(3);
    index.add_with_id(&vec![1.0, 0.0, 0.0], "col_a").unwrap();
    index.add_with_id(&vec![0.0, 1.0, 0.0], "col_b").unwrap();

    // Save index
    let temp_path = "/tmp/test_knn_index.bin";
    index.save(temp_path).unwrap();

    // Load index
    let loaded_index = KnnIndex::load(temp_path, 3).unwrap();
    assert_eq!(loaded_index.size(), 2);

    // Verify search works
    let query = vec![0.9, 0.1, 0.0];
    let neighbors = loaded_index.search(&query, 1).unwrap();
    assert_eq!(neighbors[0].id.as_ref().unwrap(), "col_a");

    // Cleanup
    std::fs::remove_file(temp_path).ok();
}

#[test]
fn test_pure_rust_fallback() {
    // Test the pure Rust implementation when FAISS is not available
    let mut index = KnnIndex::new_pure_rust(3);

    index.add(&vec![1.0, 0.0, 0.0]).unwrap();
    index.add(&vec![0.0, 1.0, 0.0]).unwrap();
    index.add(&vec![0.0, 0.0, 1.0]).unwrap();

    let query = vec![0.8, 0.6, 0.0];
    let neighbors = index.search(&query, 2).unwrap();

    assert_eq!(neighbors.len(), 2);
    // Should return indices 0 and 1 as closest
    assert!(neighbors.iter().any(|n| n.index == 0));
    assert!(neighbors.iter().any(|n| n.index == 1));
}

#[test]
fn test_empty_index_search() {
    let index = KnnIndex::new(3);
    let query = vec![1.0, 0.0, 0.0];
    let result = index.search(&query, 5);

    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 0);
}

#[test]
fn test_dimension_mismatch() {
    let mut index = KnnIndex::new(3);
    index.add(&vec![1.0, 0.0, 0.0]).unwrap();

    // Wrong dimension query
    let query = vec![1.0, 0.0]; // 2D instead of 3D
    let result = index.search(&query, 1);

    assert!(result.is_err());
}
