use data_obfuscator::metrics::Metrics;

#[test]
fn counter_increments() {
    let metrics = Metrics::new();
    metrics.request_count.inc();
    assert_eq!(metrics.request_count.get(), 1);
}
