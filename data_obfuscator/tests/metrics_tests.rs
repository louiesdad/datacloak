use data_obfuscator::metrics::Metrics;
use prometheus::Registry;

#[test]
fn counter_increments() {
    let registry = Registry::new();
    let metrics = Metrics::new(&registry);
    metrics.request_count.inc();
    assert_eq!(metrics.request_count.get(), 1);
}