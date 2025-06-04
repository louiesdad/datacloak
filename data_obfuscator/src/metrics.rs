use prometheus::{IntCounter, Registry};

pub struct Metrics {
    pub request_count: IntCounter,
}

impl Metrics {
    pub fn new() -> Self {
        let request_count = IntCounter::new("request_count", "Number of requests").unwrap();
        let registry = Registry::new();
        registry.register(Box::new(request_count.clone())).unwrap();
        Self { request_count }
    }
}
