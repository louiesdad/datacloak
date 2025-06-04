use prometheus::{
    Counter, Histogram, HistogramOpts, Opts, Registry,
};

pub struct Metrics {
    pub request_count: Counter,
    pub request_duration: Histogram,
    pub obfuscation_duration: Histogram,
    pub llm_duration: Histogram,
    pub error_count: Counter,
}

impl Default for Metrics {
    fn default() -> Self {
        let registry = Registry::new();
        Self::new(&registry)
    }
}

impl Metrics {
    pub fn new(registry: &Registry) -> Self {
        let request_count =
            Counter::with_opts(Opts::new("request_count_total", "Total number of requests"))
                .unwrap();
        let request_duration = Histogram::with_opts(HistogramOpts::new(
            "request_duration_seconds",
            "End-to-end request latency",
        ))
        .unwrap();
        let obfuscation_duration = Histogram::with_opts(HistogramOpts::new(
            "obfuscation_duration_seconds",
            "Time spent obfuscating input",
        ))
        .unwrap();
        let llm_duration = Histogram::with_opts(HistogramOpts::new(
            "llm_duration_seconds",
            "Time spent waiting for LLM",
        ))
        .unwrap();
        let error_count =
            Counter::with_opts(Opts::new("error_count_total", "Total number of errors"))
                .unwrap();

        registry.register(Box::new(request_count.clone())).unwrap();
        registry.register(Box::new(request_duration.clone())).unwrap();
        registry.register(Box::new(obfuscation_duration.clone())).unwrap();
        registry.register(Box::new(llm_duration.clone())).unwrap();
        registry.register(Box::new(error_count.clone())).unwrap();

        Self {
            request_count,
            request_duration,
            obfuscation_duration,
            llm_duration,
            error_count,
        }
    }
}