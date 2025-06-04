use tracing_subscriber::{fmt, EnvFilter, prelude::*};

pub fn init_logging() {
    let fmt_layer = fmt::layer()
        .with_target(false)
        .with_thread_ids(true)
        .json();

    let filter_layer = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(filter_layer)
        .with(fmt_layer)
        .init();
}
