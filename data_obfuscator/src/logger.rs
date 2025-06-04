use tracing_subscriber::fmt::Subscriber;

pub fn init() {
    Subscriber::builder().with_env_filter("info").init();
}
