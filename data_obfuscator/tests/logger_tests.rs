use data_obfuscator::logger;

#[test]
fn init_logger() {
    logger::init_logging();
    tracing::info!("logger initialized");
}
