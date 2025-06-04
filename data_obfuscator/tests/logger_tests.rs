use data_obfuscator::logger;

#[test]
fn init_logger() {
    logger::init();
    tracing::info!("logger initialized");
}
