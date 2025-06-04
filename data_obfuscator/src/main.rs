mod config;
mod obfuscator;
mod llm_client;
mod deobfuscator;
mod errors;
mod metrics;
mod logger;

use clap::Parser;
use errors::AppError;
use obfuscator::Obfuscator;
use config::load_config;
use llm_client::LlmClient;
use deobfuscator::deobfuscate_text;
use prometheus::Registry;
use metrics::Metrics;

#[derive(Parser)]
struct Cli {
    #[arg(long, default_value = "config/obfuscation_rules.json")]
    rules: String,
    #[arg(long, default_value = "http://localhost")] 
    llm_endpoint: String,
    #[arg(long, default_value = "test-key")]
    api_key: String,
    #[arg(long, default_value = "")]
    input: String,
}

#[tokio::main]
async fn main() -> Result<(), AppError> {
    logger::init();
    let cli = Cli::parse();
    let registry = Registry::new();
    let metrics = Metrics::new(&registry);

    let overall_timer = metrics.request_duration.start_timer();

    let cfg = load_config(&cli.rules, &cli.llm_endpoint, &cli.api_key)?;
    let mut obfuscator = Obfuscator::new(&cfg.rules)?;
    let obf_timer = metrics.obfuscation_duration.start_timer();
    let obfuscated = obfuscator.obfuscate_text(&cli.input);
    obf_timer.observe_duration();

    let client = LlmClient::new(cfg.llm_endpoint, cfg.api_key);
    let llm_timer = metrics.llm_duration.start_timer();
    let reply = client.chat(&obfuscated).await?;
    llm_timer.observe_duration();

    metrics.request_count.inc();
    overall_timer.observe_duration();

    let final_reply = deobfuscate_text(&reply, obfuscator.placeholder_map());
    println!("{}", final_reply);
    Ok(())
}
