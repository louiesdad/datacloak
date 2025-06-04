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
    logger::init_logging();
    let cli = Cli::parse();
    let cfg = load_config(&cli.rules, &cli.llm_endpoint, &cli.api_key)?;
    let mut obfuscator = Obfuscator::new(&cfg.rules)?;
    let obfuscated = obfuscator.obfuscate_text(&cli.input);

    let client = LlmClient::new(cfg.llm_endpoint, cfg.api_key);
    let reply = client.chat(&obfuscated).await?;

    let final_reply = deobfuscate_text(&reply, obfuscator.placeholder_map());
    println!("{}", final_reply);
    Ok(())
}
