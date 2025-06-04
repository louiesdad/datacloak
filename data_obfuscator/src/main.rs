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
use metrics::Metrics;
use tokio::io::BufReader;
use prometheus::Registry;
use std::path::Path;
use tracing::{info, error};

#[derive(Parser)]
#[command(name = "data-obfuscator", version)]
struct Cli {
    #[arg(short, long)]
    customer_id: Option<i64>,

    #[arg(long, conflicts_with = "customer_id")]
    document_path: Option<String>,

    #[arg(short, long, default_value = "config/obfuscation_rules.json")]
    rules: String,

    #[arg(long, default_value = "https://api.openai.com/v1/chat/completions")]
    llm_endpoint: String,

    #[arg(long)]
    api_key: Option<String>,

    #[arg(long)]
    debug_obfuscated_path: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), AppError> {
    logger::init();
    let cli = Cli::parse();
    let cfg = load_config(&cli.rules, &cli.llm_endpoint, &cli.api_key)?;

    let registry = prometheus::Registry::new();
    let metrics = Metrics::new(&registry);

    let mut obfuscator = Obfuscator::new(&cfg.rules)?;

    let obfuscated_text = if let Some(id) = cli.customer_id {
        info!("Fetching customer ID {}", id);
        // Placeholder DB record
        let raw_blob = format!(
            "Customer ID: {}\nName: John Doe\nEmail: john{}@example.com\nPhone: 555-1234\nNotes: Example\n",
            id, id
        );
        obfuscator.obfuscate_text(&raw_blob)
    } else if let Some(path_str) = cli.document_path.clone() {
        info!("Reading document from {}", path_str);
        let file = tokio::fs::File::open(Path::new(&path_str)).await?;
        let reader = BufReader::new(file);
        let mut buf = Vec::new();
        obfuscator.obfuscate_stream(reader, &mut buf).await?;
        String::from_utf8(buf).map_err(|e| AppError::Other(e.to_string()))?
    } else {
        error!("Either --customer-id or --document-path must be provided.");
        return Err(AppError::Other("Missing input source".into()));
    };

    if let Some(ref path) = cli.debug_obfuscated_path {
        tokio::fs::write(path, &obfuscated_text).await?;
    }

    let client = LlmClient::new(cfg.llm_endpoint, cfg.api_key);
    info!("Sending obfuscated payload to LLM");
    let obfuscated_reply = client.chat(&obfuscated_text).await?;
    metrics.request_count.inc();

    info!("De-obfuscating LLM response");
    let final_reply = deobfuscate_text(&obfuscated_reply, obfuscator.placeholder_map());
    println!("{}", final_reply);
    Ok(())
}
