mod config;
mod obfuscator;
mod llm_client;
mod deobfuscator;
mod errors;
mod metrics;
mod logger;

use clap::Parser;
use errors::AppError;
use obfuscator::{Obfuscator, StreamConfig};
use config::load_config;
use llm_client::LlmClient;
use deobfuscator::deobfuscate_text;
use metrics::Metrics;
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

    #[arg(long, default_value = "262144")]
    chunk_size: usize,
}

#[tokio::main]
async fn main() -> Result<(), AppError> {
    logger::init_logging();
    let cli = Cli::parse();
    let registry = Registry::new();
    let metrics = Metrics::new(&registry);

    let overall_timer = metrics.request_duration.start_timer();
    let cfg = load_config(&cli.rules, &cli.llm_endpoint, &cli.api_key)?;

    let mut obfuscator = Obfuscator::new(&cfg.rules)?;

    let obfuscated_text = if let Some(id) = cli.customer_id {
        info!("Fetching customer ID {}", id);
        // Placeholder DB record
        let raw_blob = format!(
            "Customer ID: {}\nName: John Doe\nEmail: john{}@example.com\nPhone: 555-1234\nNotes: Example\n",
            id, id
        );
        let obf_timer = metrics.obfuscation_duration.start_timer();
        let text = obfuscator.obfuscate_text(&raw_blob);
        obf_timer.observe_duration();
        text
    } else if let Some(path_str) = cli.document_path.clone() {
        info!("Reading document from {} with chunk size: {}", path_str, cli.chunk_size);
        let file = tokio::fs::File::open(Path::new(&path_str)).await?;
        let mut buf = Vec::new();
        let stream_config = StreamConfig { chunk_size: cli.chunk_size };
        let obf_timer = metrics.obfuscation_duration.start_timer();
        obfuscator.stream_file(file, &mut buf, &stream_config).await?;
        obf_timer.observe_duration();
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
    let llm_timer = metrics.llm_duration.start_timer();
    let obfuscated_reply = client.chat(&obfuscated_text).await?;
    llm_timer.observe_duration();

    metrics.request_count.inc();
    overall_timer.observe_duration();

    info!("De-obfuscating LLM response");
    let final_reply = deobfuscate_text(&obfuscated_reply, obfuscator.placeholder_map());
    println!("{}", final_reply);
    Ok(())
}