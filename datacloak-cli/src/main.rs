mod cli;
mod mock_llm;
mod scenarios;
mod test_framework;

use anyhow::Result;
use clap::Parser;
use cli::{Cli, Commands};
use tracing_subscriber::{fmt, EnvFilter, prelude::*};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Detect { file, rows, output } => {
            cli::detect_command(file, rows, output).await
        }
        Commands::Obfuscate { file, patterns, rows, output } => {
            cli::obfuscate_command(file, patterns, rows, output).await
        }
        Commands::Analyze { file, rows, patterns, api_key, dry_run, output } => {
            cli::analyze_command(file, rows, patterns, api_key, dry_run, output).await
        }
        Commands::MockServer { port, scenario } => {
            mock_llm::start_mock_server(port, scenario).await
        }
        Commands::TestScenario { scenario, mock_port } => {
            test_framework::run_scenario_test(scenario, mock_port).await
        }
        Commands::TestAll { mock_port } => {
            test_framework::run_all_tests(mock_port).await
        }
    }
}