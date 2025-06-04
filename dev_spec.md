# dev_spec.md

## Project Structure

```
data_obfuscator/
├── Cargo.toml
├── src/
│   ├── main.rs
│   ├── config.rs
│   ├── obfuscator.rs
│   ├── llm_client.rs
│   ├── deobfuscator.rs
│   ├── errors.rs
│   ├── metrics.rs
│   └── logger.rs
├── config/
│   └── obfuscation_rules.json
├── Dockerfile
└── README.md
```

---

## 1. Cargo.toml

```toml
[package]
name = "data_obfuscator"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <you@example.com>"]

[dependencies]
# CLI parsing
clap = { version = "4.2", features = ["derive"] }

# Configuration (JSON/YAML)
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
serde_yaml = "0.9"
config = "0.13"

# Regex engine
regex = "1.7"

# Async & HTTP
tokio = { version = "1.29", features = ["full"] }
reqwest = { version = "0.11", features = ["json", "gzip", "stream", "rustls-tls"] }

# Logging & metrics
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["fmt", "env-filter", "json"] }
prometheus = "0.14"

# Optional ML / NER
# rust-bert = { git = "https://github.com/guillaume-be/rust-bert.git", branch = "master" }
# tract-onnx = "0.21"

# Database connectivity (if needed)
# sqlx = { version = "0.6", features = ["runtime-tokio-rustls", "postgres", "macros"] }

anyhow = "1.0"
thiserror = "1.0"
```

---

## 2. Module Responsibilities

### 2.1 `config.rs`

- **Purpose:** Load and validate configuration (CLI flags + config file).  
- **Key Functions:**  
  - `fn load_config() -> Result<AppConfig, ConfigError>`  
    - Read `config/obfuscation_rules.json` (or override via `--rules <PATH>`).  
    - Optionally read `LLM_ENDPOINT`, `OPENAI_API_KEY` from environment.  
  - `struct AppConfig { rules: Vec<Rule>, llm_endpoint: String, api_key: String, ... }`  
  - `struct Rule { pattern: String, label: String }`  
  - Use [`config`](https://docs.rs/config/latest/config/) crate to layer defaults, files, and env vars.

### 2.2 `obfuscator.rs`

- **Purpose:** Identify and replace sensitive data with placeholders.  
- **Key Types/Functions:**  
  ```rust
  pub struct Obfuscator {
      rules: Vec<(Regex, String)>, // compiled patterns + label
      placeholder_counter: usize,
      placeholder_map: HashMap<String, String>, // token → original
      reverse_map: HashMap<String, String>,     // original → token (optional)
  }

  impl Obfuscator {
      /// Initialize: compile regexes, set counters to zero
      pub fn new(rules: Vec<RuleConfig>) -> Result<Self, ObfuscationError> { ... }

      /// Given a &str (field or chunk), returns obfuscated String
      pub fn obfuscate_text(&mut self, input: &str) -> String { ... }

      /// For large files: stream line-by-line
      pub async fn obfuscate_stream<R, W>(
          &mut self,
          reader: R,
          mut writer: W
      ) -> Result<(), ObfuscationError>
      where
          R: AsyncBufRead + Unpin,
          W: AsyncWrite + Unpin,
      { ... }

      /// Retrieve the placeholder map (token → original)
      pub fn retrieve_map(&self) -> &HashMap<String, String> { ... }
  }
  ```
- **Implementation Notes:**  
  - Use [`regex::Regex::replace_all`](https://docs.rs/regex/latest/regex/struct.Regex.html#method.replace_all).  
  - For streaming: use `tokio::io::BufReader::lines().await` to get each line, call `obfuscate_text`, then write out via `writer.write_all()`.  
  - Optionally, to be more memory-efficient, process fixed‐size byte buffers (e.g., 8 KB at a time) but handle boundary cases for partial matches.

### 2.3 `llm_client.rs`

- **Purpose:** Wrap all interactions with the external LLM API.  
- **Key Types/Functions:**  
  ```rust
  pub struct LlmClient {
      endpoint: String,
      api_key: String,
      http_client: reqwest::Client,
  }

  impl LlmClient {
      /// Create with endpoint & API key
      pub fn new(endpoint: String, api_key: String) -> Self { ... }

      /// Sends a ChatCompletion request; returns raw JSON or parsed struct
      pub async fn chat(&self, prompt: &str) -> Result<String, LlmError> {
          let request_body = serde_json::json!({
              "model": "gpt-4",
              "messages": [
                  { "role": "system", "content": "You are a secure data processor." },
                  { "role": "user", "content": prompt }
              ]
          });

          let resp = self
              .http_client
              .post(&self.endpoint)
              .bearer_auth(&self.api_key)
              .json(&request_body)
              .send()
              .await?
              .error_for_status()?
              .json::<serde_json::Value>()
              .await?;

          // Navigate JSON to extract the assistant’s content
          let content = resp["choices"][0]["message"]["content"]
              .as_str()
              .ok_or(LlmError::InvalidResponse)?.to_string();
          Ok(content)
      }
  }
  ```
- **References:**  
  - [OpenAI Chat Completions API docs](https://platform.openai.com/docs/api-reference/chat)  
  - [`reqwest` usage examples](https://docs.rs/reqwest/latest/reqwest/#examples)  

### 2.4 `deobfuscator.rs`

- **Purpose:** Reverse the placeholder substitution, reinserting real values.  
- **Key Function:**  
  ```rust
  pub fn deobfuscate_text(
      obfuscated: &str,
      placeholder_map: &HashMap<String, String>
  ) -> String {
      let mut result = obfuscated.to_string();
      for (token, original) in placeholder_map.iter() {
          result = result.replace(token, original);
      }
      result
  }
  ```
- **Performance Consideration:**  
  - If the output is large, consider using a streaming approach (e.g., split on tokens, rebuild).  
  - In most cases, `String::replace` is acceptable up to a few MBs.

### 2.5 `errors.rs`

- **Purpose:** Define custom error types using [`thiserror`](https://docs.rs/thiserror/latest/thiserror/).  
- **Example:**
  ```rust
  use thiserror::Error;

  #[derive(Error, Debug)]
  pub enum ObfuscationError {
      #[error("failed to compile regex: {0}")]
      RegexCompileError(String),

      #[error("I/O error: {0}")]
      IoError(#[from] std::io::Error),

      #[error("tokio I/O error: {0}")]
      TokioIoError(#[from] tokio::io::Error),

      // … more as needed
  }

  #[derive(Error, Debug)]
  pub enum LlmError {
      #[error("HTTP request error: {0}")]
      HttpError(#[from] reqwest::Error),

      #[error("Invalid response format")]
      InvalidResponse,
  }
  ```

### 2.6 `metrics.rs`

- **Purpose:** Expose Prometheus‐style metrics for observability (optional).  
- **Key Metrics:**  
  - `request_count_total` – total number of requests processed.  
  - `request_duration_seconds` – histogram of end‐to‐end latency.  
  - `obfuscation_duration_seconds` – time spent in obfuscator.  
  - `llm_duration_seconds` – time spent waiting for LLM.  
  - `error_count_total` – total number of failed requests.  

- **Implementation:**  
  ```rust
  use prometheus::{Opts, Registry, Counter, Histogram, register_counter, register_histogram};

  pub struct Metrics {
      pub request_count: Counter,
      pub request_latency: Histogram,
      pub error_count: Counter,
      // …
  }

  impl Metrics {
      pub fn new(registry: &Registry) -> Self { ... }
  }
  ```

### 2.7 `logger.rs`

- **Purpose:** Set up structured logging with [`tracing`](https://docs.rs/tracing/latest/tracing/) and [`tracing-subscriber`](https://docs.rs/tracing-subscriber/latest/tracing_subscriber/).  
- **Initialization (in `main.rs`):**  
  ```rust
  use tracing_subscriber::{fmt, EnvFilter};

  pub fn init_logging() {
      let fmt_layer = fmt::layer()
          .with_target(false)       // don’t log module path
          .with_thread_ids(true)    // for concurrent tracing
          .json();                  // output in JSON to stdout

      let filter_layer = EnvFilter::try_from_default_env()
          .unwrap_or_else(|_| EnvFilter::new("info"));

      tracing_subscriber::registry()
          .with(filter_layer)
          .with(fmt_layer)
          .init();
  }
  ```

---

## 3. `main.rs` (Entry Point)

```rust
use clap::Parser;
use tokio::io::{self, AsyncBufReadExt, BufReader};
use std::fs::File;
use std::path::Path;
use tracing::{info, error};

mod config;
mod obfuscator;
mod llm_client;
mod deobfuscator;
mod errors;
mod metrics;
mod logger;

use config::AppConfig;
use obfuscator::Obfuscator;
use llm_client::LlmClient;
use deobfuscator::deobfuscate_text;
use metrics::Metrics;

#[derive(Parser)]
#[command(name = "data-obfuscator", version)]
struct Opts {
    /// Customer ID to process
    #[arg(short, long)]
    customer_id: Option<i64>,

    /// Path to a “large document” to process instead of a DB record
    #[arg(long, conflicts_with = "customer_id")]
    document_path: Option<String>,

    /// Path to obfuscation rules (JSON or YAML)
    #[arg(short, long, default_value = "config/obfuscation_rules.json")]
    rules: String,

    /// LLM API endpoint
    #[arg(long, default_value = "https://api.openai.com/v1/chat/completions")]
    llm_endpoint: String,

    /// API key (or read from OPENAI_API_KEY env var)
    #[arg(long)]
    api_key: Option<String>,

    /// Write obfuscated intermediate to a file (for debugging)
    #[arg(long)]
    debug_obfuscated_path: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    logger::init_logging();
    let opts = Opts::parse();

    // 1. Load configuration
    let mut cfg = AppConfig::load(&opts.rules, &opts.llm_endpoint, &opts.api_key)?;
    let rules_cfg = cfg.rules.clone();
    let api_key = cfg.api_key.clone();
    let llm_endpoint = cfg.llm_endpoint.clone();

    // 2. Initialize metrics
    let registry = prometheus::Registry::new();
    let metrics = Metrics::new(&registry);

    // 3. Initialize Obfuscator
    let mut obfuscator = Obfuscator::new(rules_cfg)?;

    # 4. Prepare input (DB vs. File)
    let obfuscated_text = if let Some(customer_id) = opts.customer_id {
        info!("Fetching customer ID {}", customer_id);
        # 4a. Fetch from DB (pseudo-code; implement your own DB layer)
        let customer = db::get_customer_by_id(customer_id).await?;
        let raw_blob = format!(
            "Customer ID: {}\nName: {} {}\nEmail: {}\nPhone: {}\nNotes: {}\n",
            customer.id,
            customer.first_name,
            customer.last_name,
            customer.email,
            customer.phone,
            customer.notes
        );
        let obf = obfuscator.obfuscate_text(&raw_blob);
        obf
    } else if let Some(doc_path) = opts.document_path.clone() {
        info!("Reading document from {}", doc_path);
        let path = Path::new(&doc_path);
        let file = tokio::fs::File::open(path).await?;
        let reader = BufReader::new(file);
        let mut buffer = Vec::new();
        obfuscator.obfuscate_stream(reader, &mut buffer).await?;
        String::from_utf8(buffer)?
    } else {
        error!("Either --customer-id or --document-path must be provided.");
        return Err(anyhow::anyhow!("Missing input source"));
    };

    # 5. (Optional) Write obfuscated to disk for debugging
    if let Some(ref debug_path) = opts.debug_obfuscated_path {
        tokio::fs::write(debug_path, &obfuscated_text).await?;
    }

    # 6. Call LLM
    let llm = LlmClient::new(llm_endpoint.clone(), api_key.clone());
    info!("Sending obfuscated payload to LLM");
    let obfuscated_reply = llm.chat(&obfuscated_text).await?;
    metrics.request_count.inc();

    # 7. De-obfuscate
    info!("De-obfuscating LLM response");
    let final_reply = deobfuscate_text(&obfuscated_reply, &obfuscator.retrieve_map());

    # 8. Output (to stdout or return via web framework)
    println!("{}", final_reply);

    Ok(())
}
```

---

## 4. Configuration & Rule File

### 4.1 `config/obfuscation_rules.json`

```jsonc
[
  {
    "pattern": "\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
    "label": "EMAIL"
  },
  {
    "pattern": "\b\d{3}-\d{2}-\d{4}\b",
    "label": "SSN"
  },
  {
    "pattern": "\b\(?\d{3}\)?[- ]?\d{3}-\d{4}\b",
    "label": "PHONE"
  },
  {
    "pattern": "\b\d{16}\b",
    "label": "CREDIT_CARD"
  },
  {
    "pattern": "\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b",
    "label": "CREDIT_CARD"
  }
]
```

> **Tip:** You can extend this file with additional patterns (bank account numbers, IP addresses, proprietary customer IDs, etc.). Use [regex101.com](https://regex101.com) or [Regex Cheat Sheet](https://www.rexegg.com/regex-quickstart.html) to test patterns.

---

## 5. Detailed Implementation Notes

### 5.1 Regex Compilation & Replacement

- **Compile all patterns once at startup** (in `Obfuscator::new`).  
  ```rust
  for rule in rules_cfg.iter() {
      let compiled = Regex::new(&rule.pattern)
          .map_err(|e| ObfuscationError::RegexCompileError(e.to_string()))?;
      self.rules.push((compiled, rule.label.clone()));
  }
  ```
- **Obfuscation Algorithm (pseudocode):**  
  ```rust
  pub fn obfuscate_text(&mut self, input: &str) -> String {
      let mut intermediate = input.to_string();
      for (regex, label) in &self.rules {
          intermediate = regex.replace_all(&intermediate, |caps: &Captures| {
              let matched = caps.get(0).unwrap().as_str().to_string();
              if let Some(existing_token) = self.reverse_map.get(&matched) {
                  existing_token.clone()
              } else {
                  let token = format!("[{}-{}]", label, self.placeholder_counter);
                  self.placeholder_counter += 1;
                  self.placeholder_map.insert(token.clone(), matched.clone());
                  self.reverse_map.insert(matched.clone(), token.clone());
                  token
              }
          }).into_owned();
      }
      intermediate
  }
  ```
  - We use both `placeholder_map: token → original` and `reverse_map: original → token` to avoid generating multiple tokens for the same sensitive value.  
  - `Captures` is from `use regex::Captures`.

---

## 6. Next Steps for Implementation

1. **Scaffold the project** using `cargo new data_obfuscator --bin`.  
2. **Populate `Cargo.toml`** dependencies as shown above.  
3. **Implement `config.rs`**, then write unit tests to ensure rules load & parse.  
4. **Implement `obfuscator.rs`**, test with a variety of PII patterns.  
5. **Implement `llm_client.rs`**, mock the endpoint locally to verify payload structure.  
6. **Implement `deobfuscator.rs`**, ensure placeholders are correctly replaced.  
7. **Wire everything up in `main.rs`**, adding CLI flags, error handling, and logging.  
8. **Add tests** (`tests/` folder) for each module.  
9. **Dockerize** and run local smoke tests.  
10. **Deploy to staging** (container + Kubernetes), load testing with large documents, and monitor metrics.

By following this **dev_spec.md** as a blueprint, a Rust engineer (or Codex‐based code generator) can systematically build, test, and deploy a robust data‐obfuscation agent that integrates with any LLM provider while safeguarding PII.
