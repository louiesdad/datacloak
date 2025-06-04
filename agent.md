# agent.md

## Overview

**Purpose:**  
This agent implements a secure, performant pipeline that:

1. Fetches a customer record (or arbitrary large document) from a local datastore.  
2. Excerpts and obfuscates all sensitive data (PII/PHI) before sending to an LLM (e.g., OpenAI).  
3. Sends the obfuscated payload to the LLM’s API, awaits a response.  
4. Re-inserts (“de-obfuscates”) real sensitive values into the LLM’s output.  
5. Returns the fully reconstituted response to the caller, ensuring that at no point is raw PII sent externally.

**Why Rust?**  
- Near–C performance for regex‐heavy obfuscation (often 10×–50× faster than Python).  
- Zero-cost abstractions (`regex`, `tokio`, `reqwest`) and memory safety (no GC pauses).  
- Single static binary—easy distribution on Linux/Mac/Windows without runtime dependencies.  
- Excellent async I/O for streaming large documents or handling multiple concurrent requests.

---

## High-Level Data Flow

```mermaid
flowchart LR
    A[Client Request<br/>(e.g., “Process Customer 12345”)] --> B[Fetch Record<br/> (DB or File)]
    B --> C[Assemble “Raw Document”]
    C --> D[Obfuscator Engine<br/>(Rule-Based Regex + Optional ML)] 
    D --> E[Obfuscated Payload] 
    E --> F[LLM API Call<br/>(OpenAI, Azure, etc.)] 
    F --> G[Obfuscated LLM Response]
    G --> H[De-obfuscator Module] 
    H --> I[Final Response<br/>(with real PII/PHI restored)] 
    I --> J[Return to Client]
```

1. **Fetch Record/Document (B):**  
   - Query relational DB (e.g., SQL Server, PostgreSQL) or read a local file (JSON, CSV, TXT).  
   - Deserialize into a Rust struct or stream text.

2. **Assemble Raw Document (C):**  
   - If multiple fields (e.g., `first_name`, `last_name`, `email`, `notes`), concatenate into one text blob.  
   - If a large file (>MBs), treat as a streaming source (line by line or fixed‐size chunks).

3. **Obfuscator Engine (D):**  
   - Load a JSON/YAML config of “sensitive‐field” rules—each rule has a regex pattern and replacement label (e.g., `EMAIL`, `SSN`, `PHONE`).  
   - For each rule:  
     1. Compile using [`regex::Regex`](https://docs.rs/regex/latest/regex/).  
     2. Walk the text (streamed or in memory), replacing each match with a unique token `"[LABEL-<ID>]"`.  
     3. Store a `HashMap<String, String>` mapping token → original value.  
   - Optionally: if using an ML‐based Named Entity Recognizer (NER), load an ONNX or TensorFlow‐rust model (e.g., [`rust-bert`](https://github.com/guillaume-be/rust-bert) with an ONNX export for PII detection).  
   - Produce a fully obfuscated text payload (no raw PII remains).

4. **LLM API Call (F):**  
   - Build JSON according to the chosen LLM’s specification (e.g., OpenAI Chat Completions API).  
     ```jsonc
     {
       "model": "gpt-4",
       "messages": [
         { "role": "system", "content": "You are a secure data processor." },
         { "role": "user", "content": "<OBFUSCATED_TEXT>" }
       ]
     }
     ```
   - Send via [`reqwest::Client`](https://docs.rs/reqwest/latest/reqwest/) (async) with `Bearer <API_KEY>` header.  
   - Await status, parse JSON into Rust structs (e.g., using [`serde`](https://docs.rs/serde/latest/serde/) / [`serde_json`](https://docs.rs/serde_json/latest/serde_json/)).

5. **De-obfuscator Module (H):**  
   - Extract the LLM’s reply (still containing tokens like `"[EMAIL-0]"`).  
   - Iterate over the `HashMap<placeholder, original>` and perform `String::replace()` on the LLM response.  
   - Return the final text with all real sensitive values re-inserted.

6. **Return to Client (I → J):**  
   - Package in a JSON object or custom response type (e.g., `{"response": "<DE-OBFUSCATED_TEXT>"}`).  
   - Send HTTP 200 to caller.

---

## Agent Responsibilities & Capabilities

1. **Configuration Loader**  
   - Read `config/obfuscation_rules.json` (or `.yaml`).  
   - Validate each rule’s regex compiles.  
   - Example rule entry:  
     ```jsonc
     {
       "pattern": "\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
       "label": "EMAIL"
     }
     ```

2. **Sensitive Data Scanner & Replacer**  
   - For in‐memory fields < 1 MB: read entire string → run all regex replacements → return obfuscated string.  
   - For large files (>10 MB): use streaming (`tokio::io::BufReader::lines()` or fixed‐size chunk reads).  
   - Always update `HashMap<String, String>`: key = `"[<LABEL>-<AUTO_INCREMENT_ID>]"`, value = raw match.

3. **LLM Client**  
   - Configurable endpoint (e.g., `OPENAI_ENDPOINT=https://api.openai.com/v1/chat/completions`).  
   - Configurable API key via environment variable (`OPENAI_API_KEY`).  
   - Support both JSON and streaming (“chunked”) responses if needed.  

4. **De-obfuscation Layer**  
   - One‐pass string replacements: for each `(token, original)` in placeholder map, run `reply.replace(token, original)`.  
   - If performance is a concern (large output), consider splitting on tokens and rebuilding—though simple `replace` is often adequate.

5. **Error Handling & Logging**  
   - Gracefully handle regex compile errors on startup (fail-fast).  
   - Validate DB connectivity or file existence.  
   - Retry LLM calls on 5xx (with exponential backoff).  
   - Log at `INFO` for requests, `DEBUG` for token mapping, and `ERROR` for failures (use [`tracing`](https://docs.rs/tracing/latest/tracing/) + [`tracing-subscriber`](https://docs.rs/tracing-subscriber/latest/tracing_subscriber/)).

6. **Security & Privacy**  
   - Never log raw PII—only log placeholders or counts.  
   - Keep `placeholder_map` in memory only per‐request; do not persist to disk, or if you must (e.g., for async workflows), encrypt at rest.  
   - Do not include the original raw text in sent payloads; only send obfuscated versions to the LLM.

---

## Key Rust Crates & Reference Material

- **Regex & Streaming I/O**  
  - [`regex`](https://docs.rs/regex/latest/regex/) – official crate for compiled, thread-safe regex.  
  - [`tokio`](https://docs.rs/tokio/latest/tokio/) – async runtime for streaming large files.  
  - [`tokio::io::BufReader`](https://docs.rs/tokio/latest/tokio/io/struct.BufReader.html) – buffered line‐by‐line reading.

- **HTTP & JSON**  
  - [`reqwest`](https://docs.rs/reqwest/latest/reqwest/) – async HTTP client (supports JSON, TLS, gzip).  
  - [`serde`](https://docs.rs/serde/latest/serde/) + [`serde_json`](https://docs.rs/serde_json/latest/serde_json/) – (de)serialization of JSON.  
  - [OpenAI API Reference](https://platform.openai.com/docs/api-reference) – exact request/response schemas.

- **Async Transformers / NER (Optional)**  
  - [`rust-bert`](https://github.com/guillaume-be/rust-bert) – for on‐device BERT‐based NER (export your own ONNX model or use a pretrained one).  
  - [`tract-onnx`](https://github.com/sonos/tract) – run ONNX models directly in Rust.

- **Logging & Configuration**  
  - [`tracing`](https://docs.rs/tracing/latest/tracing/) + [`tracing-subscriber`](https://docs.rs/tracing-subscriber/latest/tracing_subscriber/) – structured logging with spans.  
  - [`config`](https://docs.rs/config/latest/config/) – load layered config (JSON/YAML/ENV) if you need more than just CLI flags.

- **Database Connectivity (if needed)**  
  - [`sqlx`](https://docs.rs/sqlx/latest/sqlx/) – async, compile‐time–checked SQL (PostgreSQL, MySQL, SQLite).  
  - [`tokio-postgres`](https://docs.rs/tokio-postgres/latest/tokio_postgres/) – lower‐level async driver.

---

## Example Endpoint Contracts

1. **HTTP POST /process-customer**  
   - **Request JSON**:  
     ```jsonc
     {
       "customer_id": 12345,
       "mode": "summarize_notes"  // or "flag_risk", etc.
     }
     ```
   - **Response JSON**:  
     ```jsonc
     {
       "status": "ok",
       "analysis": "I recommend contacting John Smith …"
     }
     ```

2. **HTTP POST /process-document** (for arbitrary text)  
   - **Request JSON**:  
     ```jsonc
     {
       "document": "<BASE64_ENCODED_TEXT_OR_RAW_STRING>",
       "task": "extract_key_clauses"
     }
     ```
   - **Response JSON**:  
     ```jsonc
     {
       "status": "ok",
       "result": "Clause 1: …\nClause 2: …"
     }
     ```

---

## Agent Deployment & Scaling

1. **Dockerize**  
   - Build a multi‐stage Dockerfile:  
     ```dockerfile
     FROM rust:1.70 as builder
     WORKDIR /usr/src/app
     COPY Cargo.toml Cargo.lock ./
     COPY src ./src
     RUN cargo build --release

     FROM debian:buster-slim
     COPY --from=builder /usr/src/app/target/release/data_obfuscator /usr/local/bin/data_obfuscator
     ENTRYPOINT ["data_obfuscator"]
     ```
   - Final image ~10–12 MB (Debian + static binary).

2. **Kubernetes**  
   - Deploy as a `Deployment` with autoscaling on CPU/memory.  
   - Configure environment variables for `OPENAI_API_KEY`, `LLM_ENDPOINT`, and `RULES_PATH`.  
   - Expose as a `Service` behind an Ingress (HTTPS).

3. **Monitoring**  
   - Expose Prometheus metrics (e.g., request_count, request_latency, error_count) via [`prometheus`](https://docs.rs/prometheus/latest/prometheus/) crate.  
   - Use Grafana dashboards to visualize throughput vs. latency.

---

## Summary

This `agent.md` defines a Rust‐based obfuscation‐forwarding agent that:

- Safely removes all sensitive data before calling an external LLM.  
- Retains a reversible mapping to re-insert real values into the LLM’s output.  
- Leverages Rust’s performance (regex + async I/O) and binary-distribution model.  
- Provides enough detail and references so that a Rust engineer (or Codex) can scaffold and implement the full pipeline.
