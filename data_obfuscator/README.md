# Data Obfuscator

This project provides a Rust service that obfuscates sensitive text before it is sent to an LLM and then restores the original values in the response.

## Building

```
cargo build --release
```

Or build the Docker image:

```
docker build -t data_obfuscator .
```

## Running

The binary accepts a few CLI flags:

```
./target/release/data_obfuscator \
    --rules config/obfuscation_rules.json \
    --llm-endpoint http://localhost \
    --api-key test-key \
    --input "example text"
```

With Docker:

```
docker run --rm data_obfuscator \
    --rules /app/config/obfuscation_rules.json \
    --llm-endpoint http://localhost \
    --api-key test-key \
    --input "example text"
```

The service will print the de-obfuscated LLM response to stdout.
