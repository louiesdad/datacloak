FROM rust:1.70 AS builder
WORKDIR /usr/src/app
COPY Cargo.toml ./
COPY src ./src
RUN cargo build --release

FROM debian:buster-slim
COPY --from=builder /usr/src/app/target/release/data_obfuscator /usr/local/bin/data_obfuscator
ENTRYPOINT ["data_obfuscator"]
