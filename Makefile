# DataCloak Makefile
# Convenient commands for development and testing

.PHONY: help test test-unit test-integration test-perf test-all coverage dashboard clean build release

# Default target
help:
	@echo "DataCloak Development Commands:"
	@echo "  make test          - Run all tests"
	@echo "  make test-unit     - Run unit tests only"
	@echo "  make test-integration - Run integration tests"
	@echo "  make test-perf     - Run performance tests"
	@echo "  make coverage      - Generate test coverage report"
	@echo "  make dashboard     - Start test dashboard server"
	@echo "  make build         - Build all crates"
	@echo "  make release       - Build release version"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make bench         - Run benchmarks"

# Test targets
test:
	@echo "Running all tests..."
	@cd datacloak-core && cargo test

test-unit:
	@echo "Running unit tests..."
	@cd datacloak-core && cargo test --test adaptive_sampling_tests
	@cd datacloak-core && cargo test --test bounded_cache_tests
	@cd datacloak-core && cargo test --test bounded_obfuscator_tests
	@cd datacloak-core && cargo test --test thread_config_tests

test-integration:
	@echo "Running integration tests..."
	@cd datacloak-core && cargo test --test enhanced_integration_tests
	@cd datacloak-core && cargo test --test streaming_detection_tests

test-perf:
	@echo "Running performance tests..."
	@cd datacloak-core && cargo test --test performance_optimization_tests --release
	@cd datacloak-core && cargo test --test performance_regression_tests --release

test-all: test-unit test-integration test-perf

# Coverage
coverage:
	@echo "Generating test coverage..."
	@cd datacloak-core && cargo tarpaulin --out Html --output-dir ../test-results/coverage

# Dashboard
dashboard:
	@echo "Starting test dashboard..."
	@python3 test-dashboard/server.py

dashboard-simple:
	@echo "Opening simple dashboard..."
	@./test-runner.sh

# Build targets
build:
	@echo "Building all crates..."
	@cargo build --all

release:
	@echo "Building release version..."
	@cargo build --all --release

# Benchmarks
bench:
	@echo "Running benchmarks..."
	@cd datacloak-core && cargo bench

# Clean
clean:
	@echo "Cleaning build artifacts..."
	@cargo clean
	@rm -rf test-results/

# Development helpers
fmt:
	@echo "Formatting code..."
	@cargo fmt --all

lint:
	@echo "Running clippy..."
	@cargo clippy --all -- -D warnings

check:
	@echo "Checking code..."
	@cargo check --all

# Quick test for CI
ci-test: fmt lint check test

# Install development dependencies
dev-setup:
	@echo "Installing development dependencies..."
	@cargo install cargo-tarpaulin
	@cargo install cargo-watch
	@cargo install cargo-edit
	@pip3 install aiohttp aiohttp-cors

# Watch for changes and run tests
watch:
	@echo "Watching for changes..."
	@cargo watch -x test

# Run a specific test by name
test-specific:
	@echo "Running test: $(TEST)"
	@cd datacloak-core && cargo test $(TEST)

# Performance profiling
profile:
	@echo "Running performance profiling..."
	@cd datacloak-core && cargo build --release
	@cd datacloak-core && CARGO_PROFILE_RELEASE_DEBUG=true cargo flamegraph --test enhanced_integration_tests

# Documentation
docs:
	@echo "Generating documentation..."
	@cargo doc --all --no-deps --open

# Example usage
example:
	@echo "Running example..."
	@cd datacloak-cli && cargo run -- detect --input data/sample.csv