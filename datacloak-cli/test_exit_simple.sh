#!/bin/bash

echo "=== Testing: cargo run -- obfuscate --dry-run sample.csv ==="
echo

# First build in release mode to avoid warnings
echo "Building release version..."
cargo build --release 2>/dev/null

echo
echo "Running: ./target/release/datacloak-cli obfuscate --dry-run --file sample.csv"
echo

# Run the command and capture exit code
./target/release/datacloak-cli obfuscate --dry-run --file sample.csv
EXIT_CODE=$?

echo
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS: Command returned exit code 0"
else
    echo "❌ FAILURE: Command returned exit code $EXIT_CODE"
fi

# Verify JSON output by running again and parsing
echo
echo "Verifying JSON output..."
OUTPUT=$(./target/release/datacloak-cli obfuscate --dry-run --file sample.csv 2>&1 | tail -n +3)
if echo "$OUTPUT" | jq . > /dev/null 2>&1; then
    echo "✅ SUCCESS: Valid JSON summary produced"
    echo
    echo "Summary fields:"
    echo "$OUTPUT" | jq '{mode, input_file, records_processed, patterns_loaded}'
else
    echo "❌ FAILURE: Invalid JSON output"
fi