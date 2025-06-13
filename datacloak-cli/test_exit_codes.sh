#!/bin/bash
set -e

echo "=== Testing datacloak-cli exit codes ==="
echo

# Test 1: Dry run with valid file - should return 0
echo "Test 1: Dry run with valid file"
# Run command and capture full output
OUTPUT=$(cargo run --quiet -- obfuscate --dry-run --file sample.csv 2>&1)
EXIT_CODE=$?
echo "Exit code: $EXIT_CODE"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ PASS: Exit code is 0"
else
    echo "❌ FAIL: Expected exit code 0, got $EXIT_CODE"
fi

# Extract JSON from output (skip header lines)
echo "$OUTPUT" | tail -n +3 > /tmp/dry_run_output.json

# Verify JSON output
if jq . /tmp/dry_run_output.json > /dev/null 2>&1; then
    echo "✅ PASS: Valid JSON output produced"
    echo "Summary:"
    jq '{mode, records_processed, patterns_loaded}' /tmp/dry_run_output.json
else
    echo "❌ FAIL: Invalid JSON output"
    echo "Output was:"
    head -5 /tmp/dry_run_output.json
fi
echo

# Test 2: Dry run with non-existent file - should return non-zero
echo "Test 2: Dry run with non-existent file"
cargo run --quiet -- obfuscate --dry-run --file nonexistent.csv 2>/dev/null
EXIT_CODE=$?
echo "Exit code: $EXIT_CODE"
if [ $EXIT_CODE -ne 0 ]; then
    echo "✅ PASS: Exit code is non-zero for missing file"
else
    echo "❌ FAIL: Expected non-zero exit code, got $EXIT_CODE"
fi
echo

# Test 3: Normal mode creating output file - should return 0
echo "Test 3: Normal mode with output file"
rm -f /tmp/test_output.json
cargo run --quiet -- obfuscate --file sample.csv --output /tmp/test_output.json --rows 2 2>/dev/null
EXIT_CODE=$?
echo "Exit code: $EXIT_CODE"
if [ $EXIT_CODE -eq 0 ] && [ -f /tmp/test_output.json ]; then
    echo "✅ PASS: Exit code is 0 and output file created"
else
    echo "❌ FAIL: Expected exit code 0 and output file"
fi
echo

# Test 4: Invalid arguments - should return non-zero
echo "Test 4: Invalid arguments"
cargo run --quiet -- obfuscate --invalid-flag 2>/dev/null
EXIT_CODE=$?
echo "Exit code: $EXIT_CODE" 
if [ $EXIT_CODE -ne 0 ]; then
    echo "✅ PASS: Exit code is non-zero for invalid arguments"
else
    echo "❌ FAIL: Expected non-zero exit code, got $EXIT_CODE"
fi
echo

echo "=== Exit code tests complete ==="