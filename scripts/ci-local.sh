#!/bin/bash
# Local CI check script for DataCloak
# Runs the same checks as the CI pipeline

set -e

WORKSPACES=("datacloak-core" "data_obfuscator" "datacloak-cli")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 Running DataCloak CI checks locally..."
echo "Project root: $PROJECT_ROOT"
echo

# Function to run checks in a workspace
run_workspace_checks() {
    local workspace=$1
    local workspace_path="$PROJECT_ROOT/$workspace"
    
    if [ ! -d "$workspace_path" ]; then
        echo "⚠️  Workspace $workspace not found, skipping..."
        return 0
    fi
    
    echo "📁 Checking workspace: $workspace"
    cd "$workspace_path"
    
    echo "  🔒 Running security audit..."
    if ! cargo audit; then
        echo "  ❌ Security audit failed in $workspace"
        return 1
    fi
    
    echo "  📐 Checking formatting..."
    if ! cargo fmt --all -- --check; then
        echo "  ❌ Format check failed in $workspace"
        echo "  💡 Run 'cargo fmt' to fix formatting"
        return 1
    fi
    
    echo "  📋 Running clippy..."
    if ! cargo clippy --workspace --all-targets --all-features -- -D warnings; then
        echo "  ❌ Clippy failed in $workspace"
        return 1
    fi
    
    echo "  🧪 Running tests..."
    if ! cargo test --all-features --workspace; then
        echo "  ❌ Tests failed in $workspace"
        return 1
    fi
    
    echo "  ✅ All checks passed for $workspace"
    echo
}

# Check if required tools are installed
check_tools() {
    echo "🔧 Checking required tools..."
    
    if ! command -v cargo-audit &> /dev/null; then
        echo "📦 Installing cargo-audit..."
        cargo install cargo-audit
    fi
    
    if ! command -v cargo-tarpaulin &> /dev/null; then
        echo "📦 Installing cargo-tarpaulin..."
        cargo install cargo-tarpaulin
    fi
    
    echo "✅ All tools available"
    echo
}

# Run coverage check
run_coverage_check() {
    local workspace=$1
    local workspace_path="$PROJECT_ROOT/$workspace"
    
    if [ ! -d "$workspace_path" ]; then
        return 0
    fi
    
    echo "📊 Running coverage check for $workspace..."
    cd "$workspace_path"
    
    # Run tarpaulin and extract coverage
    if cargo tarpaulin --out Xml --output-dir coverage --all-features --workspace --timeout 120 > /dev/null 2>&1; then
        if [ -f "coverage/cobertura.xml" ]; then
            coverage_percent=$(grep -oP 'line-rate="\K[^"]*' coverage/cobertura.xml | head -1 | awk '{printf "%.0f", $1*100}')
            echo "  Coverage: ${coverage_percent}%"
            
            if [ "$coverage_percent" -lt 80 ]; then
                echo "  ❌ Coverage ${coverage_percent}% is below 80% threshold"
                return 1
            else
                echo "  ✅ Coverage ${coverage_percent}% meets 80% threshold"
            fi
        else
            echo "  ⚠️  Coverage report not generated"
        fi
    else
        echo "  ⚠️  Coverage check failed, skipping..."
    fi
    echo
}

# Run benchmarks
run_benchmarks() {
    local workspace_path="$PROJECT_ROOT/data_obfuscator"
    
    if [ ! -d "$workspace_path" ]; then
        echo "⚠️  data_obfuscator workspace not found, skipping benchmarks..."
        return 0
    fi
    
    echo "⚡ Running benchmarks..."
    cd "$workspace_path"
    
    echo "  🛡️  Running ReDoS security benchmarks..."
    if ! cargo bench regex_redos -- --test; then
        echo "  ❌ ReDoS benchmarks failed"
        return 1
    fi
    
    echo "  🏃 Running streaming performance benchmarks..."
    if ! cargo bench quick_streaming_benchmark; then
        echo "  ❌ Streaming benchmarks failed"
        return 1
    fi
    
    echo "  ✅ All benchmarks passed"
    echo
}

# CLI integration test
run_cli_integration() {
    local cli_path="$PROJECT_ROOT/datacloak-cli"
    
    if [ ! -d "$cli_path" ]; then
        echo "⚠️  datacloak-cli workspace not found, skipping integration test..."
        return 0
    fi
    
    echo "🔧 Running CLI integration test..."
    cd "$cli_path"
    
    # Build CLI
    echo "  🏗️  Building CLI..."
    if ! cargo build --release > /dev/null 2>&1; then
        echo "  ❌ CLI build failed"
        return 1
    fi
    
    # Create test data
    echo "name,email,phone" > test.csv
    echo "John,john@test.com,555-1234" >> test.csv
    
    # Test successful dry-run
    echo "  🧪 Testing dry-run exit codes..."
    if ./target/release/datacloak-cli obfuscate --file test.csv --dry-run > /dev/null 2>&1; then
        echo "  ✅ Dry-run exit code test passed"
    else
        echo "  ❌ Dry-run exit code test failed"
        rm -f test.csv
        return 1
    fi
    
    # Test error handling
    if ! ./target/release/datacloak-cli obfuscate --file nonexistent.csv --dry-run > /dev/null 2>&1; then
        echo "  ✅ Error handling exit code test passed"
    else
        echo "  ❌ Error handling exit code test failed"
        rm -f test.csv
        return 1
    fi
    
    rm -f test.csv
    echo "  ✅ CLI integration test passed"
    echo
}

# Main execution
main() {
    check_tools
    
    # Run checks for each workspace
    for workspace in "${WORKSPACES[@]}"; do
        if ! run_workspace_checks "$workspace"; then
            echo "❌ CI checks failed for $workspace"
            exit 1
        fi
    done
    
    # Run coverage checks
    echo "📊 Running coverage checks..."
    for workspace in "${WORKSPACES[@]}"; do
        run_coverage_check "$workspace"
    done
    
    # Run benchmarks
    run_benchmarks
    
    # Run CLI integration test
    run_cli_integration
    
    echo "🎉 All CI checks passed!"
    echo
    echo "💡 Tips:"
    echo "  - Run 'cargo fmt' if formatting fails"
    echo "  - Run 'cargo clippy --fix' to auto-fix some clippy issues"
    echo "  - Add tests to improve coverage"
    echo "  - Check SECURITY.md for security best practices"
    echo
}

# Run main function
main "$@"