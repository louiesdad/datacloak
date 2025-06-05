# DataCloak CLI

A comprehensive CLI tool for functional testing and analysis of the DataCloak data obfuscation pipeline.

## Features

- **Pattern Detection**: Automatically detect sensitive data patterns in CSV files
- **Data Obfuscation**: Obfuscate sensitive data while preserving structure
- **Full Pipeline Analysis**: Complete detect → obfuscate → LLM analysis workflow
- **Mock LLM Server**: Realistic mock server for testing without API costs
- **Test Scenarios**: Pre-built scenarios for different use cases
- **Integration Testing**: Automated test framework with validation

## Quick Start

### 1. Build the CLI
```bash
cd datacloak-cli
cargo build --release
```

### 2. Run Your First Test
```bash
# Create test scenarios and run customer churn analysis
./target/release/datacloak-cli test-scenario -s customer-churn
```

### 3. Start Mock Server and Run Full Pipeline
```bash
# Terminal 1: Start mock LLM server
./target/release/datacloak-cli mock-server -p 3001 -s customer-churn

# Terminal 2: Run test with mock server
./target/release/datacloak-cli test-scenario -s customer-churn --mock-port 3001
```

## Command Reference

### Pattern Detection
Detect sensitive data patterns in CSV files.

```bash
# Basic pattern detection
./target/release/datacloak-cli detect -f customers.csv -r 100

# With output file
./target/release/datacloak-cli detect -f data/customers.csv -r 500 -o detected_patterns.yaml
```

**Example Output:**
```
🔍 Pattern Detection Results
===========================
Analyzed 100 records

✓ Email (45 instances) - sample: jo***@example.com
✓ Phone (38 instances) - sample: 55***4567
✓ SSN (12 instances) - sample: 12***6789

Save pattern config? (y/n):
```

### Data Obfuscation
Obfuscate data while preserving structure for analysis.

```bash
# Using auto-detected patterns
./target/release/datacloak-cli obfuscate -f customers.csv -r 100

# Using custom pattern file
./target/release/datacloak-cli obfuscate -f customers.csv -p patterns.yaml -r 100 -o obfuscated_data.json

# Process larger dataset
./target/release/datacloak-cli obfuscate -f large_dataset.csv -r 10000 -o results.json
```

**Example Output:**
```
🔒 Obfuscation Results
======================
Processed 100 records
Generated 95 obfuscation tokens
Loaded 3 patterns
```

### Full Pipeline Analysis
Complete workflow: detect → obfuscate → LLM analysis.

```bash
# With real OpenAI API
export OPENAI_API_KEY="your-api-key-here"
./target/release/datacloak-cli analyze -f customers.csv -r 100

# Dry run (no API calls)
./target/release/datacloak-cli analyze -f customers.csv -r 100 --dry-run

# With custom patterns and output
./target/release/datacloak-cli analyze -f customers.csv -p custom_patterns.yaml -r 200 -o analysis_results.txt

# Using specific API key
./target/release/datacloak-cli analyze -f customers.csv -r 100 -k sk-your-api-key
```

**Example Output:**
```
📊 Churn Analysis Results
=========================
Total records processed: 100
Average churn probability: 42.3%
High-risk customers (>70%): 15

🔍 Top Risk Customers:
  Customer CUST-000001: 85.2% churn risk (confidence: 92.1%)
    Reasoning: High support contact (5 tickets), payment delays (3)
  Customer CUST-000047: 78.9% churn risk (confidence: 87.3%)
    Reasoning: Multiple risk indicators
```

### Mock LLM Server
Start a realistic mock server for testing without API costs.

```bash
# Basic mock server
./target/release/datacloak-cli mock-server

# With specific port and scenario
./target/release/datacloak-cli mock-server -p 3001 -s customer-churn

# For medical data analysis
./target/release/datacloak-cli mock-server -p 8080 -s medical-records

# For fraud detection
./target/release/datacloak-cli mock-server -p 3001 -s financial-fraud
```

**Server Features:**
- OpenAI-compatible API endpoints
- Realistic latency (50-200ms)
- Rate limiting (10 req/s)
- 2% error simulation
- Context-aware responses

**Health Check:**
```bash
curl http://localhost:3001/health
# Response: OK
```

### Test Scenarios
Run pre-built test scenarios with validation.

```bash
# Run single scenario
./target/release/datacloak-cli test-scenario -s customer-churn

# Run with mock server
./target/release/datacloak-cli test-scenario -s customer-churn --mock-port 3001

# Run all scenarios
./target/release/datacloak-cli test-all

# Run all with custom mock port
./target/release/datacloak-cli test-all --mock-port 8080
```

**Example Test Output:**
```bash
$ ./target/release/datacloak-cli test-all

🧪 Running 3 test scenarios
==============================
✅ customer-churn - PASSED (3.2s)
✅ medical-records - PASSED (2.1s)  
✅ financial-fraud - PASSED (4.7s)

📊 Test Summary
===============
Total scenarios: 3
Passed: 3 ✅
Failed: 0 ❌

🎉 All tests passed!
```

## Practical Examples

### Example 1: Customer Support Analysis
```bash
# 1. Detect PII in customer support data
./target/release/datacloak-cli detect -f support_tickets.csv -r 1000 -o support_patterns.yaml

# 2. Obfuscate the data
./target/release/datacloak-cli obfuscate -f support_tickets.csv -p support_patterns.yaml -r 1000 -o obfuscated_tickets.json

# 3. Analyze sentiment while protecting privacy
./target/release/datacloak-cli analyze -f support_tickets.csv -p support_patterns.yaml -r 1000 --dry-run
```

### Example 2: Healthcare Data Pipeline
```bash
# 1. Start medical records mock server
./target/release/datacloak-cli mock-server -p 3001 -s medical-records &

# 2. Run medical data analysis test
./target/release/datacloak-cli test-scenario -s medical-records --mock-port 3001

# 3. Custom medical data analysis
./target/release/datacloak-cli analyze -f patient_data.csv -r 500 --dry-run
```

### Example 3: Financial Transaction Monitoring
```bash
# 1. Detect financial PII patterns
./target/release/datacloak-cli detect -f transactions.csv -r 5000

# 2. Run fraud detection with mock
./target/release/datacloak-cli mock-server -p 3001 -s financial-fraud &
./target/release/datacloak-cli test-scenario -s financial-fraud --mock-port 3001
```

### Example 4: Development Workflow
```bash
# 1. Test pattern detection on new dataset
./target/release/datacloak-cli detect -f new_dataset.csv -r 100

# 2. Validate obfuscation works correctly  
./target/release/datacloak-cli obfuscate -f new_dataset.csv -r 100 -o test_obfuscation.json

# 3. Run dry-run analysis to test pipeline
./target/release/datacloak-cli analyze -f new_dataset.csv -r 100 --dry-run

# 4. Run full test suite to ensure everything works
./target/release/datacloak-cli test-all
```

## Command Options Reference

### Global Options
All commands support these options:
- `-h, --help` - Show help information
- `--verbose` - Enable verbose logging

### detect
- `-f, --file <PATH>` - Input CSV file (required)
- `-r, --rows <N>` - Number of rows to analyze (default: 100)
- `-o, --output <PATH>` - Output pattern file

### obfuscate  
- `-f, --file <PATH>` - Input CSV file (required)
- `-p, --patterns <PATH>` - Pattern configuration file
- `-r, --rows <N>` - Number of rows to process (default: 100)
- `-o, --output <PATH>` - Output obfuscated data file

### analyze
- `-f, --file <PATH>` - Input CSV file (required)
- `-r, --rows <N>` - Number of rows to process (default: 100)
- `-p, --patterns <PATH>` - Pattern configuration file
- `-k, --api-key <KEY>` - LLM API key (or use OPENAI_API_KEY env var)
- `--dry-run` - Run without making LLM API calls
- `-o, --output <PATH>` - Output analysis results

### mock-server
- `-p, --port <PORT>` - Server port (default: 3001)
- `-s, --scenario <NAME>` - Scenario type (customer-churn, medical-records, financial-fraud)

### test-scenario
- `-s, --scenario <NAME>` - Scenario name (required)
- `--mock-port <PORT>` - Mock server port (default: 3001)

### test-all
- `--mock-port <PORT>` - Mock server port (default: 3001)

## Test Scenarios

The CLI includes three pre-built test scenarios:

### Customer Churn Analysis
- **Data**: 100 customer records with emails, phones, support tickets
- **Purpose**: Predict customer churn risk
- **Expected Output**: Churn predictions with risk scores

### Medical Records Processing
- **Data**: 50 patient records with SSNs, diagnoses, visit data
- **Purpose**: Analyze health trends while protecting PII
- **Expected Output**: Anonymized health insights

### Financial Fraud Detection
- **Data**: 200 transaction records with credit cards, amounts, locations
- **Purpose**: Detect fraudulent transaction patterns
- **Expected Output**: Fraud risk assessments

## Architecture

### Mock LLM Server
- Realistic API behavior (latency, rate limits, errors)
- Context-aware responses based on obfuscated data
- Deterministic but varied output for consistent testing
- Supports multiple scenarios with different response patterns

### Test Framework
- Automated validation of results
- Performance metrics collection
- Error handling and reporting
- Configurable expectations per scenario

## Example Workflow

1. **Start Mock Server**:
   ```bash
   datacloak-cli mock-server -p 3001 -s customer-churn
   ```

2. **Run Test Scenario**:
   ```bash
   datacloak-cli test-scenario -s customer-churn --mock-port 3001
   ```

3. **Expected Output**:
   ```
   ✅ Test scenario 'customer-churn' PASSED
      Duration: 2.3s
      Records processed: 100
      Patterns detected: 3
      Predictions generated: 100
   ```

## Configuration

### Pattern Files (YAML)
```yaml
patterns:
  - type: email
    priority: 100
    enabled: true
  - type: phone
    priority: 90
    enabled: true
  - type: ssn
    priority: 95
    enabled: true
```

### Scenario Configuration
```yaml
scenario:
  name: "Customer Churn Analysis"
  description: "Detect churn risk in customer data"
  data_file: "customer_data.csv"
  expected_patterns:
    - email
    - phone
    - ssn
  llm_context: "Analyze customer behavior for churn risk"
```

## Development

### Running Tests
```bash
# Build the CLI
cargo build

# Run integration tests
cargo test

# Test specific scenario
./target/debug/datacloak-cli test-scenario -s customer-churn
```

### Adding New Scenarios
1. Create scenario directory: `test-scenarios/my-scenario/`
2. Add `config.yaml` with scenario definition
3. Create sample data CSV file
4. Update test expectations in config

## Performance

The mock LLM server simulates realistic conditions:
- **Latency**: 50-200ms per request
- **Rate Limiting**: 10 requests/second
- **Error Rate**: 2% random failures
- **Batch Processing**: 10 records per LLM call

## Troubleshooting

### Common Issues

1. **Mock server not responding**:
   ```bash
   # Check if server is running
   curl http://localhost:3001/health
   ```

2. **Test data not found**:
   ```bash
   # Scenarios auto-create data on first run
   datacloak-cli test-scenario -s customer-churn
   ```

3. **Build errors**:
   ```bash
   # Ensure datacloak-core is built first
   cd ../datacloak-core && cargo build
   cd ../datacloak-cli && cargo build
   ```