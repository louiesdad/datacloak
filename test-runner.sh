#!/bin/bash

# DataCloak Test Runner with Dashboard
# This script runs all tests and generates a comprehensive test report

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test categories
UNIT_TESTS=("adaptive_sampling" "bounded_cache" "bounded_obfuscator" "thread_config")
INTEGRATION_TESTS=("enhanced_integration" "streaming_detection")
PERFORMANCE_TESTS=("performance_optimization" "performance_regression")
ALL_TESTS=("${UNIT_TESTS[@]}" "${INTEGRATION_TESTS[@]}" "${PERFORMANCE_TESTS[@]}")

# Output directory
OUTPUT_DIR="test-results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_DIR="${OUTPUT_DIR}/${TIMESTAMP}"

# Create output directories
mkdir -p "${REPORT_DIR}/logs"
mkdir -p "${REPORT_DIR}/coverage"

# Dashboard HTML file
DASHBOARD="${REPORT_DIR}/dashboard.html"

# Function to run a test and capture output
run_test() {
    local test_name=$1
    local test_type=$2
    local log_file="${REPORT_DIR}/logs/${test_name}.log"
    local json_file="${REPORT_DIR}/logs/${test_name}.json"
    
    echo -e "${BLUE}Running ${test_type} test: ${test_name}${NC}"
    
    # Run test with JSON output
    if CARGO_TERM_COLOR=always cargo test --test "${test_name}_tests" -- --nocapture --test-threads=4 \
        2>&1 | tee "${log_file}" | grep -E "(test result:|running|finished|FAILED|passed)"; then
        echo -e "${GREEN}✓ ${test_name} passed${NC}"
        echo "PASSED" > "${REPORT_DIR}/logs/${test_name}.status"
        return 0
    else
        echo -e "${RED}✗ ${test_name} failed${NC}"
        echo "FAILED" > "${REPORT_DIR}/logs/${test_name}.status"
        return 1
    fi
}

# Function to generate test metrics
generate_metrics() {
    local log_file=$1
    local metrics_file="${log_file%.log}.metrics"
    
    # Extract test metrics from log
    local total_tests=$(grep -c "test .* \.\.\." "${log_file}" 2>/dev/null || echo "0")
    local passed_tests=$(grep -c "test .* \.\.\. ok" "${log_file}" 2>/dev/null || echo "0")
    local failed_tests=$(grep -c "test .* \.\.\. FAILED" "${log_file}" 2>/dev/null || echo "0")
    local test_time=$(grep "test result:" "${log_file}" 2>/dev/null | sed -E 's/.*finished in ([0-9.]+)s.*/\1/' || echo "0")
    
    cat > "${metrics_file}" <<EOF
{
    "total": ${total_tests},
    "passed": ${passed_tests},
    "failed": ${failed_tests},
    "duration": "${test_time}s"
}
EOF
}

# Function to start the dashboard HTML
init_dashboard() {
    cat > "${DASHBOARD}" <<'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DataCloak Test Dashboard</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .summary-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .summary-card h3 {
            margin: 0;
            color: #666;
            font-size: 14px;
            text-transform: uppercase;
        }
        .summary-card .value {
            font-size: 36px;
            font-weight: bold;
            margin: 10px 0;
        }
        .passed { color: #4CAF50; }
        .failed { color: #f44336; }
        .warning { color: #ff9800; }
        .test-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .test-card {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .test-card.passed { border-top: 4px solid #4CAF50; }
        .test-card.failed { border-top: 4px solid #f44336; }
        .test-card.skipped { border-top: 4px solid #ff9800; }
        .test-header {
            padding: 15px;
            background: #fafafa;
            border-bottom: 1px solid #eee;
        }
        .test-header h3 {
            margin: 0;
            font-size: 18px;
        }
        .test-body {
            padding: 15px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 5px 0;
        }
        .metric-label {
            color: #666;
        }
        .metric-value {
            font-weight: bold;
        }
        .log-link {
            display: inline-block;
            margin-top: 10px;
            color: #2196F3;
            text-decoration: none;
        }
        .log-link:hover {
            text-decoration: underline;
        }
        .performance-chart {
            margin: 20px 0;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .chart-container {
            position: relative;
            height: 300px;
        }
        .timestamp {
            color: #666;
            font-size: 14px;
            margin-top: 20px;
        }
        pre {
            background: #f5f5f5;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 12px;
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <h1>DataCloak Test Dashboard</h1>
EOF
}

# Function to add test result to dashboard
add_test_to_dashboard() {
    local test_name=$1
    local test_type=$2
    local status_file="${REPORT_DIR}/logs/${test_name}.status"
    local metrics_file="${REPORT_DIR}/logs/${test_name}.metrics"
    local log_file="logs/${test_name}.log"
    
    local status="UNKNOWN"
    if [ -f "${status_file}" ]; then
        status=$(cat "${status_file}")
    fi
    
    local status_class="skipped"
    if [ "${status}" = "PASSED" ]; then
        status_class="passed"
    elif [ "${status}" = "FAILED" ]; then
        status_class="failed"
    fi
    
    # Read metrics if available
    local total="0"
    local passed="0"
    local failed="0"
    local duration="0s"
    
    if [ -f "${metrics_file}" ]; then
        total=$(grep '"total"' "${metrics_file}" | sed 's/.*: \([0-9]*\).*/\1/')
        passed=$(grep '"passed"' "${metrics_file}" | sed 's/.*: \([0-9]*\).*/\1/')
        failed=$(grep '"failed"' "${metrics_file}" | sed 's/.*: \([0-9]*\).*/\1/')
        duration=$(grep '"duration"' "${metrics_file}" | sed 's/.*: "\([^"]*\)".*/\1/')
    fi
    
    cat >> "${DASHBOARD}" <<EOF
        <div class="test-card ${status_class}">
            <div class="test-header">
                <h3>${test_name}</h3>
                <span style="color: #666; font-size: 14px;">${test_type}</span>
            </div>
            <div class="test-body">
                <div class="metric">
                    <span class="metric-label">Status:</span>
                    <span class="metric-value ${status_class}">${status}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Tests:</span>
                    <span class="metric-value">${total}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Passed:</span>
                    <span class="metric-value passed">${passed}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Failed:</span>
                    <span class="metric-value failed">${failed}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Duration:</span>
                    <span class="metric-value">${duration}</span>
                </div>
                <a href="${log_file}" class="log-link">View Log →</a>
            </div>
        </div>
EOF
}

# Function to finalize dashboard
finalize_dashboard() {
    local total_tests=0
    local total_passed=0
    local total_failed=0
    local total_duration=0
    
    # Calculate totals
    for metrics_file in "${REPORT_DIR}/logs/"*.metrics; do
        if [ -f "${metrics_file}" ]; then
            local tests=$(grep '"total"' "${metrics_file}" | sed 's/.*: \([0-9]*\).*/\1/')
            local passed=$(grep '"passed"' "${metrics_file}" | sed 's/.*: \([0-9]*\).*/\1/')
            local failed=$(grep '"failed"' "${metrics_file}" | sed 's/.*: \([0-9]*\).*/\1/')
            
            total_tests=$((total_tests + tests))
            total_passed=$((total_passed + passed))
            total_failed=$((total_failed + failed))
        fi
    done
    
    local pass_rate=0
    if [ ${total_tests} -gt 0 ]; then
        pass_rate=$((total_passed * 100 / total_tests))
    fi
    
    # Add summary at the beginning of the file
    local temp_file="${DASHBOARD}.tmp"
    
    # Copy header
    head -n 150 "${DASHBOARD}" > "${temp_file}"
    
    # Add summary
    cat >> "${temp_file}" <<EOF
        <div class="summary">
            <div class="summary-card">
                <h3>Total Tests</h3>
                <div class="value">${total_tests}</div>
            </div>
            <div class="summary-card">
                <h3>Passed</h3>
                <div class="value passed">${total_passed}</div>
            </div>
            <div class="summary-card">
                <h3>Failed</h3>
                <div class="value failed">${total_failed}</div>
            </div>
            <div class="summary-card">
                <h3>Pass Rate</h3>
                <div class="value ${pass_rate -ge 80 && echo "passed" || echo "warning"}">${pass_rate}%</div>
            </div>
        </div>
        
        <h2>Test Results</h2>
        <div class="test-grid">
EOF
    
    # Copy test results
    tail -n +151 "${DASHBOARD}" >> "${temp_file}"
    
    # Add footer
    cat >> "${temp_file}" <<EOF
        </div>
        
        <div class="performance-chart">
            <h2>Performance Metrics</h2>
            <canvas id="performanceChart"></canvas>
        </div>
        
        <p class="timestamp">Generated at: $(date)</p>
    </div>
    
    <script>
        // Performance chart
        const ctx = document.getElementById('performanceChart').getContext('2d');
        const performanceChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Unit Tests', 'Integration Tests', 'Performance Tests'],
                datasets: [{
                    label: 'Test Duration (seconds)',
                    data: [2.5, 5.3, 8.1], // Placeholder data
                    backgroundColor: ['#4CAF50', '#2196F3', '#FF9800']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    </script>
</body>
</html>
EOF
    
    mv "${temp_file}" "${DASHBOARD}"
}

# Main execution
echo -e "${BLUE}DataCloak Test Runner${NC}"
echo "====================="
echo "Output directory: ${REPORT_DIR}"
echo ""

# Initialize dashboard
init_dashboard

# Run unit tests
echo -e "\n${YELLOW}Running Unit Tests...${NC}"
for test in "${UNIT_TESTS[@]}"; do
    run_test "${test}" "Unit" || true
    generate_metrics "${REPORT_DIR}/logs/${test}.log"
    add_test_to_dashboard "${test}" "Unit"
done

# Run integration tests
echo -e "\n${YELLOW}Running Integration Tests...${NC}"
for test in "${INTEGRATION_TESTS[@]}"; do
    run_test "${test}" "Integration" || true
    generate_metrics "${REPORT_DIR}/logs/${test}.log"
    add_test_to_dashboard "${test}" "Integration"
done

# Run performance tests if requested
if [ "$1" = "--with-perf" ]; then
    echo -e "\n${YELLOW}Running Performance Tests...${NC}"
    for test in "${PERFORMANCE_TESTS[@]}"; do
        run_test "${test}" "Performance" || true
        generate_metrics "${REPORT_DIR}/logs/${test}.log"
        add_test_to_dashboard "${test}" "Performance"
    done
fi

# Run coverage if requested
if [ "$1" = "--with-coverage" ]; then
    echo -e "\n${YELLOW}Generating Code Coverage...${NC}"
    cargo tarpaulin --out Html --output-dir "${REPORT_DIR}/coverage" || true
fi

# Finalize dashboard
finalize_dashboard

echo -e "\n${GREEN}Test run complete!${NC}"
echo "Dashboard available at: ${DASHBOARD}"

# Open dashboard in browser if available
if command -v open &> /dev/null; then
    open "${DASHBOARD}"
elif command -v xdg-open &> /dev/null; then
    xdg-open "${DASHBOARD}"
fi