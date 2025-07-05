function testDashboard() {
    return {
        // State
        stats: {
            total: 0,
            passed: 0,
            failed: 0,
            passRate: 0
        },
        unitTests: [],
        integrationTests: [],
        performanceTests: [],
        allTests: [],
        showModal: false,
        selectedTest: {},
        autoRefresh: false,
        filterStatus: 'all',
        lastUpdated: new Date().toLocaleString(),
        refreshInterval: null,
        isRunning: false,
        selectedForRun: new Set(),
        charts: {
            duration: null,
            trend: null
        },

        // Initialize
        init() {
            this.loadTestData();
            this.initCharts();
            
            if (this.autoRefresh) {
                this.startAutoRefresh();
            }
        },

        // Load test data from server
        async loadTestData() {
            try {
                const response = await fetch('/api/results');
                const data = await response.json();
                
                // Clear existing data
                this.unitTests = [];
                this.integrationTests = [];
                this.performanceTests = [];
                
                // Categorize tests
                if (data.results && data.results.length > 0) {
                    data.results.forEach(test => {
                        if (test.name.includes('adaptive_sampling') || 
                            test.name.includes('bounded_cache') || 
                            test.name.includes('bounded_obfuscator') || 
                            test.name.includes('thread_config')) {
                            this.unitTests.push(test);
                        } else if (test.name.includes('integration') || 
                                   test.name.includes('streaming_detection')) {
                            this.integrationTests.push(test);
                        } else if (test.name.includes('performance')) {
                            this.performanceTests.push(test);
                        }
                    });
                } else {
                    // If no results yet, show default structure
                    this.unitTests = [
                        { name: 'adaptive_sampling', status: 'pending', duration: '-', total: 0, passed: 0, failed: 0, output: 'Not run yet' },
                        { name: 'bounded_cache', status: 'pending', duration: '-', total: 0, passed: 0, failed: 0, output: 'Not run yet' },
                        { name: 'bounded_obfuscator', status: 'pending', duration: '-', total: 0, passed: 0, failed: 0, output: 'Not run yet' },
                        { name: 'thread_config', status: 'pending', duration: '-', total: 0, passed: 0, failed: 0, output: 'Not run yet' }
                    ];
                    
                    this.integrationTests = [
                        { name: 'enhanced_integration', status: 'pending', duration: '-', total: 0, passed: 0, failed: 0, output: 'Not run yet' },
                        { name: 'streaming_detection', status: 'pending', duration: '-', total: 0, passed: 0, failed: 0, output: 'Not run yet' }
                    ];
                    
                    this.performanceTests = [
                        { name: 'performance_optimization', status: 'pending', duration: '-', total: 0, passed: 0, failed: 0, output: 'Not run yet' },
                        { name: 'performance_regression', status: 'pending', duration: '-', total: 0, passed: 0, failed: 0, output: 'Not run yet' }
                    ];
                }
                
                this.allTests = [...this.unitTests, ...this.integrationTests, ...this.performanceTests];
                this.calculateStats();
                this.updateCharts();
                this.lastUpdated = new Date().toLocaleString();
                
                // Update running status
                this.isRunning = data.is_running || false;
                
            } catch (error) {
                console.error('Failed to load test data:', error);
            }
        },

        // Calculate statistics
        calculateStats() {
            this.stats.total = 0;
            this.stats.passed = 0;
            this.stats.failed = 0;

            this.allTests.forEach(test => {
                this.stats.total += test.total;
                this.stats.passed += test.passed;
                this.stats.failed += test.failed;
            });

            this.stats.passRate = this.stats.total > 0 
                ? Math.round((this.stats.passed / this.stats.total) * 100)
                : 0;
        },

        // Get test class based on status
        getTestClass(status) {
            switch(status) {
                case 'passed': return 'test-passed';
                case 'failed': return 'test-failed';
                case 'running': return 'test-running animate-pulse-slow';
                case 'pending': return 'test-pending';
                default: return 'test-pending';
            }
        },

        // Get status icon
        getStatusIcon(status) {
            switch(status) {
                case 'passed': return 'fa-check-circle text-green-600';
                case 'failed': return 'fa-times-circle text-red-600';
                case 'running': return 'fa-spinner fa-spin text-yellow-600';
                case 'pending': return 'fa-clock text-gray-400';
                default: return 'fa-clock text-gray-400';
            }
        },

        // Show test details
        showTestDetails(test) {
            this.selectedTest = test;
            this.showModal = true;
        },

        // Run all tests
        async runAllTests() {
            if (this.isRunning) {
                alert('Tests are already running!');
                return;
            }
            
            try {
                // Mark all tests as running in UI
                this.allTests.forEach(test => {
                    test.status = 'running';
                });
                
                // Call API to start tests
                const response = await fetch('/api/run', { method: 'POST' });
                const data = await response.json();
                
                if (data.status === 'started') {
                    console.log('Tests started successfully');
                    // Start polling for results
                    this.pollForResults();
                } else if (data.error) {
                    alert('Error starting tests: ' + data.error);
                }
            } catch (error) {
                console.error('Failed to start tests:', error);
                alert('Failed to start tests. Check console for details.');
            }
        },

        // Run selected tests
        async runSelectedTests() {
            const selectedTests = Array.from(this.selectedForRun);
            
            if (selectedTests.length === 0) {
                alert('Please select at least one test to run');
                return;
            }
            
            // Mark selected tests as running
            this.allTests.forEach(test => {
                if (selectedTests.includes(test.name)) {
                    test.status = 'running';
                }
            });
            
            // Run each selected test
            for (const testName of selectedTests) {
                try {
                    const response = await fetch(`/api/run/${testName}`, { method: 'POST' });
                    const result = await response.json();
                    console.log(`Test ${testName} result:`, result);
                    
                    // Update test in the UI
                    const test = this.allTests.find(t => t.name === testName);
                    if (test) {
                        Object.assign(test, result);
                    }
                } catch (error) {
                    console.error(`Failed to run test ${testName}:`, error);
                    // Mark as failed
                    const test = this.allTests.find(t => t.name === testName);
                    if (test) {
                        test.status = 'failed';
                        test.output = `Error: ${error.message}`;
                    }
                }
            }
            
            // Reload results after all tests complete
            setTimeout(() => this.loadTestData(), 1000);
        },
        
        // Toggle test selection
        toggleTestSelection(testName) {
            if (this.selectedForRun.has(testName)) {
                this.selectedForRun.delete(testName);
            } else {
                this.selectedForRun.add(testName);
            }
        },
        
        // Poll for test results
        pollForResults() {
            const pollInterval = setInterval(async () => {
                await this.loadTestData();
                
                // Stop polling if tests are no longer running
                if (!this.isRunning) {
                    clearInterval(pollInterval);
                }
            }, 1000);
        },

        // Clear results
        clearResults() {
            this.allTests.forEach(test => {
                test.status = 'pending';
                test.passed = 0;
                test.failed = 0;
                test.output = '';
            });
            this.calculateStats();
            this.updateCharts();
        },

        // Filter tests
        filterTests() {
            // Implementation for filtering tests based on status
            console.log('Filtering by:', this.filterStatus);
        },

        // Auto refresh
        startAutoRefresh() {
            this.refreshInterval = setInterval(() => {
                this.loadTestData();
            }, 5000);
        },

        stopAutoRefresh() {
            if (this.refreshInterval) {
                clearInterval(this.refreshInterval);
                this.refreshInterval = null;
            }
        },

        // Initialize charts
        initCharts() {
            // Duration chart
            const durationCtx = document.getElementById('durationChart').getContext('2d');
            this.charts.duration = new Chart(durationCtx, {
                type: 'bar',
                data: {
                    labels: ['Unit Tests', 'Integration Tests', 'Performance Tests'],
                    datasets: [{
                        label: 'Average Duration (seconds)',
                        data: [0, 0, 0],
                        backgroundColor: ['#3B82F6', '#10B981', '#8B5CF6']
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

            // Trend chart
            const trendCtx = document.getElementById('trendChart').getContext('2d');
            this.charts.trend = new Chart(trendCtx, {
                type: 'line',
                data: {
                    labels: ['Run 1', 'Run 2', 'Run 3', 'Run 4', 'Run 5'],
                    datasets: [{
                        label: 'Pass Rate %',
                        data: [95, 97, 94, 98, 100],
                        borderColor: '#10B981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });
        },

        // Update charts
        updateCharts() {
            if (!this.charts.duration || !this.charts.trend) return;

            // Calculate average durations
            const unitAvg = this.calculateAverageDuration(this.unitTests);
            const integrationAvg = this.calculateAverageDuration(this.integrationTests);
            const performanceAvg = this.calculateAverageDuration(this.performanceTests);

            // Update duration chart
            this.charts.duration.data.datasets[0].data = [unitAvg, integrationAvg, performanceAvg];
            this.charts.duration.update();

            // Update trend chart with current pass rate
            const trendData = this.charts.trend.data.datasets[0].data;
            trendData.push(this.stats.passRate);
            if (trendData.length > 10) {
                trendData.shift();
            }
            this.charts.trend.update();
        },

        // Calculate average duration
        calculateAverageDuration(tests) {
            if (tests.length === 0) return 0;
            const total = tests.reduce((sum, test) => {
                const duration = parseFloat(test.duration.replace('s', ''));
                return sum + duration;
            }, 0);
            return (total / tests.length).toFixed(2);
        },

        // Watch for autoRefresh changes
        $watch('autoRefresh', (value) => {
            if (value) {
                this.startAutoRefresh();
            } else {
                this.stopAutoRefresh();
            }
        })
    };
}