-- Initial schema for DataCloak API v2

-- Files table to store uploaded file metadata
CREATE TABLE IF NOT EXISTS files (
    file_id UUID PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    row_count INTEGER,
    column_count INTEGER,
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Analysis runs table (single column version)
CREATE TABLE IF NOT EXISTS analysis_runs_v1 (
    run_id UUID PRIMARY KEY,
    file_id UUID NOT NULL REFERENCES files(file_id),
    selected_column VARCHAR(100) NOT NULL,
    chain_type VARCHAR(50) NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    total_rows INTEGER,
    processed_rows INTEGER DEFAULT 0,
    error_message TEXT,
    metadata JSONB
);

-- Analysis logs table (single column version)
CREATE TABLE IF NOT EXISTS analysis_logs_v1 (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES analysis_runs_v1(run_id),
    record_id VARCHAR(255) NOT NULL,
    result JSONB NOT NULL,
    latency_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for v1 tables
CREATE INDEX idx_analysis_runs_v1_file_id ON analysis_runs_v1(file_id);
CREATE INDEX idx_analysis_runs_v1_status ON analysis_runs_v1(status);
CREATE INDEX idx_analysis_logs_v1_run_id ON analysis_logs_v1(run_id);
CREATE INDEX idx_analysis_logs_v1_created_at ON analysis_logs_v1(created_at);