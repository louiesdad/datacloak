-- Migration to support multi-column analysis

-- First, alter existing analysis_runs table if it exists
ALTER TABLE IF EXISTS analysis_runs 
    DROP COLUMN IF EXISTS selected_column,
    ADD COLUMN IF NOT EXISTS selected_columns JSON NOT NULL DEFAULT '[]';

-- If analysis_runs doesn't exist, create it with multi-column support
CREATE TABLE IF NOT EXISTS analysis_runs (
    run_id UUID PRIMARY KEY,
    file_id UUID NOT NULL REFERENCES files(file_id),
    selected_columns JSON NOT NULL DEFAULT '[]',
    chain_type VARCHAR(50) NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    total_rows INTEGER,
    processed_rows INTEGER DEFAULT 0,
    error_message TEXT,
    metadata JSONB,
    -- New fields for multi-column support
    worker_count INTEGER DEFAULT 1,
    estimated_completion TIMESTAMP WITH TIME ZONE,
    cost_estimate DECIMAL(10, 4)
);

-- Alter existing analysis_logs table to add column tracking
ALTER TABLE IF EXISTS analysis_logs
    ADD COLUMN IF NOT EXISTS column_name VARCHAR(100) NOT NULL DEFAULT 'default',
    ADD COLUMN IF NOT EXISTS worker_id INTEGER,
    ADD COLUMN IF NOT EXISTS sequence_number BIGINT;

-- Create analysis_logs if it doesn't exist
CREATE TABLE IF NOT EXISTS analysis_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES analysis_runs(run_id),
    record_id VARCHAR(255) NOT NULL,
    column_name VARCHAR(100) NOT NULL DEFAULT 'default',
    result JSONB NOT NULL,
    latency_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    worker_id INTEGER,
    sequence_number BIGINT
);

-- Checkpoint table for recovery
CREATE TABLE IF NOT EXISTS analysis_checkpoints (
    worker_id INTEGER NOT NULL,
    run_id UUID NOT NULL,
    last_offset BIGINT NOT NULL,
    last_record_id VARCHAR(255),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (worker_id, run_id)
);

-- Profiling results cache
CREATE TABLE IF NOT EXISTS profile_cache (
    file_id UUID NOT NULL REFERENCES files(file_id),
    profile_result JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    PRIMARY KEY (file_id)
);

-- ETA estimation history
CREATE TABLE IF NOT EXISTS eta_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_id UUID NOT NULL REFERENCES files(file_id),
    column_count INTEGER NOT NULL,
    row_count INTEGER NOT NULL,
    chain_type VARCHAR(50) NOT NULL,
    estimated_seconds INTEGER NOT NULL,
    actual_seconds INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for new tables
CREATE INDEX idx_analysis_runs_file_id ON analysis_runs(file_id);
CREATE INDEX idx_analysis_runs_status ON analysis_runs(status);
CREATE INDEX idx_analysis_runs_started_at ON analysis_runs(started_at);

CREATE INDEX idx_analysis_logs_run_id ON analysis_logs(run_id);
CREATE INDEX idx_analysis_logs_column_name ON analysis_logs(run_id, column_name);
CREATE INDEX idx_analysis_logs_created_at ON analysis_logs(created_at);
CREATE INDEX idx_analysis_logs_sequence ON analysis_logs(run_id, sequence_number);

CREATE INDEX idx_checkpoints_run_id ON analysis_checkpoints(run_id);
CREATE INDEX idx_checkpoints_updated ON analysis_checkpoints(updated_at);

CREATE INDEX idx_profile_cache_expires ON profile_cache(expires_at);

CREATE INDEX idx_eta_history_params ON eta_history(file_id, column_count, chain_type);

-- Migrate existing data from v1 to v2 tables
INSERT INTO analysis_runs (run_id, file_id, selected_columns, chain_type, started_at, completed_at, status, total_rows, processed_rows, error_message, metadata)
SELECT 
    run_id, 
    file_id, 
    json_build_array(selected_column),
    chain_type,
    started_at,
    completed_at,
    status,
    total_rows,
    processed_rows,
    error_message,
    metadata
FROM analysis_runs_v1;

INSERT INTO analysis_logs (id, run_id, record_id, column_name, result, latency_ms, created_at)
SELECT 
    id,
    run_id,
    record_id,
    (SELECT selected_column FROM analysis_runs_v1 WHERE analysis_runs_v1.run_id = analysis_logs_v1.run_id),
    result,
    latency_ms,
    created_at
FROM analysis_logs_v1;