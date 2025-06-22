use crate::models::{AnalysisRun, RunStatus};
use sqlx::{PgPool, postgres::PgRow, Row};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use anyhow::Result;

#[derive(Clone)]
pub struct AnalysisRunRepository {
    pool: PgPool,
}

impl AnalysisRunRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
    
    pub async fn create(&self, run: &AnalysisRun) -> Result<()> {
        let selected_columns_json = serde_json::to_value(&run.selected_columns)?;
        
        sqlx::query!(
            r#"
            INSERT INTO analysis_runs (
                run_id, file_id, selected_columns, chain_type, 
                started_at, status, total_rows, processed_rows
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            "#,
            run.run_id,
            run.file_id,
            selected_columns_json,
            run.chain_type.to_string(),
            run.started_at,
            run.status.to_string(),
            run.total_rows,
            run.processed_rows
        )
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }
    
    pub async fn get(&self, run_id: Uuid) -> Result<AnalysisRun> {
        let row = sqlx::query!(
            r#"
            SELECT run_id, file_id, selected_columns, chain_type,
                   started_at, completed_at, status, total_rows, processed_rows
            FROM analysis_runs
            WHERE run_id = $1
            "#,
            run_id
        )
        .fetch_one(&self.pool)
        .await?;
        
        let selected_columns: Vec<String> = serde_json::from_value(row.selected_columns)?;
        
        Ok(AnalysisRun {
            run_id: row.run_id,
            file_id: row.file_id,
            selected_columns,
            chain_type: row.chain_type.parse()?,
            started_at: row.started_at,
            completed_at: row.completed_at,
            status: row.status.parse()?,
            total_rows: row.total_rows,
            processed_rows: row.processed_rows,
        })
    }
    
    pub async fn update_progress(&self, run_id: &Uuid, processed_rows: i32) -> Result<()> {
        sqlx::query!(
            r#"
            UPDATE analysis_runs
            SET processed_rows = $2
            WHERE run_id = $1
            "#,
            run_id,
            processed_rows
        )
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }
    
    pub async fn complete(&self, run_id: &Uuid) -> Result<()> {
        sqlx::query!(
            r#"
            UPDATE analysis_runs
            SET status = $2, completed_at = $3
            WHERE run_id = $1
            "#,
            run_id,
            RunStatus::Completed.to_string(),
            Utc::now()
        )
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }
    
    pub async fn fail(&self, run_id: &Uuid, error_message: &str) -> Result<()> {
        sqlx::query!(
            r#"
            UPDATE analysis_runs
            SET status = $2, completed_at = $3, error_message = $4
            WHERE run_id = $1
            "#,
            run_id,
            RunStatus::Failed.to_string(),
            Utc::now(),
            error_message
        )
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }
    
    pub async fn list_by_file(&self, file_id: &Uuid) -> Result<Vec<AnalysisRun>> {
        let rows = sqlx::query!(
            r#"
            SELECT run_id, file_id, selected_columns, chain_type,
                   started_at, completed_at, status, total_rows, processed_rows
            FROM analysis_runs
            WHERE file_id = $1
            ORDER BY started_at DESC
            "#,
            file_id
        )
        .fetch_all(&self.pool)
        .await?;
        
        let mut runs = Vec::new();
        for row in rows {
            let selected_columns: Vec<String> = serde_json::from_value(row.selected_columns)?;
            runs.push(AnalysisRun {
                run_id: row.run_id,
                file_id: row.file_id,
                selected_columns,
                chain_type: row.chain_type.parse()?,
                started_at: row.started_at,
                completed_at: row.completed_at,
                status: row.status.parse()?,
                total_rows: row.total_rows,
                processed_rows: row.processed_rows,
            });
        }
        
        Ok(runs)
    }
}

impl std::str::FromStr for RunStatus {
    type Err = anyhow::Error;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "running" => Ok(RunStatus::Running),
            "completed" => Ok(RunStatus::Completed),
            "failed" => Ok(RunStatus::Failed),
            "cancelled" => Ok(RunStatus::Cancelled),
            _ => Err(anyhow::anyhow!("Invalid run status: {}", s)),
        }
    }
}

impl std::fmt::Display for RunStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunStatus::Running => write!(f, "running"),
            RunStatus::Completed => write!(f, "completed"),
            RunStatus::Failed => write!(f, "failed"),
            RunStatus::Cancelled => write!(f, "cancelled"),
        }
    }
}

impl std::str::FromStr for crate::models::ChainType {
    type Err = anyhow::Error;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "sentiment" => Ok(crate::models::ChainType::Sentiment),
            "entity" => Ok(crate::models::ChainType::Entity),
            "classification" => Ok(crate::models::ChainType::Classification),
            custom => Ok(crate::models::ChainType::Custom(custom.to_string())),
        }
    }
}

impl std::fmt::Display for crate::models::ChainType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            crate::models::ChainType::Sentiment => write!(f, "sentiment"),
            crate::models::ChainType::Entity => write!(f, "entity"),
            crate::models::ChainType::Classification => write!(f, "classification"),
            crate::models::ChainType::Custom(s) => write!(f, "{}", s),
        }
    }
}