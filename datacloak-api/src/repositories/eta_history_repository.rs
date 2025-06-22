use crate::models::ChainType;
use crate::services::{HistoricalEstimate};
use sqlx::PgPool;
use uuid::Uuid;
use anyhow::Result;

#[derive(Clone)]
pub struct ETAHistoryRepository {
    pool: PgPool,
}

impl ETAHistoryRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
    
    pub async fn record_estimate(
        &self,
        file_id: Uuid,
        row_count: usize,
        column_count: usize,
        chain_type: &ChainType,
        estimated_seconds: u64,
    ) -> Result<Uuid> {
        let record_id = Uuid::new_v4();
        
        sqlx::query!(
            r#"
            INSERT INTO eta_history (
                id, file_id, row_count, column_count, chain_type, estimated_seconds, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, NOW())
            "#,
            record_id,
            file_id,
            row_count as i32,
            column_count as i32,
            chain_type.to_string(),
            estimated_seconds as i32
        )
        .execute(&self.pool)
        .await?;
        
        Ok(record_id)
    }
    
    pub async fn record_actual_result(
        &self,
        file_id: Uuid,
        estimated_seconds: u64,
        actual_seconds: u64,
    ) -> Result<()> {
        sqlx::query!(
            r#"
            UPDATE eta_history 
            SET actual_seconds = $3
            WHERE file_id = $1 AND estimated_seconds = $2
            "#,
            file_id,
            estimated_seconds as i32,
            actual_seconds as i32
        )
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }
    
    pub async fn get_similar_estimates(
        &self,
        target_rows: usize,
        target_columns: usize,
        chain_type: &ChainType,
        limit: usize,
    ) -> Result<Vec<HistoricalEstimate>> {
        let rows = sqlx::query!(
            r#"
            SELECT row_count, column_count, estimated_seconds, actual_seconds, chain_type,
                   ABS(row_count - $1) + ABS(column_count - $2) as similarity_score
            FROM eta_history
            WHERE chain_type = $3 AND created_at > NOW() - INTERVAL '30 days'
            ORDER BY similarity_score ASC
            LIMIT $4
            "#,
            target_rows as i32,
            target_columns as i32,
            chain_type.to_string(),
            limit as i64
        )
        .fetch_all(&self.pool)
        .await?;
        
        let estimates = rows.into_iter().map(|row| {
            HistoricalEstimate {
                row_count: row.row_count as usize,
                column_count: row.column_count as usize,
                estimated_seconds: row.estimated_seconds as u64,
                actual_seconds: row.actual_seconds.map(|s| s as u64),
                chain_type: row.chain_type.parse().unwrap_or(ChainType::Sentiment),
            }
        }).collect();
        
        Ok(estimates)
    }
    
    pub async fn get_accuracy_stats(&self, days: i32) -> Result<AccuracyStats> {
        let row = sqlx::query!(
            r#"
            SELECT 
                COUNT(*) as total_estimates,
                AVG(ABS(CAST(actual_seconds AS FLOAT) - CAST(estimated_seconds AS FLOAT)) / CAST(estimated_seconds AS FLOAT)) as avg_error_rate,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ABS(CAST(actual_seconds AS FLOAT) - CAST(estimated_seconds AS FLOAT)) / CAST(estimated_seconds AS FLOAT)) as median_error_rate
            FROM eta_history
            WHERE actual_seconds IS NOT NULL 
            AND created_at > NOW() - INTERVAL '%d days'
            "#,
            days
        )
        .fetch_one(&self.pool)
        .await?;
        
        Ok(AccuracyStats {
            total_estimates: row.total_estimates.unwrap_or(0) as usize,
            avg_error_rate: row.avg_error_rate.unwrap_or(0.0),
            median_error_rate: row.median_error_rate.unwrap_or(0.0),
        })
    }
    
    pub async fn cleanup_old_records(&self, days: i32) -> Result<u64> {
        let result = sqlx::query!(
            r#"
            DELETE FROM eta_history
            WHERE created_at < NOW() - INTERVAL '%d days'
            "#,
            days
        )
        .execute(&self.pool)
        .await?;
        
        Ok(result.rows_affected())
    }
}

#[derive(Debug)]
pub struct AccuracyStats {
    pub total_estimates: usize,
    pub avg_error_rate: f64,
    pub median_error_rate: f64,
}

// Removed duplicate FromStr implementation - already exists in analysis_run_repository.rs