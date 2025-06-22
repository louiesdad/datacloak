use crate::models::AnalysisLog;
use sqlx::PgPool;
use uuid::Uuid;
use anyhow::Result;

#[derive(Clone)]
pub struct AnalysisLogRepository {
    pool: PgPool,
}

impl AnalysisLogRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
    
    pub async fn create(&self, log: &AnalysisLog) -> Result<()> {
        sqlx::query!(
            r#"
            INSERT INTO analysis_logs (
                id, run_id, record_id, column_name, result, latency_ms, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            "#,
            log.id,
            log.run_id,
            log.record_id,
            log.column_name,
            log.result,
            log.latency_ms,
            log.created_at
        )
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }
    
    pub async fn create_batch(&self, logs: &[AnalysisLog]) -> Result<()> {
        if logs.is_empty() {
            return Ok(());
        }
        
        // Build bulk insert query
        let mut query_builder = sqlx::QueryBuilder::new(
            "INSERT INTO analysis_logs (id, run_id, record_id, column_name, result, latency_ms, created_at) "
        );
        
        query_builder.push_values(logs, |mut b, log| {
            b.push_bind(log.id)
                .push_bind(log.run_id)
                .push_bind(&log.record_id)
                .push_bind(&log.column_name)
                .push_bind(&log.result)
                .push_bind(log.latency_ms)
                .push_bind(log.created_at);
        });
        
        let query = query_builder.build();
        query.execute(&self.pool).await?;
        
        Ok(())
    }
    
    pub async fn get_by_column(&self, run_id: &Uuid, column_name: &str) -> Result<Vec<AnalysisLog>> {
        let rows = sqlx::query!(
            r#"
            SELECT id, run_id, record_id, column_name, result, latency_ms, created_at
            FROM analysis_logs
            WHERE run_id = $1 AND column_name = $2
            ORDER BY created_at
            "#,
            run_id,
            column_name
        )
        .fetch_all(&self.pool)
        .await?;
        
        let logs = rows.into_iter().map(|row| AnalysisLog {
            id: row.id,
            run_id: row.run_id,
            record_id: row.record_id,
            column_name: row.column_name,
            result: row.result,
            latency_ms: row.latency_ms.unwrap_or(0),
            created_at: row.created_at,
        }).collect();
        
        Ok(logs)
    }
    
    pub async fn get_by_run(&self, run_id: &Uuid, limit: Option<i64>, offset: Option<i64>) -> Result<Vec<AnalysisLog>> {
        let limit = limit.unwrap_or(1000);
        let offset = offset.unwrap_or(0);
        
        let rows = sqlx::query!(
            r#"
            SELECT id, run_id, record_id, column_name, result, latency_ms, created_at
            FROM analysis_logs
            WHERE run_id = $1
            ORDER BY created_at
            LIMIT $2 OFFSET $3
            "#,
            run_id,
            limit,
            offset
        )
        .fetch_all(&self.pool)
        .await?;
        
        let logs = rows.into_iter().map(|row| AnalysisLog {
            id: row.id,
            run_id: row.run_id,
            record_id: row.record_id,
            column_name: row.column_name,
            result: row.result,
            latency_ms: row.latency_ms.unwrap_or(0),
            created_at: row.created_at,
        }).collect();
        
        Ok(logs)
    }
    
    pub async fn count_by_run(&self, run_id: &Uuid) -> Result<i64> {
        let count = sqlx::query_scalar!(
            r#"
            SELECT COUNT(*) as "count!"
            FROM analysis_logs
            WHERE run_id = $1
            "#,
            run_id
        )
        .fetch_one(&self.pool)
        .await?;
        
        Ok(count)
    }
    
    pub async fn get_average_latency(&self, run_id: &Uuid) -> Result<Option<f64>> {
        let avg = sqlx::query_scalar!(
            r#"
            SELECT AVG(latency_ms)::FLOAT8 as "avg"
            FROM analysis_logs
            WHERE run_id = $1 AND latency_ms IS NOT NULL
            "#,
            run_id
        )
        .fetch_one(&self.pool)
        .await?;
        
        Ok(avg)
    }
}