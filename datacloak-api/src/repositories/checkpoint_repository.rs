use sqlx::PgPool;
use uuid::Uuid;
use anyhow::Result;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub worker_id: i32,
    pub run_id: Uuid,
    pub last_offset: i64,
    pub last_record_id: String,
}

#[derive(Clone)]
pub struct CheckpointRepository {
    pool: PgPool,
}

impl CheckpointRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
    
    pub async fn save(&self, checkpoint: &Checkpoint) -> Result<()> {
        sqlx::query!(
            r#"
            INSERT INTO analysis_checkpoints (worker_id, run_id, last_offset, last_record_id, updated_at)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (worker_id, run_id) 
            DO UPDATE SET 
                last_offset = EXCLUDED.last_offset,
                last_record_id = EXCLUDED.last_record_id,
                updated_at = EXCLUDED.updated_at
            "#,
            checkpoint.worker_id,
            checkpoint.run_id,
            checkpoint.last_offset,
            checkpoint.last_record_id,
            Utc::now()
        )
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }
    
    pub async fn load(&self, worker_id: i32, run_id: &Uuid) -> Result<Checkpoint> {
        let row = sqlx::query!(
            r#"
            SELECT worker_id, run_id, last_offset, last_record_id
            FROM analysis_checkpoints
            WHERE worker_id = $1 AND run_id = $2
            "#,
            worker_id,
            run_id
        )
        .fetch_one(&self.pool)
        .await?;
        
        Ok(Checkpoint {
            worker_id: row.worker_id,
            run_id: row.run_id,
            last_offset: row.last_offset,
            last_record_id: row.last_record_id.unwrap_or_default(),
        })
    }
    
    pub async fn load_all_for_run(&self, run_id: &Uuid) -> Result<Vec<Checkpoint>> {
        let rows = sqlx::query!(
            r#"
            SELECT worker_id, run_id, last_offset, last_record_id
            FROM analysis_checkpoints
            WHERE run_id = $1
            ORDER BY worker_id
            "#,
            run_id
        )
        .fetch_all(&self.pool)
        .await?;
        
        let checkpoints = rows.into_iter().map(|row| Checkpoint {
            worker_id: row.worker_id,
            run_id: row.run_id,
            last_offset: row.last_offset,
            last_record_id: row.last_record_id.unwrap_or_default(),
        }).collect();
        
        Ok(checkpoints)
    }
    
    pub async fn delete_for_run(&self, run_id: &Uuid) -> Result<()> {
        sqlx::query!(
            r#"
            DELETE FROM analysis_checkpoints
            WHERE run_id = $1
            "#,
            run_id
        )
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }
    
    pub async fn get_latest_checkpoint_time(&self, run_id: &Uuid) -> Result<Option<DateTime<Utc>>> {
        let time = sqlx::query_scalar!(
            r#"
            SELECT MAX(updated_at) as "latest"
            FROM analysis_checkpoints
            WHERE run_id = $1
            "#,
            run_id
        )
        .fetch_one(&self.pool)
        .await?;
        
        Ok(time)
    }
}