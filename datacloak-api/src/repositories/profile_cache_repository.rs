use sqlx::PgPool;
use uuid::Uuid;
use chrono::{Utc, Duration};
use serde_json::Value;
use anyhow::Result;

#[derive(Clone)]
pub struct ProfileCacheRepository {
    pool: PgPool,
}

impl ProfileCacheRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
    
    pub async fn cache(&self, file_id: &Uuid, profile_result: &Value, ttl_seconds: i64) -> Result<()> {
        let expires_at = Utc::now() + Duration::seconds(ttl_seconds);
        
        sqlx::query!(
            r#"
            INSERT INTO profile_cache (file_id, profile_result, created_at, expires_at)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (file_id) 
            DO UPDATE SET 
                profile_result = EXCLUDED.profile_result,
                created_at = EXCLUDED.created_at,
                expires_at = EXCLUDED.expires_at
            "#,
            file_id,
            profile_result,
            Utc::now(),
            expires_at
        )
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }
    
    pub async fn get(&self, file_id: &Uuid) -> Result<Option<Value>> {
        let row = sqlx::query!(
            r#"
            SELECT profile_result
            FROM profile_cache
            WHERE file_id = $1 AND expires_at > $2
            "#,
            file_id,
            Utc::now()
        )
        .fetch_optional(&self.pool)
        .await?;
        
        Ok(row.map(|r| r.profile_result))
    }
    
    pub async fn invalidate(&self, file_id: &Uuid) -> Result<()> {
        sqlx::query!(
            r#"
            DELETE FROM profile_cache
            WHERE file_id = $1
            "#,
            file_id
        )
        .execute(&self.pool)
        .await?;
        
        Ok(())
    }
    
    pub async fn cleanup_expired(&self) -> Result<u64> {
        let result = sqlx::query!(
            r#"
            DELETE FROM profile_cache
            WHERE expires_at <= $1
            "#,
            Utc::now()
        )
        .execute(&self.pool)
        .await?;
        
        Ok(result.rows_affected())
    }
    
    pub async fn invalidate_if_older_than(&self, file_id: &Uuid, timestamp: chrono::DateTime<Utc>) -> Result<bool> {
        let result = sqlx::query!(
            r#"
            DELETE FROM profile_cache
            WHERE file_id = $1 AND created_at < $2
            "#,
            file_id,
            timestamp
        )
        .execute(&self.pool)
        .await?;
        
        Ok(result.rows_affected() > 0)
    }
}