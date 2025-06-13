//! Data source abstractions for various input types

use crate::{DataCloakError, Result};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use tokio_postgres::NoTls;
use uuid;

/// Configuration for different data sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataSourceConfig {
    PostgreSQL {
        connection_string: String,
        query: String,
        fetch_size: Option<usize>,
    },
    CSV {
        path: PathBuf,
        delimiter: Option<u8>,
        has_headers: bool,
    },
    Parquet {
        path: PathBuf,
    },
    Memory {
        data: Vec<serde_json::Value>,
    },
}

/// Represents a data source that can be processed
#[derive(Debug, Clone)]
pub struct DataSource {
    config: DataSourceConfig,
}

/// A batch of records from a data source
pub type RecordBatch = Vec<serde_json::Value>;

/// Stream of record batches
pub type RecordStream = Pin<Box<dyn Stream<Item = Result<RecordBatch>> + Send>>;

impl DataSource {
    /// Create a new data source
    pub fn new(config: DataSourceConfig) -> Self {
        Self { config }
    }

    /// Create a PostgreSQL data source
    pub fn postgres(connection_string: String, query: String) -> Self {
        Self::new(DataSourceConfig::PostgreSQL {
            connection_string,
            query,
            fetch_size: Some(1000),
        })
    }

    /// Create a CSV data source
    pub fn csv(path: PathBuf) -> Self {
        Self::new(DataSourceConfig::CSV {
            path,
            delimiter: Some(b','),
            has_headers: true,
        })
    }

    /// Sample records from the data source for pattern detection
    pub async fn sample(&self, max_records: usize) -> Result<Vec<serde_json::Value>> {
        match &self.config {
            DataSourceConfig::PostgreSQL {
                connection_string,
                query,
                ..
            } => {
                self.sample_postgres(connection_string, query, max_records)
                    .await
            }
            DataSourceConfig::CSV {
                path,
                delimiter,
                has_headers,
            } => {
                self.sample_csv(path, *delimiter, *has_headers, max_records)
                    .await
            }
            DataSourceConfig::Memory { data } => {
                Ok(data.iter().take(max_records).cloned().collect())
            }
            _ => Err(DataCloakError::Other(
                "Unsupported data source for sampling".into(),
            )),
        }
    }

    /// Stream records in batches
    pub async fn stream(&self, batch_size: usize) -> Result<RecordStream> {
        match &self.config {
            DataSourceConfig::PostgreSQL {
                connection_string,
                query,
                ..
            } => {
                self.stream_postgres(connection_string, query, batch_size)
                    .await
            }
            DataSourceConfig::CSV {
                path,
                delimiter,
                has_headers,
            } => {
                self.stream_csv(path, *delimiter, *has_headers, batch_size)
                    .await
            }
            DataSourceConfig::Memory { data } => Ok(self.stream_memory(data.clone(), batch_size)),
            _ => Err(DataCloakError::Other(
                "Unsupported data source for streaming".into(),
            )),
        }
    }

    /// Sample from PostgreSQL
    async fn sample_postgres(
        &self,
        conn_str: &str,
        query: &str,
        max_records: usize,
    ) -> Result<Vec<serde_json::Value>> {
        let (client, connection) = tokio_postgres::connect(conn_str, NoTls).await?;

        // Spawn connection handler
        tokio::spawn(async move {
            if let Err(e) = connection.await {
                tracing::error!("PostgreSQL connection error: {}", e);
            }
        });

        // Add LIMIT to query if not present
        let limited_query = if query.to_lowercase().contains("limit") {
            query.to_string()
        } else {
            format!("{} LIMIT {}", query, max_records)
        };

        let rows = client.query(&limited_query, &[]).await?;
        let mut records = Vec::new();

        for row in rows {
            let mut record = serde_json::Map::new();
            for (i, column) in row.columns().iter().enumerate() {
                let value = self.postgres_value_to_json(&row, i)?;
                record.insert(column.name().to_string(), value);
            }
            records.push(serde_json::Value::Object(record));
        }

        Ok(records)
    }

    /// Sample from CSV
    async fn sample_csv(
        &self,
        path: &Path,
        delimiter: Option<u8>,
        has_headers: bool,
        max_records: usize,
    ) -> Result<Vec<serde_json::Value>> {
        use tokio::fs::File;
        use tokio::io::BufReader;

        let file = File::open(path).await?;
        let _reader = BufReader::new(file);

        // For now, use synchronous CSV reading wrapped in spawn_blocking
        let path = path.to_path_buf();
        let records = tokio::task::spawn_blocking(move || -> Result<Vec<serde_json::Value>> {
            use csv::ReaderBuilder;
            use std::fs::File;

            let file = File::open(&path)?;
            let mut reader = ReaderBuilder::new()
                .delimiter(delimiter.unwrap_or(b','))
                .has_headers(has_headers)
                .from_reader(file);

            let mut records = Vec::new();
            for (i, result) in reader.deserialize().enumerate() {
                if i >= max_records {
                    break;
                }
                let record: serde_json::Value = result?;
                records.push(record);
            }

            Ok(records)
        })
        .await
        .map_err(|e| DataCloakError::Other(e.to_string()))??;

        Ok(records)
    }

    /// Stream from PostgreSQL using cursor
    async fn stream_postgres(
        &self,
        conn_str: &str,
        query: &str,
        batch_size: usize,
    ) -> Result<RecordStream> {
        let (client, connection) = tokio_postgres::connect(conn_str, NoTls).await?;

        tokio::spawn(async move {
            if let Err(e) = connection.await {
                tracing::error!("PostgreSQL connection error: {}", e);
            }
        });

        // Create cursor-based query
        let cursor_name = format!("datacloak_cursor_{}", uuid::Uuid::new_v4());
        let declare_cursor = format!("DECLARE {} CURSOR FOR {}", cursor_name, query);

        client.execute(&declare_cursor, &[]).await?;

        let stream = futures::stream::unfold(
            (client, cursor_name, batch_size),
            |(client, cursor, batch_size)| async move {
                let fetch_query = format!("FETCH {} FROM {}", batch_size, cursor);
                match client.query(&fetch_query, &[]).await {
                    Ok(rows) if !rows.is_empty() => {
                        let mut batch = Vec::new();
                        for row in rows {
                            let mut record = serde_json::Map::new();
                            for (i, column) in row.columns().iter().enumerate() {
                                if let Ok(value) = postgres_value_to_json_static(&row, i) {
                                    record.insert(column.name().to_string(), value);
                                }
                            }
                            batch.push(serde_json::Value::Object(record));
                        }
                        Some((Ok(batch), (client, cursor, batch_size)))
                    }
                    Ok(_) => None, // Empty result, end of stream
                    Err(e) => Some((
                        Err(DataCloakError::Database(e.to_string())),
                        (client, cursor, batch_size),
                    )),
                }
            },
        );

        Ok(Box::pin(stream))
    }

    /// Stream from CSV using memory-mapped files
    async fn stream_csv(
        &self,
        path: &Path,
        delimiter: Option<u8>,
        has_headers: bool,
        batch_size: usize,
    ) -> Result<RecordStream> {
        use std::fs::File;

        let path = path.to_path_buf();
        let stream = futures::stream::unfold(
            (path, delimiter, has_headers, batch_size, 0usize),
            |(path, delimiter, has_headers, batch_size, offset)| async move {
                // This is a simplified version - in production, we'd use proper CSV parsing
                // with memory-mapped files for better performance
                tokio::task::spawn_blocking(move || -> Option<(Result<RecordBatch>, _)> {
                    use csv::ReaderBuilder;

                    match File::open(&path) {
                        Ok(file) => {
                            let mut reader = ReaderBuilder::new()
                                .delimiter(delimiter.unwrap_or(b','))
                                .has_headers(has_headers)
                                .from_reader(file);

                            let mut batch = Vec::new();
                            for (i, result) in reader.deserialize().skip(offset).enumerate() {
                                if i >= batch_size {
                                    break;
                                }
                                match result {
                                    Ok(record) => batch.push(record),
                                    Err(e) => {
                                        return Some((
                                            Err(DataCloakError::Csv(e)),
                                            (path, delimiter, has_headers, batch_size, offset),
                                        ))
                                    }
                                }
                            }

                            if batch.is_empty() {
                                None
                            } else {
                                let batch_len = batch.len();
                                Some((
                                    Ok(batch),
                                    (path, delimiter, has_headers, batch_size, offset + batch_len),
                                ))
                            }
                        }
                        Err(e) => Some((
                            Err(DataCloakError::Io(e)),
                            (path, delimiter, has_headers, batch_size, offset),
                        )),
                    }
                })
                .await
                .ok()?
            },
        );

        Ok(Box::pin(stream))
    }

    /// Stream from memory
    fn stream_memory(&self, data: Vec<serde_json::Value>, batch_size: usize) -> RecordStream {
        let chunks: Vec<RecordBatch> = data
            .chunks(batch_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        let stream = futures::stream::iter(chunks.into_iter().map(Ok));
        Box::pin(stream)
    }

    /// Convert PostgreSQL value to JSON
    fn postgres_value_to_json(
        &self,
        row: &tokio_postgres::Row,
        idx: usize,
    ) -> Result<serde_json::Value> {
        postgres_value_to_json_static(row, idx)
    }
}

// Helper function for PostgreSQL value conversion
fn postgres_value_to_json_static(
    row: &tokio_postgres::Row,
    idx: usize,
) -> Result<serde_json::Value> {
    use tokio_postgres::types::Type;

    let column_type = row.columns()[idx].type_();

    match *column_type {
        Type::INT2 => Ok(row.get::<_, i16>(idx).into()),
        Type::INT4 => Ok(row.get::<_, i32>(idx).into()),
        Type::INT8 => Ok(row.get::<_, i64>(idx).into()),
        Type::FLOAT4 => Ok(row.get::<_, f32>(idx).into()),
        Type::FLOAT8 => Ok(row.get::<_, f64>(idx).into()),
        Type::BOOL => Ok(row.get::<_, bool>(idx).into()),
        Type::TEXT | Type::VARCHAR => Ok(row.get::<_, String>(idx).into()),
        Type::JSON | Type::JSONB => Ok(row.get::<_, serde_json::Value>(idx)),
        _ => Ok(serde_json::Value::String(format!(
            "{:?}",
            row.get::<_, Option<String>>(idx)
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_source_creation() {
        let ds = DataSource::postgres(
            "postgresql://localhost/test".to_string(),
            "SELECT * FROM customers".to_string(),
        );

        match ds.config {
            DataSourceConfig::PostgreSQL { query, .. } => {
                assert_eq!(query, "SELECT * FROM customers");
            }
            _ => panic!("Wrong data source type"),
        }
    }
}
