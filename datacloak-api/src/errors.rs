use actix_web::{error::ResponseError, http::StatusCode, HttpResponse};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ApiError {
    #[error("File not found: {0}")]
    FileNotFound(String),
    
    #[error("Invalid request: {0}")]
    BadRequest(String),
    
    #[error("Internal server error: {0}")]
    InternalError(String),
    
    #[error("Database error: {0}")]
    DatabaseError(#[from] sqlx::Error),
    
    #[error("Validation error: {0}")]
    ValidationError(String),
}

impl ResponseError for ApiError {
    fn error_response(&self) -> HttpResponse {
        let status = self.status_code();
        let error_response = serde_json::json!({
            "error": self.to_string(),
            "status": status.as_u16(),
        });
        
        HttpResponse::build(status).json(error_response)
    }
    
    fn status_code(&self) -> StatusCode {
        match self {
            ApiError::FileNotFound(_) => StatusCode::NOT_FOUND,
            ApiError::BadRequest(_) => StatusCode::BAD_REQUEST,
            ApiError::ValidationError(_) => StatusCode::BAD_REQUEST,
            ApiError::InternalError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ApiError::DatabaseError(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}