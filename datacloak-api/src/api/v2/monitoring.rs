use actix_web::{web, HttpResponse, Result as ActixResult};
use crate::services::monitoring::{MonitoringService, HealthCheckResponse};
use std::sync::Arc;

// Prometheus metrics endpoint
pub async fn metrics(
    monitoring: web::Data<Arc<MonitoringService>>,
) -> ActixResult<HttpResponse> {
    match monitoring.metrics.render_metrics() {
        Ok(metrics_text) => Ok(HttpResponse::Ok()
            .content_type("text/plain; version=0.0.4; charset=utf-8")
            .body(metrics_text)),
        Err(e) => Ok(HttpResponse::InternalServerError()
            .json(serde_json::json!({
                "error": "Failed to render metrics",
                "details": e.to_string()
            }))),
    }
}

// Health check endpoint
pub async fn health(
    monitoring: web::Data<Arc<MonitoringService>>,
) -> ActixResult<HttpResponse> {
    let health_status = monitoring.health.check_health().await;
    
    let status_code = match health_status.status.as_str() {
        "healthy" => 200,
        "degraded" => 503,
        _ => 500,
    };
    
    Ok(HttpResponse::build(actix_web::http::StatusCode::from_u16(status_code).unwrap())
        .json(health_status))
}

// Readiness probe (for Kubernetes)
pub async fn ready(
    monitoring: web::Data<Arc<MonitoringService>>,
) -> ActixResult<HttpResponse> {
    let health_status = monitoring.health.check_health().await;
    
    // Service is ready if all critical components are healthy
    let is_ready = health_status.checks.iter().all(|(component, health)| {
        // Only database and cache are critical for readiness
        if component == "database" || component == "cache" {
            health.status == "healthy"
        } else {
            true // Non-critical components don't affect readiness
        }
    });
    
    if is_ready {
        Ok(HttpResponse::Ok().json(serde_json::json!({
            "status": "ready",
            "message": "Service is ready to accept requests"
        })))
    } else {
        Ok(HttpResponse::ServiceUnavailable().json(serde_json::json!({
            "status": "not_ready",
            "message": "Service is not ready",
            "details": health_status.checks
        })))
    }
}

// Liveness probe (for Kubernetes)
pub async fn live() -> ActixResult<HttpResponse> {
    // Simple liveness check - if we can respond, we're alive
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "status": "alive",
        "timestamp": chrono::Utc::now().to_rfc3339()
    })))
}

// SLO dashboard endpoint
pub async fn slo_status(
    monitoring: web::Data<Arc<MonitoringService>>,
) -> ActixResult<HttpResponse> {
    let slo_status = monitoring.slo_dashboard.check_slo_compliance(&monitoring.metrics);
    
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "slo_compliance": slo_status,
        "thresholds": {
            "error_rate": monitoring.slo_dashboard.error_rate_threshold,
            "latency_p95_ms": monitoring.slo_dashboard.latency_p95_threshold_ms,
            "availability": monitoring.slo_dashboard.availability_threshold
        },
        "timestamp": chrono::Utc::now().to_rfc3339()
    })))
}

// Active traces endpoint (for debugging)
pub async fn traces(
    monitoring: web::Data<Arc<MonitoringService>>,
) -> ActixResult<HttpResponse> {
    let active_count = monitoring.tracing.get_active_trace_count().await;
    
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "active_traces": active_count,
        "timestamp": chrono::Utc::now().to_rfc3339()
    })))
}