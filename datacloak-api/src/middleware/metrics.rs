use actix_web::{
    dev::{forward_ready, Service, ServiceRequest, ServiceResponse, Transform},
    web, Error, HttpMessage,
};
use futures_util::future::LocalBoxFuture;
use std::{
    future::{ready, Ready},
    sync::Arc,
    time::Instant,
};
use crate::services::monitoring::MonitoringService;
use uuid::Uuid;

// Middleware factory for metrics collection
pub struct MetricsMiddleware;

impl<S, B> Transform<S, ServiceRequest> for MetricsMiddleware
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type InitError = ();
    type Transform = MetricsMiddlewareService<S>;
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ready(Ok(MetricsMiddlewareService { service }))
    }
}

pub struct MetricsMiddlewareService<S> {
    service: S,
}

impl<S, B> Service<ServiceRequest> for MetricsMiddlewareService<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Future = LocalBoxFuture<'static, Result<Self::Response, Self::Error>>;

    forward_ready!(service);

    fn call(&self, req: ServiceRequest) -> Self::Future {
        let start_time = Instant::now();
        let method = req.method().to_string();
        let path = req.path().to_string();
        
        // Start tracing span if monitoring service is available
        let trace_id = if let Some(monitoring) = req.app_data::<web::Data<Arc<MonitoringService>>>() {
            let mut metadata = std::collections::HashMap::new();
            metadata.insert("method".to_string(), method.clone());
            metadata.insert("path".to_string(), path.clone());
            
            // Extract request ID from headers if present
            if let Some(request_id) = req.headers().get("x-request-id") {
                if let Ok(id_str) = request_id.to_str() {
                    metadata.insert("request_id".to_string(), id_str.to_string());
                }
            }
            
            let monitoring_clone = monitoring.clone();
            let operation = format!("{} {}", method, path);
            Some((monitoring_clone, tokio::spawn(async move {
                monitoring_clone.tracing.start_trace(&operation, metadata).await
            })))
        } else {
            None
        };
        
        let fut = self.service.call(req);
        
        Box::pin(async move {
            let result = fut.await;
            let duration = start_time.elapsed();
            let duration_seconds = duration.as_secs_f64();
            
            match &result {
                Ok(response) => {
                    let status_code = response.status().as_u16();
                    
                    // Record metrics if monitoring service is available
                    if let Some(monitoring) = response.request().app_data::<web::Data<Arc<MonitoringService>>>() {
                        monitoring.metrics.record_api_request(
                            &path,
                            &method,
                            duration_seconds,
                            status_code
                        );
                        
                        // Finish trace
                        if let Some((monitoring_service, trace_future)) = trace_id {
                            if let Ok(trace_id) = trace_future.await {
                                monitoring_service.tracing.finish_trace(
                                    trace_id,
                                    status_code < 400,
                                    if status_code >= 400 {
                                        Some(format!("HTTP {}", status_code))
                                    } else {
                                        None
                                    }
                                ).await;
                            }
                        }
                    }
                },
                Err(error) => {
                    // Record error metrics
                    if let Some(monitoring) = response.as_ref().ok()
                        .and_then(|r| r.request().app_data::<web::Data<Arc<MonitoringService>>>()) {
                        monitoring.metrics.record_api_request(
                            &path,
                            &method,
                            duration_seconds,
                            500 // Internal server error
                        );
                        
                        // Finish trace with error
                        if let Some((monitoring_service, trace_future)) = trace_id {
                            if let Ok(trace_id) = trace_future.await {
                                monitoring_service.tracing.finish_trace(
                                    trace_id,
                                    false,
                                    Some(error.to_string())
                                ).await;
                            }
                        }
                    }
                }
            }
            
            result
        })
    }
}

// Request ID middleware for distributed tracing
pub struct RequestIdMiddleware;

impl<S, B> Transform<S, ServiceRequest> for RequestIdMiddleware
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type InitError = ();
    type Transform = RequestIdMiddlewareService<S>;
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ready(Ok(RequestIdMiddlewareService { service }))
    }
}

pub struct RequestIdMiddlewareService<S> {
    service: S,
}

impl<S, B> Service<ServiceRequest> for RequestIdMiddlewareService<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error>,
    S::Future: 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Future = LocalBoxFuture<'static, Result<Self::Response, Self::Error>>;

    forward_ready!(service);

    fn call(&self, mut req: ServiceRequest) -> Self::Future {
        // Generate or extract request ID
        let request_id = req.headers().get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        
        // Store request ID in request extensions for use by handlers
        req.extensions_mut().insert(request_id.clone());
        
        let fut = self.service.call(req);
        
        Box::pin(async move {
            let mut result = fut.await?;
            
            // Add request ID to response headers
            result.headers_mut().insert(
                actix_web::http::header::HeaderName::from_static("x-request-id"),
                actix_web::http::HeaderValue::from_str(&request_id).unwrap()
            );
            
            Ok(result)
        })
    }
}