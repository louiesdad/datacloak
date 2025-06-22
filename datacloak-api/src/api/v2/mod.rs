use actix_web::{web, App, HttpServer, middleware::Logger};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;
use crate::middleware::{MetricsMiddleware, RequestIdMiddleware};

pub mod profile;
pub mod analyze;
pub mod estimate;
pub mod monitoring;
pub mod openapi;

use self::openapi::ApiDoc;

pub fn create_app() -> App<impl actix_web::dev::ServiceFactory<
    actix_web::dev::ServiceRequest,
    Config = (),
    Response = actix_web::dev::ServiceResponse,
    Error = actix_web::Error,
    InitError = (),
>> {
    let openapi = ApiDoc::openapi();
    let stream_manager = std::sync::Arc::new(crate::streaming::StreamManager::new());
    let profile_service = std::sync::Arc::new(crate::services::ProfileService::new());
    
    // Create ETA service
    let eta_estimator = crate::services::ETAEstimator::new();
    let eta_service = std::sync::Arc::new(crate::services::ETAService::new(eta_estimator));
    
    // Create worker coordinator
    let worker_coordinator = std::sync::Arc::new(crate::services::WorkerCoordinator::new(4));
    
    // Create monitoring service
    let monitoring_service = std::sync::Arc::new(
        crate::services::monitoring::MonitoringService::new("0.1.0".to_string()).unwrap()
    );
    
    App::new()
        .wrap(Logger::default())
        .wrap(RequestIdMiddleware)
        .wrap(MetricsMiddleware)
        .app_data(web::Data::new(stream_manager))
        .app_data(web::Data::new(profile_service))
        .app_data(web::Data::new(eta_service))
        .app_data(web::Data::new(worker_coordinator))
        .app_data(web::Data::new(monitoring_service))
        .service(
            web::scope("/api/v2")
                .service(profile::profile_endpoint)
                .service(analyze::analyze_endpoint)
                .service(estimate::estimate_endpoint)
        )
        .service(
            web::scope("/monitoring")
                .route("/metrics", web::get().to(monitoring::metrics))
                .route("/health", web::get().to(monitoring::health))
                .route("/ready", web::get().to(monitoring::ready))
                .route("/live", web::get().to(monitoring::live))
                .route("/slo", web::get().to(monitoring::slo_status))
                .route("/traces", web::get().to(monitoring::traces))
        )
        .service(
            SwaggerUi::new("/swagger-ui/{_:.*}")
                .url("/api-docs/openapi.json", openapi.clone())
        )
}

pub async fn start_server(bind_address: &str) -> std::io::Result<()> {
    HttpServer::new(|| create_app())
        .bind(bind_address)?
        .run()
        .await
}