use utoipa::OpenApi;
use crate::models::*;

#[derive(OpenApi)]
#[openapi(
    paths(
        crate::api::v2::profile::profile_endpoint,
        crate::api::v2::analyze::analyze_endpoint,
        crate::api::v2::estimate::estimate_endpoint
    ),
    components(
        schemas(
            ProfileRequest,
            ProfileResponse,
            ColumnCandidate,
            ColumnFeatures,
            AnalyzeRequest,
            AnalysisOptions,
            ChainType,
            AnalyzeResponse,
            RunStatus,
            AnalysisResult,
            EstimateRequest,
            ETAResponse
        )
    ),
    tags(
        (name = "profile", description = "Column profiling operations"),
        (name = "analyze", description = "Multi-field analysis operations"),
        (name = "estimate", description = "Runtime estimation operations")
    ),
    info(
        title = "DataCloak API",
        version = "2.0.0",
        description = "RESTful API for DataCloak multi-field sentiment analysis",
        contact(
            name = "DataCloak Team",
            email = "support@datacloak.io"
        ),
        license(
            name = "MIT",
            url = "https://opensource.org/licenses/MIT"
        )
    )
)]
pub struct ApiDoc;