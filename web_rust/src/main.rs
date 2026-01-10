mod models;
mod handlers;
mod python_bridge;

use axum::{
    routing::{get, post},
    Router,
    extract::DefaultBodyLimit,
};
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    tracing::info!("Starting Slim-Gest Rust Web Server...");
    
    // Initialize Python interpreter and load models
    tracing::info!("Initializing Python interpreter and loading models...");
    let models = Arc::new(python_bridge::initialize_models()?);
    tracing::info!("Models loaded successfully");
    
    // #region agent log
    // Set body limit to 500MB (Hypothesis B, C)
    let body_limit = 500 * 1024 * 1024; // 500MB
    tracing::info!("Setting body limit to {} bytes ({} MB)", body_limit, body_limit / (1024 * 1024));
    // #endregion
    
    // Build application routes
    let app = Router::new()
        .route("/", get(handlers::root))
        .route("/process-pdf", post(handlers::process_pdf))
        .route("/process-pdfs", post(handlers::process_pdfs))
        .route("/process-pdf-stream", post(handlers::process_pdf_stream))
        // #region agent log
        .layer(DefaultBodyLimit::max(body_limit)) // Hypothesis B, C
        // #endregion
        .layer(CorsLayer::permissive())
        .with_state(models);
    
    // Bind to address
    let addr = "0.0.0.0:7671";
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("Server listening on {}", addr);
    
    // Run the server
    axum::serve(listener, app).await?;
    
    Ok(())
}