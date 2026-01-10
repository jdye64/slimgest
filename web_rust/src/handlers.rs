use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::{IntoResponse, Response, Sse},
    Json,
};
use bytes::Bytes;
use futures::stream::{self, Stream, StreamExt};
use std::sync::Arc;
use tokio::fs;
use tempfile::TempDir;

use crate::models::*;
use crate::python_bridge::{Models, process_pdf_files, PdfPageProcessor};

/// Health check endpoint
pub async fn root() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        message: "Slim-Gest PDF Processing API (Rust) is running".to_string(),
        models_loaded: true,
    })
}

/// Process a single PDF file
pub async fn process_pdf(
    State(models): State<Arc<Models>>,
    mut multipart: Multipart,
) -> Result<Json<serde_json::Value>, AppError> {
    let mut dpi = 150.0;
    let mut pdf_file_data: Option<(String, Vec<u8>)> = None;
    
    // #region agent log
    let mut total_bytes_received = 0usize;
    // #endregion
    
    // Parse multipart form data
    while let Some(field) = multipart.next_field().await? {
        let name = field.name().unwrap_or("").to_string();
        
        match name.as_str() {
            "file" => {
                let filename = field.file_name().unwrap_or("unknown.pdf").to_string();
                if !filename.to_lowercase().ends_with(".pdf") {
                    return Err(AppError::BadRequest(format!(
                        "Invalid file type: {}. Only PDF files are accepted.",
                        filename
                    )));
                }
                let data: Bytes = field.bytes().await?;
                // #region agent log
                total_bytes_received += data.len();
                tracing::info!("Received file '{}': {} bytes (Hypothesis B)", filename, data.len());
                // #endregion
                pdf_file_data = Some((filename, data.to_vec()));
            }
            "dpi" => {
                let dpi_str: String = field.text().await?;
                dpi = dpi_str.parse().unwrap_or(150.0);
            }
            _ => {}
        }
    }
    
    // #region agent log
    tracing::info!("Total multipart data received: {} bytes ({} MB) (Hypothesis B)", 
        total_bytes_received, total_bytes_received / (1024 * 1024));
    // #endregion
    
    let (filename, file_data) = pdf_file_data
        .ok_or_else(|| AppError::BadRequest("No PDF file provided".to_string()))?;
    
    // Create temporary directory
    let temp_dir = TempDir::new()?;
    let file_path = temp_dir.path().join(&filename);
    
    // Write file to temp directory
    fs::write(&file_path, file_data).await?;
    
    // Process the PDF
    let pdf_path = file_path.to_string_lossy().to_string();
    let result = tokio::task::spawn_blocking({
        let models = Arc::clone(&models);
        move || process_pdf_files(&models, vec![pdf_path], dpi)
    })
    .await??;
    
    // #region agent log
    let result_json_str = serde_json::to_string(&result).unwrap_or_default();
    tracing::info!("Response size: {} bytes ({} MB) (Hypothesis D)", 
        result_json_str.len(), result_json_str.len() / (1024 * 1024));
    // #endregion
    
    // Cleanup happens automatically when temp_dir is dropped
    
    Ok(Json(result))
}

/// Process multiple PDF files
pub async fn process_pdfs(
    State(models): State<Arc<Models>>,
    mut multipart: Multipart,
) -> Result<Json<serde_json::Value>, AppError> {
    let mut dpi = 150.0;
    let mut pdf_files: Vec<(String, Vec<u8>)> = Vec::new();
    
    // Parse multipart form data
    while let Some(field) = multipart.next_field().await? {
        let name = field.name().unwrap_or("").to_string();
        
        match name.as_str() {
            "files" => {
                let filename = field.file_name().unwrap_or("unknown.pdf").to_string();
                if !filename.to_lowercase().ends_with(".pdf") {
                    return Err(AppError::BadRequest(format!(
                        "Invalid file type: {}. Only PDF files are accepted.",
                        filename
                    )));
                }
                let data: Bytes = field.bytes().await?;
                pdf_files.push((filename, data.to_vec()));
            }
            "dpi" => {
                let dpi_str: String = field.text().await?;
                dpi = dpi_str.parse().unwrap_or(150.0);
            }
            _ => {}
        }
    }
    
    if pdf_files.is_empty() {
        return Err(AppError::BadRequest("No PDF files provided".to_string()));
    }
    
    // Create temporary directory
    let temp_dir = TempDir::new()?;
    let mut pdf_paths = Vec::new();
    
    // Write all files to temp directory
    for (filename, file_data) in pdf_files {
        let file_path = temp_dir.path().join(&filename);
        fs::write(&file_path, file_data).await?;
        pdf_paths.push(file_path.to_string_lossy().to_string());
    }
    
    // Process the PDFs
    let result = tokio::task::spawn_blocking({
        let models = Arc::clone(&models);
        move || process_pdf_files(&models, pdf_paths, dpi)
    })
    .await??;
    
    // Cleanup happens automatically when temp_dir is dropped
    
    Ok(Json(result))
}

/// Process a PDF file with streaming response
pub async fn process_pdf_stream(
    State(models): State<Arc<Models>>,
    mut multipart: Multipart,
) -> Result<Sse<impl Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>>, AppError> {
    let mut dpi = 150.0;
    let mut pdf_file_data: Option<(String, Vec<u8>)> = None;
    
    // Parse multipart form data
    while let Some(field) = multipart.next_field().await? {
        let name = field.name().unwrap_or("").to_string();
        
        match name.as_str() {
            "file" => {
                let filename = field.file_name().unwrap_or("unknown.pdf").to_string();
                if !filename.to_lowercase().ends_with(".pdf") {
                    return Err(AppError::BadRequest(format!(
                        "Invalid file type: {}. Only PDF files are accepted.",
                        filename
                    )));
                }
                let data: Bytes = field.bytes().await?;
                pdf_file_data = Some((filename, data.to_vec()));
            }
            "dpi" => {
                let dpi_str: String = field.text().await?;
                dpi = dpi_str.parse().unwrap_or(150.0);
            }
            _ => {}
        }
    }
    
    let (filename, file_data) = pdf_file_data
        .ok_or_else(|| AppError::BadRequest("No PDF file provided".to_string()))?;
    
    // Create temporary directory
    let temp_dir = TempDir::new()?;
    let file_path = temp_dir.path().join(&filename);
    
    // Write file to temp directory
    fs::write(&file_path, &file_data).await?;
    
    let pdf_path = file_path.to_string_lossy().to_string();
    let pdf_name = filename.clone();
    
    // Create the stream
    let stream = stream::once(async move {
        let mut events = Vec::new();
        
        // Send start event
        let start_event = StreamStartEvent {
            status: "processing".to_string(),
            pdf: pdf_name.clone(),
        };
        events.push(
            axum::response::sse::Event::default()
                .event("start")
                .json_data(&start_event)
                .unwrap()
        );
        
        // Process pages
        let processor = PdfPageProcessor::new(pdf_path, dpi);
        match processor.process_pages(&models) {
            Ok(pages_data) => {
                let mut all_pages_data = Vec::new();
                let total_pages = pages_data.len();
                
                for (idx, (page_number, page_ocr_results, page_raw_ocr_results)) in pages_data.into_iter().enumerate() {
                    let page_text = page_ocr_results.join(" ");
                    
                    // Store page data
                    all_pages_data.push(PageData {
                        page_number,
                        ocr_text: page_text.clone(),
                        raw_ocr_results: page_raw_ocr_results,
                    });
                    
                    // Send page event
                    let page_event = StreamPageEvent {
                        page_number,
                        page_text,
                        total_pages_so_far: (idx + 1) as i32,
                    };
                    events.push(
                        axum::response::sse::Event::default()
                            .event("page")
                            .json_data(&page_event)
                            .unwrap()
                    );
                }
                
                // Send complete event
                let complete_event = StreamCompleteEvent {
                    status: "complete".to_string(),
                    total_pages: total_pages as i32,
                    pages: all_pages_data,
                    pdf_name: pdf_name.clone(),
                };
                events.push(
                    axum::response::sse::Event::default()
                        .event("complete")
                        .json_data(&complete_event)
                        .unwrap()
                );
            }
            Err(e) => {
                // Send error event
                let error_event = StreamErrorEvent {
                    status: "error".to_string(),
                    error: e.to_string(),
                };
                events.push(
                    axum::response::sse::Event::default()
                        .event("error")
                        .json_data(&error_event)
                        .unwrap()
                );
            }
        }
        
        stream::iter(events.into_iter().map(Ok))
    })
    .flatten();
    
    Ok(Sse::new(stream)
        .keep_alive(
            axum::response::sse::KeepAlive::new()
                .interval(std::time::Duration::from_secs(1))
        ))
}

/// Application error type
#[derive(Debug)]
pub enum AppError {
    BadRequest(String),
    InternalError(anyhow::Error),
}

impl From<anyhow::Error> for AppError {
    fn from(err: anyhow::Error) -> Self {
        AppError::InternalError(err)
    }
}

impl From<std::io::Error> for AppError {
    fn from(err: std::io::Error) -> Self {
        AppError::InternalError(err.into())
    }
}

impl From<axum::extract::multipart::MultipartError> for AppError {
    fn from(err: axum::extract::multipart::MultipartError) -> Self {
        AppError::InternalError(err.into())
    }
}

impl From<tokio::task::JoinError> for AppError {
    fn from(err: tokio::task::JoinError) -> Self {
        AppError::InternalError(err.into())
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_message) = match self {
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            AppError::InternalError(err) => {
                tracing::error!("Internal error: {:?}", err);
                (StatusCode::INTERNAL_SERVER_ERROR, format!("Internal server error: {}", err))
            }
        };
        
        let body = Json(serde_json::json!({
            "error": error_message,
        }));
        
        (status, body).into_response()
    }
}
