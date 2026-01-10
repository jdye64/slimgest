use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub message: String,
    pub models_loaded: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProcessPdfRequest {
    pub dpi: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PageData {
    pub page_number: i32,
    pub ocr_text: String,
    pub raw_ocr_results: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PdfResult {
    pub pdf_path: String,
    pub pages_processed: i32,
    pub ocr_text: String,
    pub raw_ocr_results: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProcessPdfResponse {
    pub total_pages_processed: i32,
    pub total_pdfs: i32,
    pub elapsed_seconds: f64,
    pub results: Vec<PdfResult>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StreamStartEvent {
    pub status: String,
    pub pdf: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StreamPageEvent {
    pub page_number: i32,
    pub page_text: String,
    pub total_pages_so_far: i32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StreamCompleteEvent {
    pub status: String,
    pub total_pages: i32,
    pub pages: Vec<PageData>,
    pub pdf_name: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StreamErrorEvent {
    pub status: String,
    pub error: String,
}
