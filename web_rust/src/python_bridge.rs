use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::Mutex;
use anyhow::{Context, Result};

/// Holds references to the loaded Python models
pub struct PythonModels {
    pub page_elements: Py<PyAny>,
    pub table_structure: Py<PyAny>,
    pub graphic_elements: Py<PyAny>,
    pub ocr: Py<PyAny>,
}

/// Thread-safe wrapper for Python models
pub struct Models {
    pub inner: Mutex<PythonModels>,
}

impl Models {
    pub fn new(models: PythonModels) -> Self {
        Self {
            inner: Mutex::new(models),
        }
    }
}

/// Initialize Python interpreter and load all models
pub fn initialize_models() -> Result<Models> {
    pyo3::prepare_freethreaded_python();
    
    Python::with_gil(|py| {
        // #region agent log
        // Check Python encoding settings (Hypothesis A, D)
        let os_module = py.import_bound("os")?;
        let environ = os_module.getattr("environ")?;
        
        let pythonioencoding = environ.call_method1("get", ("PYTHONIOENCODING", "NOT_SET"))?;
        let lang = environ.call_method1("get", ("LANG", "NOT_SET"))?;
        let lc_all = environ.call_method1("get", ("LC_ALL", "NOT_SET"))?;
        
        tracing::info!("Python environment encoding check (Hypothesis A):");
        tracing::info!("  PYTHONIOENCODING: {:?}", pythonioencoding);
        tracing::info!("  LANG: {:?}", lang);
        tracing::info!("  LC_ALL: {:?}", lc_all);
        // #endregion
        
        // Import required modules
        let sys = py.import_bound("sys")?;
        let sys_path = sys.getattr("path")?;
        
        // Add the src directory to Python path
        sys_path.call_method1("insert", (0, "/app/src"))?;
        
        // Import model definition functions
        let page_elements_module = py.import_bound("nemotron_page_elements_v3.model")?;
        let table_structure_module = py.import_bound("nemotron_table_structure_v1.model")?;
        let graphic_elements_module = py.import_bound("nemotron_graphic_elements_v1.model")?;
        let ocr_module = py.import_bound("nemotron_ocr.inference.pipeline")?;
        
        // Load models
        tracing::info!("Loading page elements model...");
        let page_elements = page_elements_module
            .getattr("define_model")?
            .call1(("page_element_v3",))?
            .into();
        
        tracing::info!("Loading table structure model...");
        let table_structure = table_structure_module
            .getattr("define_model")?
            .call1(("table_structure_v1",))?
            .into();
        
        tracing::info!("Loading graphic elements model...");
        let graphic_elements = graphic_elements_module
            .getattr("define_model")?
            .call1(("graphic_elements_v1",))?
            .into();
        
        tracing::info!("Loading OCR model...");
        let ocr_class = ocr_module.getattr("NemotronOCR")?;
        let ocr_model_dir = std::env::var("NEMOTRON_OCR_MODEL_DIR")
            .unwrap_or_else(|_| "/app/models/nemotron-ocr-v1/checkpoints".to_string());
        
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("model_dir", ocr_model_dir)?;
        let ocr = ocr_class.call((), Some(&kwargs))?.into();
        
        let models = PythonModels {
            page_elements,
            table_structure,
            graphic_elements,
            ocr,
        };
        
        Ok::<Models, anyhow::Error>(Models::new(models))
    })
    .context("Failed to initialize Python models")
}

/// Process PDF files using the Python pipeline
pub fn process_pdf_files(
    models: &Models,
    pdf_paths: Vec<String>,
    dpi: f64,
) -> Result<serde_json::Value> {
    Python::with_gil(|py| {
        let models_guard = models.inner.lock().unwrap();
        
        // Import the run_pipeline function
        let pipeline_module = py.import_bound("slimgest.local.simple_all_gpu")?;
        let run_pipeline = pipeline_module.getattr("run_pipeline")?;
        
        // Convert pdf_paths to Python list
        let py_pdf_paths = PyList::new_bound(py, pdf_paths);
        
        // Create kwargs dictionary
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("pdf_files", py_pdf_paths)?;
        kwargs.set_item("page_elements_model", &models_guard.page_elements)?;
        kwargs.set_item("table_structure_model", &models_guard.table_structure)?;
        kwargs.set_item("graphic_elements_model", &models_guard.graphic_elements)?;
        kwargs.set_item("ocr_model", &models_guard.ocr)?;
        kwargs.set_item("dpi", dpi)?;
        kwargs.set_item("return_results", true)?;
        
        // Call the function
        let result = run_pipeline.call((), Some(&kwargs))?;
        
        // Convert Python result to JSON
        let json_module = py.import_bound("json")?;
        let json_dumps = json_module.getattr("dumps")?;
        let json_str: String = json_dumps.call1((result,))?.extract()?;
        
        // Parse JSON string to serde_json::Value
        let json_value: serde_json::Value = serde_json::from_str(&json_str)?;
        
        Ok::<serde_json::Value, anyhow::Error>(json_value)
    })
    .context("Failed to process PDF files")
}

/// Generator-based PDF processing for streaming
pub struct PdfPageProcessor {
    pdf_path: String,
    dpi: f64,
}

impl PdfPageProcessor {
    pub fn new(pdf_path: String, dpi: f64) -> Self {
        Self { pdf_path, dpi }
    }
    
    /// Process PDF pages and yield results
    pub fn process_pages(
        &self,
        models: &Models,
    ) -> Result<Vec<(i32, Vec<String>, Vec<String>)>> {
        Python::with_gil(|py| {
            let models_guard = models.inner.lock().unwrap();
            
            // Import the process_pdf_pages function
            let pipeline_module = py.import_bound("slimgest.local.simple_all_gpu")?;
            let process_pdf_pages = pipeline_module.getattr("process_pdf_pages")?;
            
            // Call the generator function
            let generator = process_pdf_pages.call1((
                &self.pdf_path,
                &models_guard.page_elements,
                &models_guard.table_structure,
                &models_guard.graphic_elements,
                &models_guard.ocr,
                "cuda",
                self.dpi,
            ))?;
            
            // Iterate through generator results
            let mut results = Vec::new();
            for item in generator.iter()? {
                let item = item?;
                
                // Extract tuple: (page_number, tensor, page_ocr_results, page_raw_ocr_results)
                let tuple = item.downcast::<pyo3::types::PyTuple>()
                    .map_err(|e| anyhow::anyhow!("Failed to downcast to tuple: {}", e))?;
                let page_number: i32 = tuple.get_item(0)?.extract()?;
                
                // Extract OCR results (skip tensor)
                let page_ocr_results: Vec<String> = tuple.get_item(2)?.extract()?;
                let page_raw_ocr_results: Vec<String> = tuple.get_item(3)?.extract()?;
                
                results.push((page_number, page_ocr_results, page_raw_ocr_results));
            }
            
            Ok::<Vec<(i32, Vec<String>, Vec<String>)>, anyhow::Error>(results)
        })
        .context("Failed to process PDF pages")
    }
}
