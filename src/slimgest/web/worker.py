"""
Worker process for handling PDF processing tasks.
"""
import os
import json
import traceback
from pathlib import Path
from typing import Dict, Any
from multiprocessing import Process, Queue

import torch
from nemotron_page_elements_v3.model import define_model as define_model_page_elements
from nemotron_table_structure_v1.model import define_model as define_model_table_structure
from nemotron_graphic_elements_v1.model import define_model as define_model_graphic_elements
from nemotron_ocr.inference.pipeline import NemotronOCR

from slimgest.local.simple_all_gpu import run_pipeline, process_pdf_pages


class PDFWorker:
    """Worker process that handles PDF processing tasks."""
    
    def __init__(self, worker_id: int, request_queue: Queue, result_queue: Queue, ocr_model_dir: str, device: str = "cuda"):
        """
        Initialize the PDF worker.
        
        Args:
            worker_id: Unique identifier for this worker
            request_queue: Queue to receive processing requests
            result_queue: Queue to send results back
            ocr_model_dir: Directory containing the OCR model
            device: Device to use for models (e.g., 'cuda', 'cuda:0', 'cpu')
        """
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.ocr_model_dir = ocr_model_dir
        self.device = device
        self.models = {}
        
    def load_models(self):
        """Load all models in the worker process."""
        print(f"[WORKER-{self.worker_id}] Loading models on device: {self.device}...")
        
        # Load models
        self.models["page_elements"] = define_model_page_elements("page_element_v3").to(self.device)
        self.models["table_structure"] = define_model_table_structure("table_structure_v1").to(self.device)
        self.models["graphic_elements"] = define_model_graphic_elements("graphic_elements_v1").to(self.device)
        self.models["ocr"] = NemotronOCR(model_dir=self.ocr_model_dir, device=self.device)
        
        print(f"[WORKER-{self.worker_id}] Models loaded")
        print(f"[WORKER-{self.worker_id}]   - Page Elements (device: {self.models['page_elements'].device})")
        print(f"[WORKER-{self.worker_id}]   - Table Structure (device: {self.models['table_structure'].device})")
        print(f"[WORKER-{self.worker_id}]   - Graphic Elements (device: {self.models['graphic_elements'].device})")
        print(f"[WORKER-{self.worker_id}]   - OCR Model (device: {self.device})")
        
    def process_batch_request(self, job_id: str, pdf_paths: list, dpi: float) -> Dict[str, Any]:
        """
        Process a batch of PDFs.
        
        Args:
            job_id: Unique identifier for this job
            pdf_paths: List of paths to PDF files
            dpi: Resolution for PDF rendering
            
        Returns:
            Processing results
        """
        print(f"[WORKER-{self.worker_id}] process_batch_request called")
        print(f"[WORKER-{self.worker_id}]   job_id: {job_id}")
        print(f"[WORKER-{self.worker_id}]   pdf_paths: {pdf_paths}")
        print(f"[WORKER-{self.worker_id}]   dpi: {dpi}")
        
        try:
            # Check each file before processing
            print(f"[WORKER-{self.worker_id}] Checking file accessibility...")
            for idx, pdf_path in enumerate(pdf_paths):
                exists = os.path.exists(pdf_path)
                is_file = os.path.isfile(pdf_path) if exists else False
                readable = os.access(pdf_path, os.R_OK) if exists else False
                size = os.path.getsize(pdf_path) if exists else 0
                abs_path = os.path.abspath(pdf_path)
                
                print(f"[WORKER-{self.worker_id}] File {idx}: {pdf_path}")
                print(f"[WORKER-{self.worker_id}]   - Absolute path: {abs_path}")
                print(f"[WORKER-{self.worker_id}]   - Exists: {exists}")
                print(f"[WORKER-{self.worker_id}]   - Is file: {is_file}")
                print(f"[WORKER-{self.worker_id}]   - Readable: {readable}")
                print(f"[WORKER-{self.worker_id}]   - Size: {size} bytes")
                
                if not exists:
                    print(f"[WORKER-{self.worker_id}] ERROR: File does not exist!")
                    # List directory contents to debug
                    try:
                        parent_dir = os.path.dirname(pdf_path)
                        print(f"[WORKER-{self.worker_id}]   - Checking parent dir: {parent_dir}")
                        if os.path.exists(parent_dir):
                            contents = os.listdir(parent_dir)
                            print(f"[WORKER-{self.worker_id}]   - Parent dir contents: {contents}")
                        else:
                            print(f"[WORKER-{self.worker_id}]   - Parent dir does not exist!")
                    except Exception as list_err:
                        print(f"[WORKER-{self.worker_id}]   - Error listing directory: {list_err}")
            
            print(f"[WORKER-{self.worker_id}] Calling run_pipeline...")
            results = run_pipeline(
                pdf_files=pdf_paths,
                page_elements_model=self.models["page_elements"],
                table_structure_model=self.models["table_structure"],
                graphic_elements_model=self.models["graphic_elements"],
                ocr_model=self.models["ocr"],
                dpi=dpi,
                return_results=True,
            )
            
            print(f"[WORKER-{self.worker_id}] run_pipeline completed successfully")
            print(f"[WORKER-{self.worker_id}]   - Results type: {type(results)}")
            print(f"[WORKER-{self.worker_id}]   - Results length: {len(results) if results else 0}")
            
            return {
                "job_id": job_id,
                "status": "success",
                "results": results,
            }
            
        except Exception as e:
            print(f"[WORKER-{self.worker_id}] EXCEPTION in process_batch_request!")
            print(f"[WORKER-{self.worker_id}]   - Error type: {type(e).__name__}")
            print(f"[WORKER-{self.worker_id}]   - Error message: {str(e)}")
            print(f"[WORKER-{self.worker_id}]   - Traceback:")
            traceback.print_exc()
            
            return {
                "job_id": job_id,
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
    
    def process_stream_request(self, job_id: str, pdf_path: str, dpi: float) -> None:
        """
        Process a single PDF and stream results page by page.
        
        Args:
            job_id: Unique identifier for this job
            pdf_path: Path to the PDF file
            dpi: Resolution for PDF rendering
        """
        try:
            # Send start event
            self.result_queue.put({
                "job_id": job_id,
                "type": "start",
                "data": {
                    "status": "processing",
                    "pdf": Path(pdf_path).name,
                }
            })
            
            all_pages_data = []
            page_count = 0
            
            # Process pages
            for page_number, tensor, page_ocr_results, page_raw_ocr_results in process_pdf_pages(
                pdf_path,
                self.models["page_elements"],
                self.models["table_structure"],
                self.models["graphic_elements"],
                self.models["ocr"],
                device=self.device,
                dpi=dpi,
            ):
                page_count += 1
                page_text = " ".join(page_ocr_results)
                
                page_data = {
                    "page_number": page_number,
                    "ocr_text": page_text,
                    "raw_ocr_results": page_raw_ocr_results,
                }
                all_pages_data.append(page_data)
                
                # Send page completion event
                self.result_queue.put({
                    "job_id": job_id,
                    "type": "page",
                    "data": {
                        "page_number": page_number,
                        "page_text": page_text,
                        "total_pages_so_far": page_count,
                    }
                })
            
            # Send completion event
            self.result_queue.put({
                "job_id": job_id,
                "type": "complete",
                "data": {
                    "status": "complete",
                    "total_pages": page_count,
                    "pages": all_pages_data,
                    "pdf_name": Path(pdf_path).name,
                }
            })
            
        except Exception as e:
            # Send error event
            self.result_queue.put({
                "job_id": job_id,
                "type": "error",
                "data": {
                    "status": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            })
    
    def run(self):
        """Main worker loop."""
        print(f"[WORKER-{self.worker_id}] Starting...")
        
        # Load models in this process
        self.load_models()
        
        print(f"[WORKER-{self.worker_id}] Ready to process requests")
        print(f"[WORKER-{self.worker_id}] PID: {os.getpid()}")
        print(f"[WORKER-{self.worker_id}] CWD: {os.getcwd()}")
        
        # Process requests
        while True:
            try:
                # Get request from queue (blocking)
                print(f"[WORKER-{self.worker_id}] Waiting for request from queue...")
                request = self.request_queue.get()
                
                # Check for shutdown signal
                if request is None or request.get("type") == "shutdown":
                    print(f"[WORKER-{self.worker_id}] Received shutdown signal")
                    break
                
                job_id = request["job_id"]
                request_type = request["type"]
                
                print(f"[WORKER-{self.worker_id}] ===== Processing job {job_id} =====")
                print(f"[WORKER-{self.worker_id}] Request type: {request_type}")
                print(f"[WORKER-{self.worker_id}] Full request: {request}")
                
                if request_type == "batch":
                    # Process batch request
                    print(f"[WORKER-{self.worker_id}] Calling process_batch_request...")
                    result = self.process_batch_request(
                        job_id=job_id,
                        pdf_paths=request["pdf_paths"],
                        dpi=request["dpi"],
                    )
                    print(f"[WORKER-{self.worker_id}] Putting result on result queue: {result.get('status')}")
                    self.result_queue.put(result)
                    
                elif request_type == "stream":
                    # Process streaming request
                    print(f"[WORKER-{self.worker_id}] Calling process_stream_request...")
                    self.process_stream_request(
                        job_id=job_id,
                        pdf_path=request["pdf_path"],
                        dpi=request["dpi"],
                    )
                
                print(f"[WORKER-{self.worker_id}] ===== Completed job {job_id} =====")
                
            except Exception as e:
                print(f"[WORKER-{self.worker_id}] ERROR in main loop: {e}")
                traceback.print_exc()
        
        print(f"[WORKER-{self.worker_id}] Shutting down")


def start_worker(worker_id: int, request_queue: Queue, result_queue: Queue, ocr_model_dir: str, device: str = "cuda"):
    """
    Start a worker process.
    
    Args:
        worker_id: Unique identifier for this worker
        request_queue: Queue to receive processing requests
        result_queue: Queue to send results back
        ocr_model_dir: Directory containing the OCR model
        device: Device to use for models (e.g., 'cuda', 'cuda:0', 'cpu')
    """
    worker = PDFWorker(worker_id, request_queue, result_queue, ocr_model_dir, device)
    worker.run()
