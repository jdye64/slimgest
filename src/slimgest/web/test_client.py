#!/usr/bin/env python3
"""
Simple test client for the slim-gest FastAPI server.
"""
import sys
from pathlib import Path
import requests


def test_health_check(base_url: str = "http://localhost:7670"):
    """Test the health check endpoint."""
    print("Testing health check...")
    response = requests.get(f"{base_url}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def process_single_pdf(pdf_path: str, base_url: str = "http://localhost:7670", dpi: float = 150.0):
    """Process a single PDF file."""
    print(f"Processing single PDF: {pdf_path}")
    
    with open(pdf_path, "rb") as f:
        files = {"file": (Path(pdf_path).name, f, "application/pdf")}
        data = {"dpi": dpi}
        
        response = requests.post(f"{base_url}/process-pdf", files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Success!")
        print(f"Total pages processed: {result['total_pages_processed']}")
        print(f"Elapsed time: {result['elapsed_seconds']:.2f}s")
        
        for pdf_result in result['results']:
            print(f"\nPDF: {pdf_result['pdf_path']}")
            print(f"Pages: {pdf_result['pages_processed']}")
            print(f"OCR Text (first 200 chars): {pdf_result['ocr_text'][:200]}...")
        
        return result
    else:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
        return None


def process_multiple_pdfs(pdf_paths: list[str], base_url: str = "http://localhost:7670", dpi: float = 150.0):
    """Process multiple PDF files."""
    print(f"Processing {len(pdf_paths)} PDFs")
    
    files = []
    for pdf_path in pdf_paths:
        with open(pdf_path, "rb") as f:
            files.append(("files", (Path(pdf_path).name, f.read(), "application/pdf")))
    
    data = {"dpi": dpi}
    response = requests.post(f"{base_url}/process-pdfs", files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Success!")
        print(f"Total PDFs: {result['total_pdfs']}")
        print(f"Total pages processed: {result['total_pages_processed']}")
        print(f"Elapsed time: {result['elapsed_seconds']:.2f}s")
        
        for pdf_result in result['results']:
            print(f"\nPDF: {Path(pdf_result['pdf_path']).name}")
            print(f"Pages: {pdf_result['pages_processed']}")
            print(f"OCR Text (first 200 chars): {pdf_result['ocr_text'][:200]}...")
        
        return result
    else:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
        return None


def main():
    """Main entry point for testing."""
    if len(sys.argv) < 2:
        print("Usage: python test_client.py <pdf_file> [<pdf_file2> ...]")
        print("\nFirst, start the server:")
        print("  python -m slimgest.web")
        print("\nThen run this script with one or more PDF files:")
        print("  python test_client.py document.pdf")
        print("  python test_client.py doc1.pdf doc2.pdf doc3.pdf")
        sys.exit(1)
    
    base_url = "http://localhost:7670"
    
    # Test health check
    test_health_check(base_url)
    
    # Get PDF files from command line
    pdf_files = sys.argv[1:]
    
    # Validate files exist
    for pdf_file in pdf_files:
        if not Path(pdf_file).exists():
            print(f"Error: File not found: {pdf_file}")
            sys.exit(1)
        if not pdf_file.lower().endswith('.pdf'):
            print(f"Error: Not a PDF file: {pdf_file}")
            sys.exit(1)
    
    # Process PDFs
    if len(pdf_files) == 1:
        process_single_pdf(pdf_files[0], base_url)
    else:
        process_multiple_pdfs(pdf_files, base_url)


if __name__ == "__main__":
    main()
