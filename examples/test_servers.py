#!/usr/bin/env python3
"""
Simple test script to verify both Python and Rust servers are working correctly.
"""
import sys
import requests
from pathlib import Path

def test_server(server_url: str, server_name: str, pdf_path: Path):
    """Test a server with a PDF file"""
    print(f"\n{'='*60}")
    print(f"Testing {server_name} server at {server_url}")
    print(f"{'='*60}")
    
    # Test health endpoint
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{server_url}/", timeout=5)
        response.raise_for_status()
        print(f"   ✓ Health check passed: {response.json()}")
    except Exception as e:
        print(f"   ✗ Health check failed: {e}")
        return False
    
    # Test PDF processing
    print(f"2. Testing PDF processing with {pdf_path.name}...")
    try:
        with open(pdf_path, "rb") as f:
            files = {"file": (pdf_path.name, f, "application/pdf")}
            data = {"dpi": "150"}
            response = requests.post(
                f"{server_url}/process-pdf",
                files=files,
                data=data,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
        print(f"   ✓ PDF processed successfully")
        print(f"   - Total pages: {result.get('total_pages_processed', 'N/A')}")
        print(f"   - Total PDFs: {result.get('total_pdfs', 'N/A')}")
        print(f"   - Elapsed: {result.get('elapsed_seconds', 'N/A'):.2f}s")
        
        return True
    except Exception as e:
        print(f"   ✗ PDF processing failed: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_servers.py <path_to_test_pdf>")
        print("Example: python test_servers.py test.pdf")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    python_url = "http://localhost:7670"
    rust_url = "http://localhost:7671"
    
    print("\n" + "="*60)
    print("Server Compatibility Test")
    print("="*60)
    print(f"Test PDF: {pdf_path}")
    print(f"Python server: {python_url}")
    print(f"Rust server: {rust_url}")
    
    # Test both servers
    python_ok = test_server(python_url, "Python FastAPI", pdf_path)
    rust_ok = test_server(rust_url, "Rust Axum", pdf_path)
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Python FastAPI: {'✓ PASS' if python_ok else '✗ FAIL'}")
    print(f"Rust Axum:      {'✓ PASS' if rust_ok else '✗ FAIL'}")
    
    if python_ok and rust_ok:
        print("\n✓ Both servers are working correctly!")
        print("You can now run the full benchmark with:")
        print(f"  python examples/benchmark_servers.py /path/to/pdfs/")
        sys.exit(0)
    else:
        print("\n✗ One or more servers failed. Check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
