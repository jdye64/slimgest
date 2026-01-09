#!/usr/bin/env python3
"""
Test client for the slim-gest FastAPI server with SSE streaming support.
"""
import sys
import json
import time
from pathlib import Path
from typing import Optional, List
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


def test_health_check(base_url: str):
    """Test the health check endpoint."""
    print("Testing health check...")
    try:
        response = requests.get(f"{base_url}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Health check failed: {e}")
    print()


def process_pdf_stream(
    pdf_path: str, 
    base_url: str, 
    dpi: float = 150.0,
    output_dir: Optional[Path] = None
) -> Optional[dict]:
    """
    Process a single PDF file using SSE streaming endpoint.
    
    Args:
        pdf_path: Path to the PDF file
        base_url: Base URL of the API server
        dpi: DPI for PDF rendering
        output_dir: Directory to save markdown output (optional)
    
    Returns:
        Dictionary with processing results or None on error
    """
    pdf_file = Path(pdf_path)
    print(f"\n{'='*80}")
    print(f"Processing: {pdf_file.name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        with open(pdf_path, "rb") as f:
            files = {"file": (pdf_file.name, f, "application/pdf")}
            data = {"dpi": dpi}
            
            # Make streaming request
            response = requests.post(
                f"{base_url}/process-pdf-stream",
                files=files,
                data=data,
                stream=True,
                timeout=600,  # 10 minute timeout
            )
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
        
        # Process SSE stream
        pages_data = []
        total_pages = 0
        accumulated_text = []
        
        for line in response.iter_lines():
            if not line:
                continue
            
            line = line.decode('utf-8')
            
            # Parse SSE format
            if line.startswith('event:'):
                event_type = line.split(':', 1)[1].strip()
            elif line.startswith('data:'):
                data_str = line.split(':', 1)[1].strip()
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                
                # Handle different event types
                if event_type == 'start':
                    print(f"Status: {data.get('status', 'unknown')}")
                
                elif event_type == 'page':
                    page_num = data.get('page_number', '?')
                    page_text = data.get('page_text', '')
                    total_so_far = data.get('total_pages_so_far', 0)
                    
                    print(f"\n--- Page {page_num} (Total processed: {total_so_far}) ---")
                    # Print first 200 chars of page text
                    preview = page_text[:200] if len(page_text) > 200 else page_text
                    print(f"Text: {preview}{'...' if len(page_text) > 200 else ''}")
                    
                    accumulated_text.append(f"\n\n## Page {page_num}\n\n{page_text}")
                
                elif event_type == 'complete':
                    total_pages = data.get('total_pages', 0)
                    pages_data = data.get('pages', [])
                    pdf_name = data.get('pdf_name', pdf_file.name)
                    
                    elapsed = time.time() - start_time
                    print(f"\n{'='*80}")
                    print(f"✓ Complete: {pdf_name}")
                    print(f"  Total pages: {total_pages}")
                    print(f"  Time: {elapsed:.2f}s")
                    print(f"{'='*80}")
                    
                    # Save to markdown if output directory is specified
                    if output_dir:
                        output_dir.mkdir(parents=True, exist_ok=True)
                        md_filename = pdf_file.stem + ".md"
                        md_path = output_dir / md_filename
                        
                        # Create markdown content
                        md_content = f"# {pdf_file.name}\n\n"
                        md_content += f"**Total Pages:** {total_pages}\n\n"
                        md_content += f"**Processed:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                        md_content += "---\n\n"
                        md_content += "".join(accumulated_text)
                        
                        with open(md_path, 'w', encoding='utf-8') as f:
                            f.write(md_content)
                        
                        print(f"✓ Saved markdown to: {md_path}")
                    
                    return {
                        'pdf_name': pdf_name,
                        'total_pages': total_pages,
                        'elapsed_seconds': elapsed,
                        'pages': pages_data,
                        'full_text': "".join(accumulated_text),
                    }
                
                elif event_type == 'error':
                    error_msg = data.get('error', 'Unknown error')
                    print(f"Error: {error_msg}")
                    return None
        
        return None
    
    except Exception as e:
        print(f"Exception processing {pdf_file.name}: {e}")
        return None


def process_directory(
    directory: Path,
    base_url: str,
    dpi: float = 150.0,
    output_dir: Optional[Path] = None,
    max_workers: int = 16,
):
    """
    Process all PDF files in a directory with concurrent requests.
    
    Args:
        directory: Directory containing PDF files
        base_url: Base URL of the API server
        dpi: DPI for PDF rendering
        output_dir: Directory to save markdown outputs
        max_workers: Maximum number of concurrent requests (default: 16)
    """
    # Find all PDF files
    pdf_files = sorted([f for f in directory.iterdir() if f.is_file() and f.suffix.lower() == '.pdf'])
    
    if not pdf_files:
        print(f"No PDF files found in {directory}")
        return
    
    print(f"\nFound {len(pdf_files)} PDF files in {directory}")
    print(f"Processing with up to {max_workers} concurrent requests...\n")
    
    start_time = time.time()
    results = []
    
    # Process PDFs concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_pdf = {
            executor.submit(process_pdf_stream, str(pdf), base_url, dpi, output_dir): pdf
            for pdf in pdf_files
        }
        
        # Process completed tasks
        for future in as_completed(future_to_pdf):
            pdf = future_to_pdf[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error processing {pdf.name}: {e}")
    
    # Summary
    elapsed = time.time() - start_time
    successful = len(results)
    failed = len(pdf_files) - successful
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total PDFs: {len(pdf_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Average time per PDF: {elapsed/len(pdf_files):.2f}s")
    
    total_pages = sum(r['total_pages'] for r in results)
    print(f"Total pages processed: {total_pages}")
    
    if output_dir:
        print(f"\nMarkdown files saved to: {output_dir}")
    print(f"{'='*80}\n")


def main():
    """Main entry point for testing."""
    if len(sys.argv) < 2:
        print("Usage: python test_client.py <pdf_file_or_directory> [--output-dir <dir>] [--dpi <dpi>] [--workers <n>] [--url <url>]")
        print("\nExamples:")
        print("  # Process a single PDF")
        print("  python test_client.py document.pdf --output-dir ./output")
        print()
        print("  # Process all PDFs in a directory (16 concurrent by default)")
        print("  python test_client.py ./pdfs/ --output-dir ./output")
        print()
        print("  # Adjust concurrency level")
        print("  python test_client.py ./pdfs/ --output-dir ./output --workers 32")
        print()
        print("  # Set server URL")
        print("  python test_client.py document.pdf --url http://myhost:7777")
        print()
        print("Options:")
        print("  --output-dir <dir>   Directory to save markdown files (default: ./output)")
        print("  --dpi <dpi>          DPI for PDF rendering (default: 150.0)")
        print("  --workers <n>        Max concurrent requests for directory processing (default: 16)")
        print("  --url <url>          Base URL of API server (default: http://localhost:7670)")
        sys.exit(1)
    
    # Default values
    base_url = "http://localhost:7670"
    input_path = Path(sys.argv[1])
    output_dir = Path("./output")
    dpi = 150.0
    max_workers = 4

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--output-dir' and i + 1 < len(sys.argv):
            output_dir = Path(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--dpi' and i + 1 < len(sys.argv):
            dpi = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--workers' and i + 1 < len(sys.argv):
            max_workers = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--url' and i + 1 < len(sys.argv):
            base_url = sys.argv[i + 1].rstrip("/")
            i += 2
        else:
            i += 1
    
    # Validate input path
    if not input_path.exists():
        print(f"Error: Path not found: {input_path}")
        sys.exit(1)
    
    # Test health check
    test_health_check(base_url)
    
    # Process based on input type
    if input_path.is_file():
        # Single file
        if not input_path.suffix.lower() == '.pdf':
            print(f"Error: Not a PDF file: {input_path}")
            sys.exit(1)
        
        process_pdf_stream(str(input_path), base_url, dpi, output_dir)
    
    elif input_path.is_dir():
        # Directory with multiple PDFs
        process_directory(input_path, base_url, dpi, output_dir, max_workers)
    
    else:
        print(f"Error: Invalid input path: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
