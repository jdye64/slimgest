"""
Example client demonstrating how to use the Slim-Gest API.
"""
import requests
import json
import time


def process_pdf_batch(pdf_path: str, api_url: str = "http://localhost:7670"):
    """
    Process a PDF using the batch endpoint.
    
    Args:
        pdf_path: Path to the PDF file
        api_url: Base URL of the API
    """
    print(f"Processing {pdf_path} (batch mode)...")
    
    with open(pdf_path, "rb") as f:
        print(f"Uploading {pdf_path}...")
        files = {"files": (pdf_path, f, "application/pdf")}
        print(f"Files: {files}")
        response = requests.post(f"{api_url}/process-pdfs", files=files)
        print(f"Response: {response.text}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Success! Processed {len(result.get('pdfs', []))} PDF(s)")
        return result
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


def process_pdf_stream(pdf_path: str, api_url: str = "http://localhost:7670"):
    """
    Process a PDF using the streaming endpoint.
    
    Args:
        pdf_path: Path to the PDF file
        api_url: Base URL of the API
    """
    print(f"Processing {pdf_path} (streaming mode)...")
    
    with open(pdf_path, "rb") as f:
        files = {"file": (pdf_path, f, "application/pdf")}
        
        with requests.post(
            f"{api_url}/process-pdf-stream",
            files=files,
            stream=True,
        ) as response:
            if response.status_code != 200:
                print(f"Error: {response.status_code} - {response.text}")
                return
            
            # Process Server-Sent Events
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    
                    # Parse SSE format
                    if line.startswith('event:'):
                        event_type = line.split(':', 1)[1].strip()
                    elif line.startswith('data:'):
                        data = json.loads(line.split(':', 1)[1].strip())
                        
                        if event_type == 'start':
                            print(f"Started processing: {data['pdf']}")
                        
                        elif event_type == 'page':
                            page_num = data['page_number']
                            pages_so_far = data['total_pages_so_far']
                            print(f"  Page {page_num} complete (total: {pages_so_far})")
                        
                        elif event_type == 'complete':
                            total_pages = data['total_pages']
                            print(f"Completed! Total pages: {total_pages}")
                            return data
                        
                        elif event_type == 'error':
                            print(f"Error: {data['error']}")
                            return None


def check_health(api_url: str = "http://localhost:7670"):
    """
    Check the health of the API.
    
    Args:
        api_url: Base URL of the API
    """
    response = requests.get(f"{api_url}/")
    if response.status_code == 200:
        data = response.json()
        print("API Status:")
        print(f"  Status: {data['status']}")
        print(f"  Workers: {data['workers']['alive']}/{data['workers']['total']}")
        print(f"  Queue size: {data['queue_size']}")
        return True
    else:
        print(f"API not available: {response.status_code}")
        return False


def check_job_status(job_id: str, api_url: str = "http://localhost:7670"):
    """
    Check the status of a job.
    
    Args:
        job_id: Job ID to check
        api_url: Base URL of the API
    """
    response = requests.get(f"{api_url}/jobs/{job_id}")
    if response.status_code == 200:
        data = response.json()
        print(f"Job {job_id}:")
        print(f"  Status: {data['status']}")
        print(f"  Created: {data['created_at']}")
        print(f"  Completed: {data['completed_at']}")
        print(f"  Has result: {data['has_result']}")
        if data['error']:
            print(f"  Error: {data['error']}")
        return data
    else:
        print(f"Job not found: {response.status_code}")
        return None


if __name__ == "__main__":
    import sys
    
    api_url = "http://localhost:7670"
    
    # Check health
    if not check_health(api_url):
        print("API is not running. Start it with: python -m slimgest.web")
        sys.exit(1)
    
    # Example usage
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        
        # Try batch mode
        print("\n" + "="*60)
        print("Testing batch mode:")
        print("="*60)
        result = process_pdf_batch(pdf_path, api_url)
        
        # Try streaming mode
        print("\n" + "="*60)
        print("Testing streaming mode:")
        print("="*60)
        result = process_pdf_stream(pdf_path, api_url)
        
        if result:
            print(f"\nExtracted text from {result['total_pages']} pages")
    else:
        print("Usage: python example_client.py <path_to_pdf>")
