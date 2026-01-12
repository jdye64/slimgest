"""
Example client demonstrating how to use the Slim-Gest API.

This version submits every PDF in a specified input directory to the API endpoint
asynchronously to maximize throughput.
"""
import asyncio
import aiohttp
import aiofiles
import json
import os
import sys
from pathlib import Path

API_URL_DEFAULT = "http://localhost:7670"

async def process_pdf_stream_async(pdf_path: Path, api_url: str = API_URL_DEFAULT, session: aiohttp.ClientSession = None):
    """
    Process a PDF using the streaming endpoint (async version).
    Args:
        pdf_path: Path to the PDF file
        api_url: Base URL of the API
        session: aiohttp ClientSession to reuse connections
    """
    print(f"Processing {pdf_path} (streaming mode)...")
    if not pdf_path.is_file():
        print(f"File {pdf_path} does not exist or is not a file")
        return None

    async with aiofiles.open(pdf_path, "rb") as f:
        pdf_bytes = await f.read()

    data = aiohttp.FormData()
    data.add_field(
        'file', pdf_bytes,
        filename=pdf_path.name,
        content_type='application/pdf'
    )

    try:
        async with session.post(f"{api_url}/process-pdf-stream", data=data) as response:
            if response.status != 200:
                print(f"Error: {response.status} - {await response.text()}")
                return None

            event_type = None
            result = None

            async for line_bytes in response.content:
                line = line_bytes.decode('utf-8').strip()
                if not line:
                    continue
                # Parse SSE
                if line.startswith('event:'):
                    event_type = line.split(':', 1)[1].strip()
                elif line.startswith('data:'):
                    try:
                        data_obj = json.loads(line.split(':', 1)[1].strip())
                    except Exception:
                        print(f"Could not decode JSON data line: {line}")
                        continue

                    if event_type == 'start':
                        print(f"{pdf_path.name}: Started processing ({data_obj.get('pdf', '')})")
                    elif event_type == 'page':
                        page_num = data_obj.get('page_number')
                        so_far = data_obj.get('total_pages_so_far')
                        print(f"{pdf_path.name}:   Page {page_num} complete (total: {so_far})")
                    elif event_type == 'complete':
                        total_pages = data_obj.get('total_pages')
                        print(f"{pdf_path.name}: Completed! Total pages: {total_pages}")
                        result = data_obj
                        break  # Typically 'complete' is a finish event.
                    elif event_type == 'error':
                        print(f"{pdf_path.name}: Error: {data_obj.get('error')}")
                        result = None
                        break
            return result
    except Exception as e:
        print(f"{pdf_path.name}: Exception occurred: {e}")
        return None

async def check_health_async(api_url: str = API_URL_DEFAULT, session: aiohttp.ClientSession = None):
    try:
        async with session.get(f"{api_url}/") as response:
            if response.status == 200:
                data = await response.json()
                print("API Status:")
                print(f"  Status: {data['status']}")
                print(f"  Workers: {data['workers']['alive']}/{data['workers']['total']}")
                print(f"  Queue size: {data['queue_size']}")
                return True
            else:
                print(f"API not available: {response.status}")
                return False
    except Exception as e:
        print(f"Exception checking API health: {e}")
        return False

async def process_directory(input_dir: Path, api_url: str = API_URL_DEFAULT, max_parallel: int = 8):
    """
    Process all PDF files in input_dir, submitting them to the API as fast as possible (async).
    Args:
        input_dir: Directory containing .pdf files
        api_url: API base URL
        max_parallel: Maximum number of concurrent requests (default 8)
    """
    pdf_files = sorted([f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"])
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    print(f"Found {len(pdf_files)} PDF(s) in {input_dir}")

    results = {}
    connector = aiohttp.TCPConnector(limit_per_host=max_parallel)
    async with aiohttp.ClientSession(connector=connector) as session:
        healthy = await check_health_async(api_url, session)
        if not healthy:
            print("API is not running. Start it with: python -m slimgest.web")
            return

        # Use asyncio.Semaphore for controlling concurrency
        semaphore = asyncio.Semaphore(max_parallel)

        async def sem_task(pdf_path):
            async with semaphore:
                return pdf_path.name, await process_pdf_stream_async(pdf_path, api_url, session)

        tasks = [sem_task(pdf_path) for pdf_path in pdf_files]
        # Schedule all tasks and gather as they finish
        for fut in asyncio.as_completed(tasks):
            name, result = await fut
            results[name] = result
            if result:
                if 'total_pages' in result:
                    print(f"{name}: Extracted text from {result['total_pages']} pages")
                else:
                    print(f"{name}: Processing succeeded.")
            else:
                print(f"{name}: Failed to process.")

    print("\nSummary:")
    for name, result in results.items():
        if not result:
            print(f"  {name}: failed")
        else:
            if 'total_pages' in result:
                print(f"  {name}: {result['total_pages']} page(s) processed")
            else:
                print(f"  {name}: processed")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process every PDF in a directory via the Slim-Gest API using async for maximum throughput.")
    parser.add_argument("input_dir", help="Directory containing PDF files")
    parser.add_argument("--api-url", help="API base URL (default: http://localhost:7670)", default=API_URL_DEFAULT)
    parser.add_argument("--max-parallel", help="Maximum concurrent requests (default: 8)", type=int, default=8)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Run the async processing loop
    asyncio.run(process_directory(input_dir, args.api_url, args.max_parallel))
