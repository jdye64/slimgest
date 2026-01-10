# Benchmarking Guide: Python vs Rust Web Servers

This guide walks you through benchmarking the Python FastAPI and Rust Axum web servers to compare their performance.

## Quick Start

### 1. Build the Docker Image

```bash
docker build -t slimgest .
```

This will:
- Install Python dependencies
- Install Rust and compile the Rust web server
- Set up both servers to run simultaneously

### 2. Run the Container

```bash
docker run -d \
  --name slimgest-benchmark \
  -p 7670:7670 \
  -p 7671:7671 \
  -v /path/to/your/models:/app/models \
  --gpus all \
  slimgest
```

Replace `/path/to/your/models` with your actual models directory.

### 3. Verify Both Servers are Running

Check Python server (port 7670):
```bash
curl http://localhost:7670/
```

Check Rust server (port 7671):
```bash
curl http://localhost:7671/
```

Both should return a health check response.

### 4. Prepare Test PDFs

Create a directory with PDF files to test:
```bash
mkdir test_pdfs
# Copy some PDF files into test_pdfs/
```

### 5. Run the Benchmark

```bash
python examples/benchmark_servers.py test_pdfs/ \
    --python-url http://localhost:7670 \
    --rust-url http://localhost:7671 \
    --dpi 150 \
    --concurrent 1 \
    --max-files 10 \
    --output benchmark_results.json
```

## Benchmark Options

- `pdf_directory`: Directory containing PDF files to process (required)
- `--python-url`: Python server URL (default: http://localhost:7670)
- `--rust-url`: Rust server URL (default: http://localhost:7671)
- `--dpi`: DPI for PDF rendering (default: 150.0)
- `--concurrent`: Number of concurrent requests (default: 1)
- `--max-files`: Limit number of PDF files to process (default: all)
- `--output`: Output file for detailed JSON results (default: benchmark_results.json)

## Understanding the Results

The benchmark will show:

1. **Avg Latency**: Average time to process a single PDF
2. **Min/Max Latency**: Fastest and slowest processing times
3. **Pages per Second**: Throughput metric
4. **Requests per Second**: How many PDFs can be processed per second
5. **Success/Failure Counts**: Reliability metrics

### Example Output

```
╭─────────────────────────────────┬──────────────────┬─────────────┬────────────╮
│ Metric                          │ Python FastAPI   │ Rust Axum   │ Difference │
├─────────────────────────────────┼──────────────────┼─────────────┼────────────┤
│ Avg Latency (seconds)           │ 2.450            │ 2.380       │ +2.9%      │
│ Pages per Second                │ 0.82             │ 0.84        │ +2.4%      │
│ Requests per Second             │ 0.41             │ 0.42        │ +2.4%      │
╰─────────────────────────────────┴──────────────────┴─────────────┴────────────╯
```

## What to Expect

### Performance Improvements

The Rust server should show improvements in:
- **HTTP request handling**: Axum has lower overhead than FastAPI/Starlette
- **Multipart parsing**: Native Rust implementation is faster
- **JSON serialization**: serde is faster than Python's json module
- **Memory efficiency**: Better memory allocation patterns

### Where Performance is Similar

Since both servers use the same Python ML models (via PyO3), the actual model inference time will be identical. The improvement is in the **web layer overhead**, not the ML processing.

### Typical Results

For GPU-bound workloads (like this OCR pipeline), expect:
- **2-8% improvement** in average latency
- **Higher improvements** for small PDFs (less ML time, more relative web overhead)
- **Lower improvements** for large PDFs (ML time dominates)

## Advanced Benchmarking

### Testing with Concurrent Requests

```bash
python examples/benchmark_servers.py test_pdfs/ \
    --concurrent 4 \
    --max-files 20
```

This tests how well each server handles concurrent load.

### Testing Different DPI Settings

```bash
# Low DPI (faster processing)
python examples/benchmark_servers.py test_pdfs/ --dpi 100 --output results_100dpi.json

# High DPI (slower processing)
python examples/benchmark_servers.py test_pdfs/ --dpi 300 --output results_300dpi.json
```

### Analyzing Detailed Results

The JSON output contains per-file timing information:

```bash
# Pretty-print the results
cat benchmark_results.json | python -m json.tool

# Extract specific metrics
jq '.python.statistics.avg_latency' benchmark_results.json
jq '.rust.statistics.avg_latency' benchmark_results.json
```

## Troubleshooting

### Servers Not Starting

Check Docker logs:
```bash
docker logs slimgest-benchmark
```

### Out of Memory Errors

Reduce concurrent requests or process fewer files:
```bash
python examples/benchmark_servers.py test_pdfs/ --concurrent 1 --max-files 5
```

### Connection Timeouts

Increase timeout for large PDFs by modifying the benchmark script's timeout value (currently 300 seconds).

## Local Development

To run servers locally without Docker:

### Python Server
```bash
cd /path/to/slim-gest
source .venv/bin/activate
python -m slimgest.web  # Runs on port 7670
```

### Rust Server
```bash
cd /path/to/slim-gest/web_rust
export PYTHONPATH=/path/to/slim-gest/src
export NEMOTRON_OCR_MODEL_DIR=/path/to/models/nemotron-ocr-v1/checkpoints
cargo run --release  # Runs on port 7671
```

Then run the benchmark as usual.

## Conclusion

This benchmark helps you understand:
1. The web layer overhead in your application
2. Whether a Rust rewrite would provide meaningful improvements
3. How the system scales with concurrent requests

For this project, the ML inference is the bottleneck, so Rust provides modest but consistent improvements in web layer efficiency.
