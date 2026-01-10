# Slim-Gest Rust Web Server

A high-performance Rust web server implementation using Axum and PyO3 to benchmark against the Python FastAPI implementation.

## Overview

This Rust server provides the same API endpoints as the Python FastAPI server but uses:
- **Axum**: A fast, ergonomic Rust web framework
- **PyO3**: Python bindings to call the existing ML models
- **Tokio**: Async runtime for concurrent request handling

## Architecture

The server consists of:
- `main.rs`: Server initialization and routing
- `handlers.rs`: HTTP request handlers
- `models.rs`: Data structures for requests/responses
- `python_bridge.rs`: PyO3 bindings to call Python ML models
- `lib.rs`: Library interface

## API Endpoints

All endpoints match the Python server:

### `GET /`
Health check endpoint

### `POST /process-pdf`
Process a single PDF file
- **Body**: multipart/form-data
  - `file`: PDF file
  - `dpi`: (optional) DPI for rendering (default: 150.0)

### `POST /process-pdfs`
Process multiple PDF files
- **Body**: multipart/form-data
  - `files`: Multiple PDF files
  - `dpi`: (optional) DPI for rendering (default: 150.0)

### `POST /process-pdf-stream`
Process a PDF with Server-Sent Events streaming
- **Body**: multipart/form-data
  - `file`: PDF file
  - `dpi`: (optional) DPI for rendering (default: 150.0)

## Building

### Development
```bash
cargo build
```

### Production
```bash
cargo build --release
```

## Running

The server runs on port **7671** by default:

```bash
./target/release/slimgest-web-rust
```

Or with cargo:
```bash
cargo run --release
```

## Environment Variables

- `NEMOTRON_OCR_MODEL_DIR`: Path to OCR model checkpoints (default: `/app/models/nemotron-ocr-v1/checkpoints`)
- `PYTHONPATH`: Should include the `src` directory for Python imports

## Docker

The Dockerfile builds both Python and Rust servers. Both servers are started automatically:
- Python server: port 7670
- Rust server: port 7671

Build the image:
```bash
docker build -t slim-gest .
```

Run the container:
```bash
docker run -p 7670:7670 -p 7671:7671 -v /path/to/models:/app/models slim-gest
```

## Benchmarking

Use the provided benchmark script to compare performance:

```bash
python examples/benchmark_servers.py /path/to/pdfs \
    --python-url http://localhost:7670 \
    --rust-url http://localhost:7671 \
    --dpi 150 \
    --concurrent 1 \
    --output benchmark_results.json
```

## Performance Expectations

The Rust server aims to reduce overhead in:
1. **HTTP request handling**: Axum is faster than FastAPI/Starlette
2. **Multipart parsing**: Native Rust implementation
3. **JSON serialization**: Faster with serde
4. **Memory management**: Better control over allocations

However, since the actual ML processing happens in Python (via PyO3), the performance improvement will primarily be in the web layer overhead, not the model inference time.

## Dependencies

Key dependencies:
- `axum = "0.7"`: Web framework
- `tokio = "1"`: Async runtime
- `pyo3 = "0.21"`: Python bindings
- `serde = "1.0"`: Serialization
- `tower-http = "0.5"`: HTTP middleware

See `Cargo.toml` for full list.

## License

MIT
