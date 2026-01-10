# Quick Reference: Rust vs Python Servers

## Server Information

| Aspect | Python FastAPI | Rust Axum |
|--------|----------------|-----------|
| **Port** | 7670 | 7671 |
| **Framework** | FastAPI + Starlette | Axum + Tokio |
| **Language** | Python 3.12 | Rust 2021 Edition |
| **Python Integration** | Native | PyO3 bindings |
| **Async Runtime** | asyncio | Tokio |

## Quick Commands

```bash
# Build
make build                    # or: docker build -t slimgest .

# Start servers
make up                       # or: docker compose up -d

# Check health
make health                   # or: curl localhost:7670/ && curl localhost:7671/

# Test single PDF
make test PDF=file.pdf       # or: python examples/test_servers.py file.pdf

# Benchmark
make benchmark DIR=pdfs/     # or: python examples/benchmark_servers.py pdfs/

# Stop servers
make down                     # or: docker compose down

# View logs
make logs                     # or: docker compose logs -f
```

## API Endpoints (Both Servers)

### Health Check
```bash
curl http://localhost:7670/    # Python
curl http://localhost:7671/    # Rust
```

### Process Single PDF
```bash
curl -X POST http://localhost:7670/process-pdf \
  -F "file=@document.pdf" \
  -F "dpi=150"
```

### Process Multiple PDFs
```bash
curl -X POST http://localhost:7670/process-pdfs \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.pdf" \
  -F "dpi=150"
```

### Stream Processing
```bash
curl -X POST http://localhost:7670/process-pdf-stream \
  -F "file=@document.pdf" \
  -F "dpi=150"
```

## File Locations

### Source Code
- Python: `src/slimgest/web/__main__.py`
- Rust: `web_rust/src/`

### Documentation
- Benchmarking Guide: `BENCHMARKING.md`
- Implementation Summary: `IMPLEMENTATION_SUMMARY.md`
- Verification Checklist: `VERIFICATION_CHECKLIST.md`
- Rust README: `web_rust/README.md`

### Testing Scripts
- Comprehensive Benchmark: `examples/benchmark_servers.py`
- Quick Test: `examples/test_servers.py`
- Shell Helper: `scripts/benchmark.sh`

### Configuration
- Docker: `Dockerfile`
- Compose: `docker-compose.yml`
- Build: `Makefile`

## Environment Variables

```bash
# Required
NEMOTRON_OCR_MODEL_DIR=/app/models/nemotron-ocr-v1/checkpoints

# Optional
CUDA_VISIBLE_DEVICES=0        # GPU selection
PYTHONPATH=/app/src           # Python import paths
```

## Typical Workflow

```bash
# 1. Build
make build

# 2. Start servers
make up

# 3. Wait for startup (30-60 seconds)
sleep 30

# 4. Verify health
make health

# 5. Quick test
make test PDF=test.pdf

# 6. Full benchmark
make benchmark DIR=test_pdfs/

# 7. Review results
cat benchmark_results.json | python -m json.tool

# 8. Stop servers
make down
```

## Performance Expectations

### What Rust Improves
- HTTP request parsing: ~10-30% faster
- Multipart handling: ~15-25% faster  
- JSON serialization: ~20-40% faster
- Memory efficiency: ~10-20% better

### What Stays the Same
- ML model inference: Identical (same Python code)
- GPU utilization: Identical
- OCR accuracy: Identical

### Overall Expected Improvement
- Small PDFs (1-3 pages): **5-10%** faster
- Medium PDFs (10-20 pages): **3-6%** faster
- Large PDFs (50+ pages): **1-3%** faster

*Lower improvement on larger PDFs because ML time dominates.*

## Troubleshooting

### Servers won't start
```bash
docker logs slimgest-benchmark
# Check for: model loading errors, port conflicts, CUDA issues
```

### Benchmark fails
```bash
# Verify servers are running
curl localhost:7670/ && curl localhost:7671/

# Check Python dependencies
docker exec slimgest-benchmark pip list

# Test with single file
python examples/test_servers.py small.pdf
```

### Out of memory
```bash
# Reduce concurrency
python examples/benchmark_servers.py pdfs/ --concurrent 1

# Process fewer files
python examples/benchmark_servers.py pdfs/ --max-files 5
```

### Compilation errors
```bash
# Check Rust version
docker run slimgest-benchmark rustc --version

# Rebuild from scratch
docker build --no-cache -t slimgest .
```

## Useful Docker Commands

```bash
# Shell into container
docker exec -it slimgest-benchmark bash

# Check processes
docker exec slimgest-benchmark ps aux | grep -E 'python|slimgest'

# Check GPU
docker exec slimgest-benchmark nvidia-smi

# Restart container
docker restart slimgest-benchmark

# Remove and rebuild
docker rm -f slimgest-benchmark
docker rmi slimgest
make build && make up
```

## Key Files to Customize

### Change Ports
Edit `docker-compose.yml`:
```yaml
ports:
  - "7670:7670"  # Change host port
  - "7671:7671"
```

### Adjust Benchmark Parameters
Edit `examples/benchmark_servers.py`:
- Line ~260: `timeout=300.0` - Request timeout
- Default DPI, concurrency, etc. in argparse

### Modify Rust Server
Edit `web_rust/src/handlers.rs`:
- Add custom endpoints
- Modify error handling
- Add logging/metrics

## Getting Help

1. **Read Documentation**
   - `BENCHMARKING.md` - Comprehensive guide
   - `VERIFICATION_CHECKLIST.md` - Testing guide
   - `web_rust/README.md` - Rust details

2. **Check Logs**
   ```bash
   make logs
   ```

3. **Verify Setup**
   ```bash
   make health
   python examples/test_servers.py test.pdf
   ```

4. **Review Implementation**
   - `IMPLEMENTATION_SUMMARY.md` - Complete overview
