# Quick Start Guide - Batch Processing Client

## 1. Start the Server

```bash
# Start the FastAPI server (GPU required)
python -m slimgest.web --host 0.0.0.0 --port 7670 --workers 1
```

Wait for the models to load (you'll see "Models loaded" messages).

## 2. Process PDFs

### Single PDF
```bash
python src/slimgest/web/test_client.py document.pdf --output-dir ./output
```

### Directory of PDFs
```bash
python src/slimgest/web/test_client.py ./my_pdfs/ --output-dir ./output --workers 4
```

### High Performance Setup
```bash
python src/slimgest/web/test_client.py ./pdfs/ \
  --output-dir ./output \
  --batch-size 64 \
  --workers 8 \
  --dpi 150
```

## 3. View Results

After processing completes, you'll find:

- **Markdown files**: One per PDF in `./output/`
- **Performance chart**: `./output/performance_chart.png`
- **Terminal output**: Detailed statistics and top 100 slowest PDFs

## What You'll See

### Real-Time Progress
```
Processing PDFs... ━━━━━━━━━━━━━━━━━━━━━━━━━ 45/100 • 0:02:30 • 0:01:15
✓ document1.pdf | Pages:   25 | Time:  12.45s | Current: 8.5 pages/s
✓ document2.pdf | Pages:   18 | Time:   9.23s | Current: 9.2 pages/s
```

### Final Report

**Overall Statistics:**
| Metric | Value |
|--------|-------|
| Total PDFs Processed | 100 / 100 |
| Total Pages Processed | 2,450 |
| Total Bytes Read | 245,678,901 |
| Total Batches Sent | 77 |
| Total Time | 180.45s |
| Average Pages/Second | 13.58 |

**Top 100 Slowest PDFs:**
Shows which documents took the longest to process, with size and page count.

**Performance Chart:**
Graphs showing pages/second over time with statistics.

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir` | `./output` | Where to save markdown files |
| `--dpi` | `150.0` | PDF rendering quality |
| `--batch-size` | `32` | Pages per batch |
| `--workers` | `4` | Concurrent PDFs |
| `--url` | `http://localhost:7670` | Server address |

## Tips

1. **Start with defaults** - They work well for most cases
2. **Monitor pages/second** - Use it to tune batch-size and workers
3. **Check the chart** - Identify performance bottlenecks
4. **Review slowest PDFs** - Find problematic documents

## Troubleshooting

- **Health check failed**: Server not running
- **Timeout errors**: Reduce batch-size or increase workers
- **Out of memory**: Lower DPI or batch-size

## Performance Tuning

For **faster processing**:
- Increase `--workers` (if server has capacity)
- Increase `--batch-size` (if network is good)
- Lower `--dpi` (if quality allows)

For **better quality**:
- Increase `--dpi` to 200-300
- Keep batch-size moderate (32)

For **stability**:
- Use default values
- Start with `--workers 1` for debugging
