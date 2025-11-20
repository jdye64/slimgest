# Implementation Summary: Two-Phase Processing

## What Changed

### Old Approach (PDF-level parallelism)
```
Worker 1: Process PDF A (10 pages) → [split → convert → process each page sequentially]
Worker 2: Process PDF B (5 pages)  → [split → convert → process each page sequentially]
Worker 3: Process PDF C (20 pages) → [split → convert → process each page sequentially]
...

Problem: Worker 2 finishes early and sits idle while Worker 3 is still processing
Problem: Within each worker, GPU processes pages sequentially (gaps between pages)
```

### New Approach (Page-level parallelism)
```
Phase 1: Extract ALL pages first
- Split PDF A → 10 pages
- Split PDF B → 5 pages
- Split PDF C → 20 pages
Total: 35 pages ready in scratch directory

Phase 2: Distribute pages evenly
Worker 1: Process pages [1, 4, 7, 10, 13, 16, 19, ...]
Worker 2: Process pages [2, 5, 8, 11, 14, 17, 20, ...]
Worker 3: Process pages [3, 6, 9, 12, 15, 18, 21, ...]
...

Benefit: All workers get ~12 pages each, finish at same time
Benefit: GPU hammered continuously with page processing
```

## Key Improvements

### 1. Even Load Distribution
- **Before**: If you had PDFs with 5, 10, and 50 pages, one worker would take 10x longer
- **After**: All pages distributed evenly, all workers finish around the same time

### 2. GPU Utilization
- **Before**: GPU idle between PDF transitions within each worker
- **After**: GPU processes pages continuously, much higher utilization

### 3. Scalability
- **Before**: Limited by number of PDFs (10 PDFs with 100 workers = 90 idle workers)
- **After**: Limited by number of pages (1000 pages works great with 100 workers)

## Code Changes

### New Functions
1. **`extract_all_pages_from_pdfs()`** - Phase 1 implementation
   - Takes all PDF files
   - Splits them into individual page PDFs
   - Returns list of all pages with metadata

2. **`process_single_page()`** - Phase 2 worker function
   - Takes a single page metadata dict
   - Runs all ML models on that page
   - Returns page metrics and counters

### Modified Function
3. **`process()`** - Main entry point
   - Now runs in two distinct phases
   - Phase 1: Extract all pages (serial or parallel at PDF level)
   - Phase 2: Process pages (parallel at page level)
   - Aggregates page results back into per-PDF metrics

### Preserved Function
4. **`process_single_pdf()`** - Still exists for backwards compatibility
   - Not used by new implementation
   - Could be used for single-PDF processing or testing

## Testing Recommendations

1. **Small test** (2-3 PDFs with different page counts):
   ```bash
   python -m slimgest.cli.local process input/ scratch/
   ```
   - Watch for Phase 1 and Phase 2 messages
   - Verify all pages extracted before processing starts
   - Check progress bar shows page-by-page progress

2. **Large test** (many PDFs):
   - Monitor GPU utilization during Phase 2
   - Should see consistently high GPU usage
   - All workers should finish around the same time

3. **Verify outputs**:
   - Check `scratch/metrics/` for per-PDF JSON files
   - Verify page results are aggregated correctly by PDF
   - Compare output format with previous runs (should be identical)

## Performance Expectations

For a workload with:
- 10 PDFs
- Average 20 pages per PDF
- 200 total pages
- 10 workers

**Before**: 
- Some workers finish in 2 minutes (small PDFs)
- Some workers take 10 minutes (large PDFs)
- Average GPU utilization: ~60%

**After**:
- All workers get ~20 pages each
- All workers finish around same time (~4-5 minutes)
- GPU utilization: ~90%+

## Monitoring

The process monitor will show:
- Phase 1: CPU-bound (PDF extraction), moderate memory
- Phase 2: GPU-bound (page processing), high GPU utilization
- More consistent resource usage throughout Phase 2

