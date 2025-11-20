# Two-Phase Processing Implementation

## Overview

The PDF processing pipeline has been redesigned to improve throughput and GPU utilization by separating page extraction from page processing, and distributing work more evenly across workers.

## Changes

### Previous Architecture (PDF-level parallelism)
- Each worker processed an entire PDF from start to finish
- Problem: Uneven workload distribution (PDFs with different page counts)
- Problem: GPU underutilization (sequential processing within each PDF)

### New Architecture (Page-level parallelism)

#### Phase 1: Page Extraction
1. Extract all pages from all PDFs **first** (before any processing)
2. All pages saved to `scratch/pdf_pages/` directory
3. Creates a list of all pages with metadata

#### Phase 2: Distributed Page Processing
1. All extracted pages are submitted to worker pool at once
2. Workers process individual pages (not full PDFs)
3. Even distribution: all workers get roughly equal number of pages
4. Better GPU utilization: GPU processes pages continuously

## New Functions

### `extract_all_pages_from_pdfs(pdf_files, scratch_dict)`
- Extracts all pages from all input PDFs
- Returns: `(page_list, extraction_metrics)`
- `page_list`: List of dicts with page metadata
- `extraction_metrics`: Statistics about extraction phase

### `process_single_page(page_info, scratch_dict)`
- Processes a single PDF page
- Returns: `(page_metrics, counters)`
- Runs all ML models on the page (page elements, tables, graphics)
- Worker processes load models once and process many pages

## Benefits

1. **Even Load Distribution**: All workers finish at roughly the same time
2. **Higher GPU Utilization**: GPU processes pages continuously without gaps
3. **Better Parallelism**: Work distributed at page level instead of PDF level
4. **Clearer Progress**: See page-by-page progress instead of PDF-by-PDF

## Metrics

The new implementation tracks:
- Phase 1 timing (page extraction)
- Phase 2 timing (page processing)
- Total pages extracted vs processed
- All previous metrics maintained (tables, graphics, OCR, etc.)

## Compatibility

- All existing output formats preserved
- Metrics still aggregated by PDF
- Individual page results saved in same locations
- Configuration (`parallel_workers`) works the same way

