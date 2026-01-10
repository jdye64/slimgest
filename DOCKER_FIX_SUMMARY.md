# Docker Unicode & Body Limit Fix

## Issues Identified

1. **UnicodeEncodeError**: Python subprocess couldn't encode non-ASCII characters (e.g., 'ä')
2. **Multipart Length Limit Exceeded**: Default 2MB body limit was too small for PDF uploads and responses

## Root Causes

### Issue 1: Missing Python UTF-8 Environment Variables
The Dockerfile didn't set `PYTHONIOENCODING=utf-8`, causing Python to default to ASCII encoding when handling I/O, which failed when OCR extracted non-ASCII characters.

### Issue 2: No Body Size Limit Configuration
Axum's router had no `DefaultBodyLimit` layer, defaulting to 2MB which is insufficient for:
- Large PDF uploads
- OCR response data with base64-encoded images and extracted text

## Fixes Applied

### 1. Dockerfile - Added UTF-8 Environment Variables
```dockerfile
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
```

### 2. main.rs - Added 500MB Body Limit
```rust
use axum::extract::DefaultBodyLimit;

let body_limit = 500 * 1024 * 1024; // 500MB
let app = Router::new()
    // ... routes ...
    .layer(DefaultBodyLimit::max(body_limit))
    .layer(CorsLayer::permissive())
    .with_state(models);
```

### 3. Added Debug Instrumentation
- handlers.rs: Log multipart data size and response size
- python_bridge.rs: Log Python encoding environment variables
- main.rs: Log configured body limit

## Testing Instructions

1. Rebuild the Docker image:
   ```bash
   docker build -t slim-gest:latest .
   ```

2. Run the container:
   ```bash
   docker run --gpus all -p 7671:7671 -v $(pwd)/models:/app/models slim-gest:latest
   ```

3. Test with the problematic PDF that previously failed

## Expected Results

- No UnicodeEncodeError (Python now handles UTF-8 properly)
- No "length limit exceeded" errors (500MB limit sufficient for large PDFs)
- Successful processing of PDFs with non-ASCII characters in OCR output

## Instrumentation (Temporary)

Debug logs added to verify fixes:
- Python environment encoding settings check
- Multipart request size logging
- Response JSON size logging
- Body limit configuration logging

Remove instrumentation after verification.
