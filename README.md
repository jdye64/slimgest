slimgest
========

PDF-to-image OCR pipeline with a FastAPI server and CLIs, using a shared core for identical behavior across modes. Rendering via PyMuPDF, preprocessing via Pillow, OCR via vLLM (DeepSeek OCR).

Install
-------
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

- download the vllm-0.8.5 [whl](https://github.com/vllm-project/vllm/releases/tag/v0.8.5)
```Shell
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
pip install flash-attn==2.7.3 --no-build-isolation
```

Run the server
--------------
```bash
slimgest-server --host 0.0.0.0 --port 8000 --vllm-url http://localhost:8001
```

Local processing
----------------
```bash
slimgest-local /path/to/input_pdfs /path/to/output_dir --dpi 220
```

REST client
-----------
```bash
# Submit and follow a job
slimgest-client submit /path/to/file.pdf --server http://localhost:8000
slimgest-client follow <job_id> --server http://localhost:8000
```

Notes
-----
- Requires Python 3.10+
- Requires a running vLLM server that serves a DeepSeek OCR-like multimodal chat API at `/v1/chat/completions`.
- Metrics are collected per phase (render, preprocess, ocr) and shown in the REST client summary and saved in local mode outputs.


# slimgest
