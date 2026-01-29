from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.traceback import install
import torch
import typer
from tqdm import tqdm

from ._io import coerce_embedding_to_vector, iter_images, normalize_l2, read_json
# import slimgest.model.local.llama_nemotron_embed_1b_v2_embedder as llama_nemotron_embed_1b_v2
import numpy as np
# from slimgest.model.local.llama_nemotron_embed_1b_v2_embedder import LlamaNemotronEmbed1BV2Embedder
import llama_nemotron_embed_1b_v2
import torch.nn.functional as F

install(show_locals=False)
console = Console()
app = typer.Typer(
    help="Stage 6: load OCR JSON from stage5, embed each OCR region, and save a torch-loadable .pt alongside each image."
)

DEFAULT_INPUT_DIR = Path("./data/pages")

def average_pool(last_hidden_states, attention_mask):
    """Average pooling with attention mask."""
    last_hidden_states_masked = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    embedding = last_hidden_states_masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    embedding = F.normalize(embedding, dim=-1)
    return embedding

def _stage5_json_for_image(img_path: Path) -> Path:
    return img_path.with_name(img_path.name + ".nemotron_ocr_v1.json")

def _pdfium_text_for_image(img_path: Path) -> Path:
    """
    pdf.convert writes embedded PDF text as:
      <pdf_basename>_page<NNNN>.pdfium_text.txt

    For an image like:
      <pdf_basename>_page<NNNN>.png

    ...this corresponds to `img_path.with_suffix(".pdfium_text.txt")`.
    """
    return img_path.with_suffix(".pdfium_text.txt")


def _read_text_file_best_effort(path: Path) -> str:
    try:
        s = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    # Keep internal newlines intact; just trim surrounding whitespace.
    return s.strip()


def _out_path_for_image(img_path: Path) -> Path:
    return img_path.with_name(img_path.name + ".embeddings.pt")


@app.command()
def run(
    input_dir: Path = typer.Option(DEFAULT_INPUT_DIR, "--input-dir", exists=True, file_okay=False),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", help="Device for embedding model."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing .pt outputs."),
    limit: Optional[int] = typer.Option(None, help="Optionally limit number of images processed."),
    embedding_endpoint: Optional[str] = typer.Option(
        None,
        help="Optional embedding endpoint URL (often OpenAI-compatible '/v1/embeddings'). If set, embeddings run remotely and the local embedding model is not loaded.",
    ),
    embedding_model_name: Optional[str] = typer.Option(
        None,
        help="Optional embedding model name for remote endpoint (sent as 'model' in payload).",
    ),
    batch_size: int = typer.Option(64, "--batch-size", min=1, help="Embedding batch size (texts per request)."),
):
    dev = torch.device(device)
    # Use the shared embedder wrapper; if endpoint is set, it runs remotely.
    embed_model = llama_nemotron_embed_1b_v2.load_model('cuda:0', True, None, True)
    tokenizer = llama_nemotron_embed_1b_v2.load_tokenizer("longest_first")

    images = iter_images(input_dir)
    if limit is not None:
        images = images[: int(limit)]

    console.print(f"[bold cyan]Stage6[/bold cyan] images={len(images)} input_dir={input_dir} device={dev}")

    chunks_created = 0
    processed = 0
    skipped = 0
    missing_stage5 = 0
    bad_stage5 = 0
    missing_pdfium_text = 0
    texts_added = 0
    ocr_added = 0
    for img_path in tqdm(images, desc=f"Stage6 images", unit="img"):
        out_path = _out_path_for_image(img_path)
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        s5_path = _stage5_json_for_image(img_path)
        if not s5_path.exists():
            missing_stage5 += 1
            continue

        pdfium_text_path = _pdfium_text_for_image(img_path)
        pdfium_text = ""
        if pdfium_text_path.exists():
            pdfium_text = _read_text_file_best_effort(pdfium_text_path)
        else:
            missing_pdfium_text += 1

        try:
            s5 = read_json(s5_path)
        except Exception:
            bad_stage5 += 1
            continue
        regions: List[Dict[str, Any]] = list(s5.get("regions") or [])
        texts: List[str] = []
        bboxes: List[List[float]] = []
        text_kinds: List[str] = []

        # Include embedded PDF text (if present) as a full-page "region" so it is embedded
        # alongside OCR detection text.
        if pdfium_text:
            texts.append(pdfium_text)
            bboxes.append([0.0, 0.0, 1.0, 1.0])
            text_kinds.append("pdfium_page_text")
            texts_added += 1


        for r in regions:
            txt = (r.get("ocr_text") or "").strip()
            bbox = r.get("bbox_xyxy_norm_in_page")
            if txt:
                texts.append(txt)
                if isinstance(bbox, list) and len(bbox) == 4:
                    bboxes.append([float(x) for x in bbox])
                else:
                    bboxes.append([0.0, 0.0, 0.0, 0.0])
                text_kinds.append("ocr_region")
                ocr_added += 1

        if len(texts) ==0:
            skipped += 1
        t0 = time.perf_counter()
        vectors: List[torch.Tensor] = []
        if texts:
            with torch.inference_mode():
                texts = ["passage: " + text for text in texts]
                tokenized_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(dev)
                embed_model_results = embed_model(**tokenized_inputs)
                emb = average_pool(embed_model_results.last_hidden_state, tokenized_inputs['attention_mask'])
                # emb = embedder.embed(texts, batch_size=batch_size)  # [N, D] on CPU

                for i in range(int(emb.shape[0])):
                    # vec = normalize_l2(emb[i])
                    vectors.append(emb[i].cpu().numpy().astype(np.float32))
        dt = time.perf_counter() - t0
        chunks_created += len(texts)

        torch.save(
            {
                "schema_version": 1,
                "stage": 6,
                "model": "llama_nemotron_embed_1b_v2",
                "image_path": str(img_path),
                "stage5_json": str(s5_path),
                "pdfium_text_path": str(pdfium_text_path),
                "texts": texts,
                "text_kinds": text_kinds,
                "bboxes_xyxy_norm_in_page": bboxes,
                "embeddings": vectors,
                "timing": {"seconds": float(dt)},
            },
            out_path,
        )
        processed += 1

    console.print(
        f"[green]Done[/green] processed={processed} skipped={skipped} missing_stage5={missing_stage5} bad_stage5={bad_stage5} "
        f"missing_pdfium_text={missing_pdfium_text} chunks_created={chunks_created} texts_added={texts_added} ocr_added={ocr_added} wrote_pt_suffix=.embeddings.pt"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()

