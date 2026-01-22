from __future__ import annotations

import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import typer
import torch

import llama_nemotron_embed_1b_v2

try:
    from rich.console import Console

    _console: Console | None = Console()
except Exception:
    _console = None


@dataclass(frozen=True)
class LoadedEmbedding:
    path: Path
    vector: torch.Tensor  # shape: [dim]


def _infer_query_field(fieldnames: Sequence[str] | None) -> Optional[str]:
    if not fieldnames:
        return None
    candidates = (
        "text_query",
        "query",
        "question",
        "prompt",
        "q",
        "text",
    )
    lowered = {f.lower(): f for f in fieldnames}
    for c in candidates:
        if c in lowered:
            return lowered[c]
    # Fall back to first column
    return fieldnames[0] if fieldnames else None


def _coerce_embedding_to_vector(obj: Any) -> torch.Tensor:
    """
    Try hard to turn a torch-saved embedding output into a single 1D vector.

    Supports:
    - torch.Tensor
    - HF-style outputs with .pooler_output or .last_hidden_state
    - dict outputs with common keys
    - tuple/list outputs (uses first element)
    """
    if isinstance(obj, torch.Tensor):
        t = obj
    elif isinstance(obj, dict):
        d: Dict[str, Any] = obj
        for k in ("pooler_output", "sentence_embedding", "embeddings", "embedding"):
            if k in d:
                return _coerce_embedding_to_vector(d[k])
        for k in ("last_hidden_state", "hidden_states"):
            if k in d:
                return _coerce_embedding_to_vector(d[k])
        raise ValueError(f"Unrecognized embedding dict keys: {list(d.keys())[:10]}")
    elif hasattr(obj, "pooler_output"):
        return _coerce_embedding_to_vector(getattr(obj, "pooler_output"))
    elif hasattr(obj, "last_hidden_state"):
        return _coerce_embedding_to_vector(getattr(obj, "last_hidden_state"))
    elif isinstance(obj, (tuple, list)) and len(obj) > 0:
        return _coerce_embedding_to_vector(obj[0])
    else:
        raise TypeError(f"Unsupported embedding object type: {type(obj)}")

    # tensor handling
    if t.ndim == 1:
        return t
    if t.ndim == 2:
        # [batch, dim] -> take first row
        return t[0]
    if t.ndim == 3:
        # [batch, seq, dim] -> mean pool over seq, take first batch
        return t[0].mean(dim=0)
    raise ValueError(f"Unsupported embedding tensor shape: {tuple(t.shape)}")


def _normalize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    v = v.float()
    return v / (v.norm(p=2) + eps)


def _torch_load_embedding_file(path: Path) -> Any:
    """
    Load a torch-saved embedding result robustly across PyTorch versions.

    PyTorch 2.6+ changed `torch.load()` default `weights_only` from False -> True.
    Our embedding files are saved from a Transformers model output object, so they
    require a full (trusted) unpickle to load.
    """
    try:
        # Prefer safe loading when possible.
        return torch.load(path, map_location="cpu", weights_only=True)  # type: ignore[arg-type]
    except TypeError:
        # Older torch versions don't accept weights_only.
        return torch.load(path, map_location="cpu")
    except pickle.UnpicklingError:
        # Fall back to full unpickle (trusted local artifacts).
        return torch.load(path, map_location="cpu", weights_only=False)  # type: ignore[arg-type]


def _load_embeddings(embeddings_dir: Path) -> List[LoadedEmbedding]:
    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Embeddings dir does not exist: {embeddings_dir}")
    if not embeddings_dir.is_dir():
        raise NotADirectoryError(f"Embeddings path is not a directory: {embeddings_dir}")

    pt_files = sorted(embeddings_dir.rglob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found under: {embeddings_dir}")

    loaded: List[LoadedEmbedding] = []
    for p in pt_files:
        try:
            obj = _torch_load_embedding_file(p)
            vec = _coerce_embedding_to_vector(obj).detach().cpu()
            vec = _normalize(vec)
            loaded.append(LoadedEmbedding(path=p, vector=vec))
        except Exception as e:
            raise RuntimeError(f"Failed loading embedding file: {p}") from e

    # sanity: all same dim
    dims = {int(x.vector.numel()) for x in loaded}
    if len(dims) != 1:
        raise ValueError(f"Loaded embeddings with inconsistent dims: {sorted(dims)}")
    return loaded


def _pdf_stem_from_embedding_path(p: Path) -> str:
    """
    Embedding files are saved as: <pdf_stem>.embedding_results.pt
    Return <pdf_stem>.
    """
    name = p.name
    suffix = ".embedding_results.pt"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    # fallback: best effort
    return p.stem


def _mark_hit(hit: bool) -> str:
    if _console is None:
        return "✓" if hit else "✗"
    return "[green]✓[/green]" if hit else "[red]✗[/red]"


def _print(line: str) -> None:
    if _console is None:
        print(line)
    else:
        _console.print(line)


def _embed_query(
    query: str,
    tokenizer,
    model,
    device: torch.device,
) -> torch.Tensor:
    batch = tokenizer(query, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(**batch)
    vec = _coerce_embedding_to_vector(out).detach()
    vec = vec.to("cpu")
    return _normalize(vec)


def _topk_cosine(
    query_vec: torch.Tensor, doc_matrix: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    # All vectors are assumed normalized, so cosine = dot product.
    scores = doc_matrix @ query_vec  # [N]
    k = min(k, int(scores.numel()))
    top_scores, top_idx = torch.topk(scores, k=k, largest=True)
    return top_scores, top_idx


def topk(
    embeddings_dir: Path = typer.Option(
        Path("/home/jdyer/datasets/slimgest-raw-results/bo767"),
        help="Directory containing saved .pt embedding files (searched recursively).",
    ),
    queries_csv: Path = typer.Option(
        Path(
            "/home/jdyer/Development/slim-gest/private/recall_data/text_query_answer_gt_page.csv"
        ),
        help="CSV file containing text queries (one per row).",
    ),
    top_k: int = typer.Option(5, "--topk", min=1, help="Number of results to show per query."),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the query embedding model on (cuda or cpu).",
    ),
    query_column: Optional[str] = typer.Option(
        None,
        help="CSV column name containing the query text. If omitted, inferred from headers.",
    ),
    limit: Optional[int] = typer.Option(
        None, help="Optionally limit number of CSV rows processed."
    ),
):
    """
    Load all .pt embeddings in a directory, then for each query in a CSV file,
    embed the query and print the top-k nearest embeddings by cosine similarity.
    """
    # Load embeddings once
    embeddings = _load_embeddings(embeddings_dir)
    doc_matrix = torch.stack([e.vector for e in embeddings], dim=0)  # [N, dim]

    # Load embedding model once
    dev = torch.device(device)
    tokenizer = llama_nemotron_embed_1b_v2.load_tokenizer()
    model = llama_nemotron_embed_1b_v2.load_model(device=str(dev))
    model.eval()

    if not queries_csv.exists():
        raise FileNotFoundError(f"Queries CSV does not exist: {queries_csv}")

    with queries_csv.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        q_field = query_column or _infer_query_field(reader.fieldnames)
        if not q_field:
            raise ValueError("Could not infer query column (CSV appears to have no headers).")

        for i, row in enumerate(reader, start=1):
            if limit is not None and i > limit:
                break

            query = (row.get(q_field) or "").strip()
            if not query:
                continue

            expected_pdf = (row.get("pdf") or "").strip()
            expected_gt_page = (row.get("gt_page") or "").strip()
            expected_pdf_stem = Path(expected_pdf).stem if expected_pdf else ""

            q_vec = _embed_query(query, tokenizer=tokenizer, model=model, device=dev)
            top_scores, top_idx = _topk_cosine(q_vec, doc_matrix, k=top_k)

            _print(f"\n=== Query {i} ({q_field}) ===")
            _print(query)
            if expected_pdf or expected_gt_page:
                _print(f"Expected: pdf={expected_pdf}  gt_page={expected_gt_page}")

            for rank, (score, idx) in enumerate(zip(top_scores.tolist(), top_idx.tolist()), start=1):
                p = embeddings[int(idx)].path
                pdf_stem = _pdf_stem_from_embedding_path(p)
                hit = bool(expected_pdf_stem) and (pdf_stem == expected_pdf_stem)
                _print(f"{rank:>2}. {_mark_hit(hit)} {float(score):.6f}  {p}")

