from __future__ import annotations

import time
from pathlib import Path

from rich.console import Console
from rich.traceback import install
import torch
import typer
from tqdm import tqdm
import glob
import pandas as pd
import os
import torch.nn.functional as F


from slimgest.model.local.llama_nemotron_embed_1b_v2_embedder import LlamaNemotronEmbed1BV2Embedder
from slimgest.local.vdb.lancedb import LanceDB
import lancedb


install(show_locals=False)
console = Console()
app = typer.Typer(
    help="Stage 7: load results from stage6, upload embeddings to VDB"
)

DEFAULT_INPUT_DIR = Path("./data/pages")


def calculate_recall(real_answers, retrieved_answers, k):
    hits = 0
    for real, retrieved in zip(real_answers, retrieved_answers):
        if real in retrieved[:k]:
            hits += 1
    return hits / len(real_answers)


def calcuate_recall_list(real_answers, retrieved_answers, ks=[1, 3, 5, 10]):
    recall_scores = {}
    for k in ks:
        recall_scores[k] = calculate_recall(real_answers, retrieved_answers, k)
    return recall_scores

def get_correct_answers(query_df):
    retrieved_pdf_pages = []
    for i in range(len(query_df)):
        retrieved_pdf_pages.append(query_df['pdf_page'][i]) 
    return retrieved_pdf_pages

def create_lancedb_results(results):
    old_results = [res["metadata"] for result in results for res in result]
    results = []
    for result in old_results:
        if result["embedding"] is None:
            continue
        results.append({
            "vector": result["embedding"], 
            "text": result["content"], 
            "metadata": result["content_metadata"]["page_number"], 
            "source": result["source_metadata"]["source_id"],
        })
    return results

def format_retrieved_answers_lance(all_answers):
    retrieved_pdf_pages = []
    for answers in all_answers:
        retrieved_pdfs = [os.path.basename(result['source']).split('.')[0] for result in answers]
        retrieved_pages = [str(result['metadata']) for result in answers]
        retrieved_pdf_pages.append([f"{pdf}_{page}" for pdf, page in zip(retrieved_pdfs, retrieved_pages)])
    return retrieved_pdf_pages



def average_pool(last_hidden_states, attention_mask):
    """Average pooling with attention mask."""
    last_hidden_states_masked = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    embedding = last_hidden_states_masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    embedding = F.normalize(embedding, dim=-1)
    return embedding



@app.command()
def run(
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", help="Device for embedding model."),
    query_file: Path = typer.Option(..., "--query-file", exists=True, file_okay=True, dir_okay=False, help="File with one query per line."),

):
    processed = 0
    skipped = 0
    embed_model = LlamaNemotronEmbed1BV2Embedder(normalize=True)
    db = lancedb.connect(uri="lancedb")
    table = db.open_table("nv-ingest")

    breakpoint()
    # Use the shared embedder wrapper; if endpoint is set, it runs remotely.
    df_query = pd.read_csv(query_file).rename(columns={'gt_page':'page'})[['query','pdf','page']]
    df_query['pdf_page'] = df_query.apply(lambda x: f"{x.pdf}_{x.page}", axis=1) 

    console.print(f"[bold cyan]Stage8[/bold cyan] images={df_query.shape[0]} input_dir={query_file} device={device}")


    all_answers = []
    query_texts = df_query['query'].tolist()
    top_k = 10
    result_fields = ["source", "metadata", "text"]
    query_embeddings = []
    for query_batch in tqdm(query_texts, desc="Stage8 query embeddings", unit="queries"):
        query_embeddings += embed_model.embed(["query: " + query_batch])
        # query_embeddings += average_pool(query_model_results.last_hidden_state, query_tokens['attention_mask'])
    for query_embed in tqdm(query_embeddings, desc="Stage8 querying VDB", unit="queries"): 
        all_answers.append(
            table.search([query_embed.detach().cpu().numpy()], vector_column_name="vector").select(result_fields).limit(top_k).to_list()
        )
        processed += 1
    retrieved_pdf_pages = format_retrieved_answers_lance(all_answers)
    golden_answers = get_correct_answers(df_query)
    recall_scores = calcuate_recall_list(golden_answers, retrieved_pdf_pages, ks=[1, 3, 5, 10])

    console.print(
        f"[green]Done[/green] processed={processed} skipped={skipped} "
        f"recall@1={recall_scores[1]:.4f} recall@3={recall_scores[3]:.4f} "
        f"recall@5={recall_scores[5]:.4f} recall@10={recall_scores[10]:.4f} "
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()

