from pathlib import Path
from typing import List, Tuple, Optional
from rich.console import Console
from rich.traceback import install
from torch import nn
import torch
import time
import json
import io
from PIL import Image
import numpy as np
import torch.nn.functional as F

from slimgest.local.vdb.lancedb import LanceDB

from nemotron_page_elements_v3.model import define_model as define_model_page_elements
from nemotron_page_elements_v3.model import resize_pad as resize_pad_page_elements
from nemotron_page_elements_v3.utils import postprocess_preds_page_element as postprocess_preds_page_element
from nemotron_table_structure_v1.model import define_model as define_model_table_structure
from nemotron_table_structure_v1.model import resize_pad as resize_pad_table_structure
from nemotron_graphic_elements_v1.model import define_model as define_model_graphic_elements
from nemotron_graphic_elements_v1.model import resize_pad as resize_pad_graphic_elements

from nemotron_ocr.inference.pipeline import NemotronOCR
import llama_nemotron_embed_1b_v2

import typer

# Import our new PDF processing utilities
from slimgest.pdf.render import iter_pdf_page_tensors
from slimgest.pdf.tensor_ops import crop_tensor_with_bbox
import os

app = typer.Typer(help="Simpliest pipeline with limited CPU parallelism while using maximum GPU possible")
install(show_locals=False)
console = Console()



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


def tensor_to_pil_image(tensor):
    """
    Converts a 3xHxW torch tensor [0,1] or [0,255] on cpu/gpu to PIL Image (RGB).
    Assumes tensor is [C,H,W] and in standard format, does NOT do normalization undoing.
    """
    if tensor.device != torch.device('cpu'):
        tensor = tensor.cpu()
    # Clamp and convert to uint8
    arr = tensor.detach().numpy()
    if arr.max() <= 1.0:
        arr = arr * 255.
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))  # HWC
    img = Image.fromarray(arr, mode='RGB')
    return img

def process_pdf_pages(
    pdf_path,
    page_elements_model,
    table_structure_model,
    graphic_element_model,
    ocr_model,
    embed_model,
    embed_tokenizer,
    device="cuda",
    dpi=150.0,
):
    """
    Generator that processes PDF pages one at a time, yielding results for each page.
    
    This is memory-efficient as it processes pages as they're rendered without loading
    all pages into memory at once.
    
    Yields:
        Tuple of (page_number, processed_tensor, ocr_results, raw_ocr_results)
    """
    page_elements_input_shape = (1024, 1024)
    table_structure_input_shape = (1024, 1024)
    graphic_elements_input_shape = (1024, 1024)
    result_rows = []
    
    # Use the new generator to iterate through PDF pages
    for page_tensor_info in iter_pdf_page_tensors(pdf_path, dpi=dpi, device=device):
        page_number = page_tensor_info.page_number
        tensor = page_tensor_info.tensor  # Shape: [3, H, W]
        bitmap_shape = (page_tensor_info.original_height, page_tensor_info.original_width)
        
        # Keep a reference to the original tensor for OCR
        original_tensor = tensor
        embeddings = []
        page_ocr_results = []
        page_raw_ocr_results = []
        
        with torch.inference_mode():
            # Resize for page elements detection
            resized_tensor = resize_pad_page_elements(tensor, page_elements_input_shape)
            preds = page_elements_model(resized_tensor, bitmap_shape)[0]
            boxes, labels, scores = postprocess_preds_page_element(
                preds, page_elements_model.thresholds_per_class, page_elements_model.labels
            )
            
            # Process detected elements (tables and graphics)
            for label, box in zip(labels, boxes):
                if label == 0:  # Table
                    cropped = crop_tensor_with_bbox(
                        resized_tensor, box, bitmap_shape, page_elements_input_shape
                    ).clone()
                    crop_shape = (cropped.shape[1], cropped.shape[2])
                    cropped_resized = resize_pad_table_structure(cropped, table_structure_input_shape)
                    table_preds = table_structure_model(cropped_resized, crop_shape)[0]
                    # print(f"Page {page_number} - Table structure results: {table_preds}")
                    
                elif label in [1, 2, 3]:  # Graphic elements
                    cropped = crop_tensor_with_bbox(
                        resized_tensor, box, bitmap_shape, page_elements_input_shape
                    ).clone()
                    crop_shape = (cropped.shape[1], cropped.shape[2])
                    cropped_resized = resize_pad_graphic_elements(cropped, graphic_elements_input_shape)
                    graphic_preds = graphic_element_model(cropped_resized, crop_shape)[0]
                    # print(f"Page {page_number} - Graphic elements results: {graphic_preds}")
            
            # Run OCR on the original (un-resized) tensor
            # Convert the tensor to a PIL image, then to a BytesIO JPEG for OCR model
            # pil_img = tensor_to_pil_image(original_tensor)
            # img_bytesio = io.BytesIO()
            # pil_img.save(img_bytesio, format="JPEG")
            # img_bytesio.seek(0)
            ocr_preds = ocr_model(tensor.clone().to(device="cuda"))

            for pred in ocr_preds:
                page_ocr_results.append(str(pred['text']))
                page_raw_ocr_results.append(str(pred))
            
            if len(page_ocr_results) > 0:
                tokenized_inputs = embed_tokenizer(page_ocr_results, return_tensors="pt", padding=True).to(device)
                embed_model_results = embed_model(**tokenized_inputs)
                # breakpoint()
                embeddings.append(average_pool(embed_model_results.last_hidden_state, tokenized_inputs['attention_mask']))

        result_rows.append({
            "pdf_path": pdf_path,
            "page_number": page_number,
            "resized_tensor": resized_tensor,
            "page_ocr_results": page_ocr_results,
            "page_raw_ocr_results": page_raw_ocr_results,
            "embeddings": embeddings
        })
    return result_rows

def run_pipeline(
    pdf_files: List[str],
    page_elements_model: nn.Module,
    table_structure_model: nn.Module,
    graphic_elements_model: nn.Module,
    ocr_model: NemotronOCR,
    embed_model: nn.Module,
    embed_tokenizer: nn.Module,
    vdb_op: None,
    raw_output_dir: Optional[Path] = None,
    dpi: float = 150.0,
    return_results: bool = False,
):
    """
    Process PDF files using a streaming pipeline that handles one page at a time.
    
    Args:
        pdf_files: List of PDF file paths as strings.
        page_elements_model, table_structure_model, graphic_elements_model, ocr_model: Models to use.
        raw_output_dir: Directory to save raw OCR results. If None, does not save.
        dpi: DPI for PDF rendering (default 150).
        return_results: If True, returns results as dict instead of just printing.
    
    Returns:
        If return_results is True, returns a dict with all results.
    """
    console.print(f"made ti to run pipeline")

    pdf_files = [pdf_files] if isinstance(pdf_files, str) else pdf_files
    start_time = time.time()
    total_pages_processed = 0
    results = []
    
    for pdf_idx, pdf_path in enumerate(pdf_files, start=1):
        console.print(f"[bold cyan]Processing:[/bold cyan] {pdf_path}")
        
        # Collect results for this PDF
        all_page_ocr_results = []
        all_page_raw_ocr_results = []
        pages_in_pdf = 0

        # Process pages one at a time using the generator
        for results_row in process_pdf_pages(
            pdf_path,
            page_elements_model,
            table_structure_model,
            graphic_elements_model,
            ocr_model,
            embed_model,
            embed_tokenizer,
            device="cuda",
            dpi=dpi,
        ):
            page_number = results_row["page_number"]
            page_ocr_results = results_row["page_ocr_results"]
            page_raw_ocr_results = results_row["page_raw_ocr_results"]
            embeddings = results_row["embeddings"]
            pages_in_pdf += 1
            total_pages_processed += 1
            
            # Collect OCR results
            all_page_ocr_results.extend(page_ocr_results)
            all_page_raw_ocr_results.extend(page_raw_ocr_results)
            # all_embeddings.append(list(embed) for embed in embeddings[0])
            # breakpoint()
            for embedding in embeddings:
                for text, ocr, embed in zip(page_ocr_results, page_raw_ocr_results, embedding):
                    results.append({
                        "source_id": pdf_path,
                        "page_number": page_number,
                        "content": text,
                        "raw_ocr_results": ocr,
                        "embedding": embed.cpu().numpy().astype("int8"),
                    })
            # # Show progress
            # console.print(
            #     f"  Processed page {page_number} | "
            #     f"Tensor shape: {list(tensor.shape)} | "
            #     f"Device: {tensor.device}"
            # )
        
        # # Summary for this PDF
        # console.print(
        #     f"Completed {pages_in_pdf} pages from {pdf_path}. "
        #     f"PDF {pdf_idx} of {len(pdf_files)}. "
        #     f"Total pages processed: {total_pages_processed}. "
        #     f"Current Runtime: {time.time() - start_time:.2f} seconds"
        # )
        
        # # Combine all OCR results for this PDF
        # ocr_final_result = " ".join(all_page_ocr_results)
        # console.print(f"OCR final result: {ocr_final_result}", markup=False)
        
        # Save raw OCR results if requested
        if raw_output_dir is not None:
            raw_output_dir = Path(raw_output_dir)
            raw_output_dir.mkdir(parents=True, exist_ok=True)
            pdf_path_obj = Path(pdf_path)
            output_json_path = raw_output_dir / pdf_path_obj.with_suffix('.page_raw_ocr_results.json').name
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(all_page_raw_ocr_results, f, ensure_ascii=False, indent=2)
            console.print(f"Saved page_raw_ocr_results to {output_json_path}")
        
        # Store results for this PDF

    # add to vdb

    vdb_op.run(results)


    elapsed = time.time() - start_time
    console.print(
        f"[bold green]Processed {total_pages_processed} pages from {len(pdf_files)} PDF(s) "
        f"in {elapsed:.2f} seconds[/bold green]"
    )
    
    if return_results:
        return {
            "total_pages_processed": total_pages_processed,
            "total_pdfs": len(pdf_files),
            "elapsed_seconds": elapsed,
            "results": results,
        }
    return None

@app.command()
def run(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=True),
    raw_output_dir: Optional[Path] = typer.Option(None, help="Directory to save raw OCR results (optional)."),
):
    # Load Page Elements model
    page_elements_model = define_model_page_elements("page_element_v3")
    table_structure_model = define_model_table_structure("table_structure_v1")
    graphic_elements_model = define_model_graphic_elements("graphic_elements_v1")
    ocr_model = NemotronOCR(model_dir="/root/.cache/huggingface/hub/models--nvidia--nemotron-ocr-v1/snapshots/90015d3b851ba898ca842f18e948690af49c2427/checkpoints")
    embed_model = llama_nemotron_embed_1b_v2.load_model('cuda:0', True, None, True)
    embed_tokenzier = llama_nemotron_embed_1b_v2.load_tokenizer("longest_first")
    
    if input_dir.is_file():
        pdf_files = [input_dir]
    else:
        pdf_files = [
            str(f) for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".pdf"
        ]

    console.print(f"Processing {len(pdf_files)} PDFs")
    console.print(f"Using page_elements_model device: {page_elements_model.device}")
    console.print(f"Using table_structure_model device: {table_structure_model.device}")
    console.print(f"Using graphic_elements_model device: {graphic_elements_model.device}")
    
    import time

    for i in range(3, 0, -1):
        console.print(f"[bold yellow]{i}[/bold yellow]", end='\r')
        time.sleep(1)
    console.print("[bold green]Go![/bold green]")
    start = time.time()
    vdb_op = LanceDB(uri="./slimgest_lancedb_simple_all_gpu")
    pipe_results = run_pipeline(
        pdf_files[:],
        page_elements_model,
        table_structure_model,
        graphic_elements_model,
        ocr_model,
        embed_model,
        embed_tokenzier,
        vdb_op=vdb_op,
        raw_output_dir=raw_output_dir,
        return_results=True,
    )
    end = time.time()
    console.print(f"Total time: {end - start:.2f} seconds")
    # results = pipe_results["results"]
    import pandas as pd
    df_query = pd.read_csv('/raid/nv-ingest/data/bo767_query_gt.csv').rename(columns={'gt_page':'page'})[['query','pdf','page']]
    df_query['pdf_page'] = df_query.apply(lambda x: f"{x.pdf}_{x.page}", axis=1) 
    
    all_answers = []
    query_texts = df_query['query'].tolist()
    top_k = 10
    result_fields = ["source", "metadata", "text"]
    query_embeddings = []
    for query_batch in query_texts:
        query_tokens = embed_tokenzier(query_batch, return_tensors="pt", padding=True).to('cuda')
        query_model_results = embed_model(**query_tokens)
        query_embeddings += average_pool(query_model_results.last_hidden_state, query_tokens['attention_mask'])
    for query_embed in query_embeddings:
        all_answers.append(
            vdb_op.table.search([query_embed.detach().cpu().numpy()], vector_column_name="vector").select(result_fields).limit(top_k).to_list()
        )
    retrieved_pdf_pages = format_retrieved_answers_lance(all_answers)
    golden_answers = get_correct_answers(df_query)    
    recall_scores = calcuate_recall_list(golden_answers, retrieved_pdf_pages, ks=[1, 3, 5, 10])

    console.print("Recall scores:")
    console.print(recall_scores)
    console.print("[bold green]Done![/bold green]")