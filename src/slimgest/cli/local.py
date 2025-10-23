from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict

from slimgest.cli.process.image_process import DeepseekOCRProcessor
import typer
from rich.console import Console

import time
from tqdm import tqdm
import torch

import os
import re
import time

if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from .config import MODEL_PATH, INPUT_PATH, OUTPUT_PATH, PROMPT, MAX_CONCURRENCY, CROP_MODE, NUM_WORKERS
from concurrent.futures import ThreadPoolExecutor
import glob
from PIL import Image
from .deepseek_ocr import DeepseekOCRForCausalLM

from vllm.model_executor.models.registry import ModelRegistry

from vllm import LLM, SamplingParams
from .process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from .process.image_process import DeepseekOCRProcessor
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)


llm = LLM(
    model=MODEL_PATH,
    hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
    block_size=256,
    enforce_eager=False,
    limit_mm_per_prompt={"image": 1},
    trust_remote_code=True, 
    max_model_len=8192,
    swap_space=0,
    max_num_seqs=MAX_CONCURRENCY,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
)

logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=40, window_size=90, whitelist_token_ids={128821, 128822})]  # window for fast；whitelist_token_ids: <td>,</td>

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    logits_processors=logits_processors,
    skip_special_tokens=False,
)

class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    RESET = '\033[0m' 

def clean_formula(text):
    formula_pattern = r'\\\[(.*?)\\\]'
    def process_formula(match):
        formula = match.group(1)
        formula = re.sub(r'\\quad\s*\([^)]*\)', '', formula)
        formula = formula.strip()
        return r'\[' + formula + r'\]'
    cleaned_text = re.sub(formula_pattern, process_formula, text)
    return cleaned_text

def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    mathes_other = []
    for a_match in matches:
        mathes_other.append(a_match[0])
    return matches, mathes_other

def process_single_image(image):
    """single image"""
    prompt_in = PROMPT
    cache_item = {
        "prompt": prompt_in,
        "multi_modal_data": {"image": DeepseekOCRProcessor().tokenize_with_images(images=[image], bos=True, eos=True, cropping=CROP_MODE)},
    }
    return cache_item

app = typer.Typer(help="Process PDFs locally using shared pipeline")
console = Console()

@app.command()
def run(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    output_dir: Path = typer.Argument(..., file_okay=False),
    dpi: int = typer.Option(220, help="Render DPI"),
    vllm_url: str = typer.Option("http://localhost:8001", help="vLLM server URL"),
    model: str = typer.Option("deepseek-ocr", help="Model name"),
    max_pages: Optional[int] = typer.Option(None, help="Limit number of pages"),
):

    # Output all results into a subdirectory named OUTPUT_PATH
    output_results_dir = os.path.abspath(os.path.join('.', OUTPUT_PATH))
    os.makedirs(output_results_dir, exist_ok=True)

    print(f'{Colors.RED}glob images.....{Colors.RESET}')

    t0_glob = time.time()
    images_path = glob.glob(f'{INPUT_PATH}/*')
    t1_glob = time.time()

    images = []

    prompt = PROMPT
    batch_size = 100

    total_time_preproc = 0.0
    total_time_infer = 0.0
    total_time_write = 0.0
    total_batches = 0
    first_batch = True

    tstart_overall = time.time()
    for image_path in images_path:
        image = Image.open(image_path).convert('RGB')
        images.append(image)

        if len(images) >= batch_size:
            t0_preproc = time.time()
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:  
                batch_inputs = list(tqdm(
                    executor.map(process_single_image, images),
                    total=len(images),
                    desc="Pre-processed images"
                ))
            t1_preproc = time.time()
            time_preproc = t1_preproc - t0_preproc
            total_time_preproc += time_preproc

            t0_infer = time.time()
            outputs_list = llm.generate(
                batch_inputs,
                sampling_params=sampling_params
            )
            t1_infer = time.time()
            time_infer = t1_infer - t0_infer
            total_time_infer += time_infer

            t0_write = time.time()
            # Write output to bo767_results instead of OUTPUT_PATH
            batch_image_paths = images_path[total_batches*batch_size:(total_batches+1)*batch_size]

            for output, image_file in zip(outputs_list, batch_image_paths):
                content = output.outputs[0].text
                base_name = os.path.basename(image_file)
                mmd_det_path = os.path.join(output_results_dir, base_name.replace('.jpg', '_det.md'))

                with open(mmd_det_path, 'w', encoding='utf-8') as afile:
                    afile.write(content)

                content = clean_formula(content)
                matches_ref, mathes_other = re_match(content)
                for idx, a_match_other in enumerate(tqdm(mathes_other, desc="other", leave=False)):
                    content = content.replace(a_match_other, '').replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n').replace('<center>', '').replace('</center>', '')

                mmd_path = os.path.join(output_results_dir, base_name.replace('.jpg', '.md'))

                with open(mmd_path, 'w', encoding='utf-8') as afile:
                    afile.write(content)
            t1_write = time.time()
            time_write = t1_write - t0_write
            total_time_write += time_write

            print(f"{Colors.GREEN}Batch {total_batches+1} complete:{Colors.RESET} Pre-processing: {time_preproc:.2f}s | Inference: {time_infer:.2f}s | Write: {time_write:.2f}s")

            images.clear()
            total_batches += 1

    # Handle any remaining images (not forming a full batch)
    if len(images) > 0:
        t0_preproc = time.time()
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            batch_inputs = list(tqdm(
                executor.map(process_single_image, images),
                total=len(images),
                desc="Pre-processed images"
            ))
        t1_preproc = time.time()
        time_preproc = t1_preproc - t0_preproc
        total_time_preproc += time_preproc

        t0_infer = time.time()
        outputs_list = llm.generate(
            batch_inputs,
            sampling_params=sampling_params
        )
        t1_infer = time.time()
        time_infer = t1_infer - t0_infer
        total_time_infer += time_infer

        t0_write = time.time()
        # Write output to bo767_results instead of OUTPUT_PATH
        batch_image_paths = images_path[total_batches*batch_size:total_batches*batch_size+len(images)]

        for output, image_file in zip(outputs_list, batch_image_paths):
            content = output.outputs[0].text
            base_name = os.path.basename(image_file)
            mmd_det_path = os.path.join(output_results_dir, base_name.replace('.jpg', '_det.md'))

            with open(mmd_det_path, 'w', encoding='utf-8') as afile:
                afile.write(content)

            content = clean_formula(content)
            matches_ref, mathes_other = re_match(content)
            for idx, a_match_other in enumerate(tqdm(mathes_other, desc="other", leave=False)):
                content = content.replace(a_match_other, '').replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n').replace('<center>', '').replace('</center>', '')

            mmd_path = os.path.join(output_results_dir, base_name.replace('.jpg', '.md'))

            with open(mmd_path, 'w', encoding='utf-8') as afile:
                afile.write(content)
        t1_write = time.time()
        time_write = t1_write - t0_write
        total_time_write += time_write

        print(f"{Colors.GREEN}Final batch complete:{Colors.RESET} Pre-processing: {time_preproc:.2f}s | Inference: {time_infer:.2f}s | Write: {time_write:.2f}s")

        total_batches += 1

    tend_overall = time.time()

    n_total_pages = len(images_path)
    total_walltime = tend_overall - tstart_overall

    print("\n--------- Runtime Summary ---------")
    print(f"Image globbing           : {(t1_glob - t0_glob):.2f}s")
    print(f"Input pre-processing     : {total_time_preproc:.2f}s")
    print(f"Model inference          : {total_time_infer:.2f}s")
    print(f"Markdown file writing    : {total_time_write:.2f}s")
    print(f"-----------------------------------")
    print(f"Total (sum of segments)  : {(t1_glob - t0_glob) + total_time_preproc + total_time_infer + total_time_write:.2f}s")
    print(f"Actual total wall time   : {total_walltime:.2f}s")
    if total_walltime > 0 and n_total_pages > 0:
        print(f"Pages processed: {n_total_pages} | Pages per second (PPS): {n_total_pages/total_walltime:.2f}")
    print("-----------------------------------")
