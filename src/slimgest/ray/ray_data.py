from pathlib import Path
from typing import List, Tuple, Optional

from rich.console import Console
from rich.traceback import install
import typer
import ray
import glob
from slimgest.local.simple_all_gpu import run_pipeline, process_pdf_pages

from nemotron_page_elements_v3.model import define_model as define_model_page_elements
from nemotron_table_structure_v1.model import define_model as define_model_table_structure
from nemotron_graphic_elements_v1.model import define_model as define_model_graphic_elements
from nemotron_ocr.inference.pipeline import NemotronOCR
import llama_nemotron_embed_1b_v2


page_elements_model = define_model_page_elements("page_elements_v3")
table_structure_model = define_model_table_structure("table_structure_v1")
graphic_elements_model = define_model_graphic_elements("graphic_elements_v1")
ocr_model = NemotronOCR(model_dir="/root/.cache/huggingface/hub/models--nvidia--nemotron-ocr-v1/snapshots/90015d3b851ba898ca842f18e948690af49c2427/checkpoints")
embed_model = llama_nemotron_embed_1b_v2.load_model('cuda:0', True, None, True)
embed_tokenzier = llama_nemotron_embed_1b_v2.load_tokenizer()

app = typer.Typer(help="Ray Data Parallelism for Slim-Gest")
install(show_locals=False)
console = Console()

class RunLogic:
    def __init__(
        self,
        raw_output_dir: Optional[Path] = None,
        page_elements_model=None,
        table_structure_model=None,
        graphic_elements_model=None,
        ocr_model=None,
        embed_model=None,
        embed_tokenizer=None,
    ):
        self.page_elements_model = page_elements_model
        self.table_structure_model = table_structure_model
        self.graphic_elements_model = graphic_elements_model
        self.ocr_model = ocr_model
        self.embed_model = embed_model
        self.embed_tokenizer = embed_tokenizer
        # ocr_model = NemotronOCR(model_dir="/home/jdyer/Development/slim-gest/models/nemotron-ocr-v1/checkpoints")
        console.print(f"Using page_elements_model device: {self.page_elements_model.device}")
        console.print(f"Using table_structure_model device: {self.table_structure_model.device}")
        console.print(f"Using graphic_elements_model device: {self.graphic_elements_model.device}")
        self.raw_output_dir = raw_output_dir
    

    def __call__(
        self,
        pdf_files: List[str],
    ):
        # run_pipeline(
        #     pdf_files,
        #     self.page_elements_model,
        #     self.table_structure_model,
        #     self.graphic_elements_model,
        #     self.ocr_model,
        #     raw_output_dir=self.raw_output_dir,
        # )
        results = []
        for pdf_path in pdf_files:
            results += process_pdf_pages(
                pdf_path,
                self.page_elements_model,
                self.table_structure_model,
                self.graphic_elements_model,
                self.ocr_model,
                self.embed_model,
                self.embed_tokenizer,
                device="cuda",
                dpi=300,
            )
        return  {"processed_files": results}


@app.command()
def run(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=True),
    raw_output_dir: Optional[Path] = typer.Option(None, help="Directory to save raw OCR results (optional)."),
):
    ray.init(runtime_env={"env_vars": {"RAY_DEBUG": "legacy"}, "working_dir": "./"}, address="local")

    # boilerplate code then call the local run_pipeline function ....
    run_logic = RunLogic(
        raw_output_dir=raw_output_dir, 
        page_elements_model=page_elements_model, 
        table_structure_model=table_structure_model, 
        graphic_elements_model=graphic_elements_model, 
        ocr_model=ocr_model,
        embed_model=embed_model,
        embed_tokenizer=embed_tokenzier,
    )
    files = glob.glob(str(input_dir / "**/*"), recursive=True)
    console.print(f"Processing {len(files)} PDFs")
    # console.print(f"Using page_elements_model device: {page_elements_model.device}")
    # console.print(f"Using table_structure_model device: {table_structure_model.device}")
    # console.print(f"Using graphic_elements_model device: {graphic_elements_model.device}")
    # breakpoint()
    # run_func = partial(run_pipeline,
    #     page_elements_model = page_elements_model,
    #     table_structure_model = table_structure_model,
    #     graphic_elements_model = graphic_elements_model,
    #     ocr_model = ocr_model,
    #     raw_output_dir = raw_output_dir,
    #     return_results=True
    # )

    ds =  ray.data.from_items(files[:])
    map_res = ds.map_batches(
        run_logic, 
        num_gpus=1,
        batch_size=1,
        compute=ray.data.TaskPoolStrategy(size=1)
        )
    results = map_res.take_all()
    print(len(results))
    pass