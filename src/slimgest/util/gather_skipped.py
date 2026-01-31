import csv
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import zipfile

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

app = typer.Typer(help="Gather files associated with skipped stage6 images into a zip file")

console = Console()


def find_stage6_skipped_file(input_dir: Path) -> Optional[Path]:
    """
    Find the most recent stage6_skipped CSV file in the input directory.
    
    Args:
        input_dir: Directory to search
        
    Returns:
        Path to the most recent stage6_skipped file, or None if not found
    """
    skipped_files = sorted(input_dir.glob("stage6_skipped_*.csv"))
    if not skipped_files:
        return None
    return skipped_files[-1]  # Return the most recent one


def read_skipped_images(csv_path: Path) -> List[Dict[str, str]]:
    """
    Read the stage6_skipped CSV file and return the list of skipped images.
    
    Args:
        csv_path: Path to the stage6_skipped CSV file
        
    Returns:
        List of dictionaries with image_path, reason, and details
    """
    skipped_images = []
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            skipped_images.append(row)
    
    return skipped_images


def get_associated_files(img_path: Path) -> Dict[str, Path]:
    """
    Get all files associated with an image file.
    
    Based on the stage999_post_mortem_analysis.py file patterns.
    
    Args:
        img_path: Path to the image file
        
    Returns:
        Dictionary mapping file type to file path
    """
    return {
        "img": img_path,
        "img_overlay": img_path.with_name(img_path.name + ".page_element_detections.png"),
        "pdfium_text": img_path.with_suffix(".pdfium_text.txt"),
        "stage2": img_path.with_name(img_path.name + ".page_elements_v3.json"),
        "stage3": img_path.with_name(img_path.name + ".graphic_elements_v1.json"),
        "stage4": img_path.with_name(img_path.name + ".table_structure_v1.json"),
        "stage5": img_path.with_name(img_path.name + ".nemotron_ocr_v1.json"),
        "embedder_input": img_path.with_name(img_path.name + ".embedder-input.txt"),
    }


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


@app.command()
def run(
    input_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        help="Directory containing stage outputs and stage6_skipped CSV file"
    ),
    output_zip: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to output zip file (default: skipped_files_<timestamp>.zip in current directory)"
    ),
    skipped_file: Optional[Path] = typer.Option(
        None,
        "--skipped-file",
        help="Path to stage6_skipped CSV file (default: auto-detect most recent in input-dir)"
    ),
    include_missing: bool = typer.Option(
        False,
        "--include-missing",
        help="Include references to missing files in the manifest"
    ),
):
    """
    Gather all files associated with skipped stage6 images into a zip file.
    
    This command reads a stage6_skipped CSV file (auto-detected or specified),
    finds all associated files for each skipped image, and packages them into
    a zip file for offline analysis.
    
    \b
    Associated files include:
    - Original image (.png)
    - Page element detections overlay (.page_element_detections.png)
    - PDFium extracted text (.pdfium_text.txt)
    - Stage 2 output (.page_elements_v3.json)
    - Stage 3 output (.graphic_elements_v1.json)
    - Stage 4 output (.table_structure_v1.json)
    - Stage 5 output (.nemotron_ocr_v1.json)
    - Embedder input (.embedder-input.txt)
    
    \b
    Examples:
        # Auto-detect stage6_skipped file and create zip
        slimgest util gather-skipped ./outputs/simple
        
        # Specify custom output location
        slimgest util gather-skipped ./outputs/simple -o ~/skipped_analysis.zip
        
        # Use specific skipped file
        slimgest util gather-skipped ./outputs/simple --skipped-file ./outputs/simple/stage6_skipped_20260131_120000.csv
    """
    
    console.print(f"\n[bold cyan]Gathering skipped stage6 files...[/bold cyan]")
    console.print(f"[yellow]Input directory:[/yellow] {input_dir}\n")
    
    # Find or validate skipped file
    if skipped_file:
        if not skipped_file.exists():
            console.print(f"[bold red]Error:[/bold red] Skipped file not found: {skipped_file}")
            raise typer.Exit(1)
        console.print(f"[yellow]Using skipped file:[/yellow] {skipped_file}")
    else:
        skipped_file = find_stage6_skipped_file(input_dir)
        if not skipped_file:
            console.print(
                f"[bold red]Error:[/bold red] No stage6_skipped_*.csv file found in {input_dir}"
            )
            console.print(
                "[yellow]Hint:[/yellow] Specify a file with --skipped-file option"
            )
            raise typer.Exit(1)
        console.print(f"[yellow]Auto-detected skipped file:[/yellow] {skipped_file}")
    
    # Read skipped images
    console.print("\n[bold cyan]Reading skipped images...[/bold cyan]")
    try:
        skipped_images = read_skipped_images(skipped_file)
    except Exception as e:
        console.print(f"[bold red]Error reading skipped file:[/bold red] {e}")
        raise typer.Exit(1)
    
    console.print(f"[green]Found {len(skipped_images)} skipped images[/green]\n")
    
    if not skipped_images:
        console.print("[yellow]No skipped images to process.[/yellow]")
        return
    
    # Set default output path
    if not output_zip:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_zip = Path(f"skipped_files_{timestamp}.zip")
    
    # Ensure output directory exists
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    
    # Gather files
    console.print("[bold cyan]Scanning for associated files...[/bold cyan]")
    
    files_to_zip: List[tuple[Path, str]] = []  # (source_path, zip_path)
    missing_files: Dict[str, List[str]] = {}  # image_name -> list of missing file types
    total_size = 0
    
    # Create a summary of what we're gathering
    file_type_counts: Dict[str, int] = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Scanning files...", total=len(skipped_images))
        
        for skip_record in skipped_images:
            image_path_str = skip_record.get("image_path", "")
            if not image_path_str:
                continue
            
            img_path = Path(image_path_str)
            img_name = img_path.name
            
            # Get all associated files
            associated_files = get_associated_files(img_path)
            
            missing_for_this_image = []
            
            for file_type, file_path in associated_files.items():
                if file_path.exists():
                    # Add to zip with a structured path
                    zip_path = f"{img_name}/{file_type}/{file_path.name}"
                    files_to_zip.append((file_path, zip_path))
                    total_size += file_path.stat().st_size
                    file_type_counts[file_type] = file_type_counts.get(file_type, 0) + 1
                else:
                    missing_for_this_image.append(file_type)
            
            if missing_for_this_image:
                missing_files[img_name] = missing_for_this_image
            
            progress.update(task, advance=1)
    
    # Add the skipped CSV file itself
    csv_zip_path = "manifest/stage6_skipped.csv"
    files_to_zip.append((skipped_file, csv_zip_path))
    total_size += skipped_file.stat().st_size
    
    # Display summary
    console.print("\n[bold cyan]═══ Summary ═══[/bold cyan]")
    
    table = Table(title="Files to Archive")
    table.add_column("File Type", style="cyan")
    table.add_column("Count", style="green", justify="right")
    
    for file_type in sorted(file_type_counts.keys()):
        table.add_row(file_type, str(file_type_counts[file_type]))
    
    table.add_row("manifest", "1", style="dim")
    console.print(table)
    
    console.print(f"\n[yellow]Total files:[/yellow] {len(files_to_zip)}")
    console.print(f"[yellow]Total size:[/yellow] {format_size(total_size)}")
    console.print(f"[yellow]Skipped images:[/yellow] {len(skipped_images)}")
    
    if missing_files:
        console.print(f"\n[yellow]⚠ Some files are missing for {len(missing_files)} images[/yellow]")
        if len(missing_files) <= 5:
            for img_name, missing in list(missing_files.items())[:5]:
                console.print(f"  - {img_name}: missing {', '.join(missing)}")
    
    # Create zip file
    console.print(f"\n[bold cyan]Creating zip file:[/bold cyan] {output_zip}")
    
    try:
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Adding files to zip...", total=len(files_to_zip))
                
                for source_path, zip_path in files_to_zip:
                    zipf.write(source_path, zip_path)
                    progress.update(task, advance=1)
            
            # Add a README with information about the archive
            readme_content = f"""Skipped Stage6 Files Archive
============================

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Source Directory: {input_dir}
Skipped File: {skipped_file}

Summary:
--------
- Skipped Images: {len(skipped_images)}
- Total Files: {len(files_to_zip)}
- Total Size: {format_size(total_size)}

File Types Included:
-------------------
"""
            for file_type, count in sorted(file_type_counts.items()):
                readme_content += f"- {file_type}: {count} files\n"
            
            if missing_files:
                readme_content += f"\nMissing Files:\n-------------\n"
                readme_content += f"Some files were missing for {len(missing_files)} images.\n"
                if include_missing:
                    readme_content += "\nDetailed missing files list:\n"
                    for img_name, missing in sorted(missing_files.items()):
                        readme_content += f"- {img_name}: {', '.join(missing)}\n"
            
            readme_content += f"\nDirectory Structure:\n-------------------\n"
            readme_content += f"Each image has its own directory with subdirectories for each file type:\n"
            readme_content += f"  <image_name>/\n"
            readme_content += f"    img/               - Original image\n"
            readme_content += f"    img_overlay/       - Detection visualization\n"
            readme_content += f"    pdfium_text/       - Extracted PDF text\n"
            readme_content += f"    stage2/            - Page elements detection\n"
            readme_content += f"    stage3/            - Graphic elements detection\n"
            readme_content += f"    stage4/            - Table structure detection\n"
            readme_content += f"    stage5/            - Nemotron OCR results\n"
            readme_content += f"    embedder_input/    - Embedding input text\n"
            readme_content += f"  manifest/\n"
            readme_content += f"    stage6_skipped.csv - Original skipped file list\n"
            
            zipf.writestr("README.txt", readme_content)
    
    except Exception as e:
        console.print(f"\n[bold red]Error creating zip file:[/bold red] {e}")
        raise typer.Exit(1)
    
    # Success!
    final_size = output_zip.stat().st_size
    console.print(f"\n[bold green]✓ Successfully created archive[/bold green]")
    console.print(f"[yellow]Output:[/yellow] {output_zip}")
    console.print(f"[yellow]Compressed size:[/yellow] {format_size(final_size)}")
    console.print(f"[yellow]Compression ratio:[/yellow] {(1 - final_size/total_size)*100:.1f}%")
