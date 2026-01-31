import csv
from pathlib import Path
from typing import Dict, List, Optional, Set

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Find skipped embeddings that appear in query ground truth")

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


def extract_pdf_page_from_image_path(image_path: str) -> Optional[str]:
    """
    Extract the pdf_page identifier from an image path.
    
    Image paths typically look like:
      /path/to/1102434_page0019.png
    
    This should extract: 1102434_20 (note: page is 1-indexed in query, 0-indexed in filename)
    
    Args:
        image_path: Path to the image file
        
    Returns:
        pdf_page identifier (e.g., "1102434_20"), or None if cannot parse
    """
    path = Path(image_path)
    filename = path.stem  # e.g., "1102434_page0019"
    
    # Try to match pattern like "1102434_page0019"
    import re
    match = re.match(r'^(\d+)_page(\d+)$', filename)
    if match:
        pdf_id = match.group(1)
        page_num_zero_indexed = int(match.group(2))
        page_num_one_indexed = page_num_zero_indexed + 1
        return f"{pdf_id}_{page_num_one_indexed}"
    
    return None


def read_query_file(query_path: Path) -> List[Dict[str, str]]:
    """
    Read the query CSV file and return the list of queries.
    
    Args:
        query_path: Path to the query CSV file
        
    Returns:
        List of dictionaries with query data
    """
    queries = []
    
    with open(query_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append(row)
    
    return queries


def get_pdf_pages_from_queries(queries: List[Dict[str, str]]) -> Set[str]:
    """
    Extract the set of pdf_page identifiers from the query list.
    
    Args:
        queries: List of query dictionaries
        
    Returns:
        Set of pdf_page identifiers (e.g., {"1102434_20", "1102434_4", ...})
    """
    pdf_pages = set()
    
    for query in queries:
        pdf_page = query.get("pdf_page", "").strip()
        if pdf_page:
            pdf_pages.add(pdf_page)
    
    return pdf_pages


@app.command()
def run(
    input_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        help="Directory containing stage6_skipped CSV file"
    ),
    query_file: Path = typer.Option(
        ...,
        "--query-file",
        exists=True,
        file_okay=True,
        help="Path to query CSV file with ground truth"
    ),
    skipped_file: Optional[Path] = typer.Option(
        None,
        "--skipped-file",
        help="Path to stage6_skipped CSV file (default: auto-detect most recent in input-dir)"
    ),
):
    """
    Find skipped embeddings that appear in query ground truth.
    
    This command identifies pages that were skipped during stage 6 embedding
    (typically due to CUDA OOM errors) but are referenced in the query ground
    truth file. These pages are critical for recall evaluation and need to be
    re-processed.
    
    \b
    Examples:
        # Auto-detect skipped file
        slimgest util find-skipped-embeddings-in-query ./outputs/simple --query-file bo767_query_gt.csv
        
        # Specify custom skipped file
        slimgest util find-skipped-embeddings-in-query ./outputs/simple \\
            --query-file queries.csv \\
            --skipped-file ./outputs/simple/stage6_skipped_20260131_120000.csv
    """
    
    console.print(f"\n[bold cyan]Finding skipped embeddings in query ground truth...[/bold cyan]")
    console.print(f"[yellow]Input directory:[/yellow] {input_dir}")
    console.print(f"[yellow]Query file:[/yellow] {query_file}\n")
    
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
    
    console.print(f"[green]Found {len(skipped_images)} skipped images[/green]")
    
    # Extract pdf_page identifiers from skipped images
    skipped_pdf_pages: Dict[str, Dict[str, str]] = {}  # pdf_page -> skip_record
    unparseable_count = 0
    
    for skip_record in skipped_images:
        image_path = skip_record.get("image_path", "")
        pdf_page = extract_pdf_page_from_image_path(image_path)
        
        if pdf_page:
            skipped_pdf_pages[pdf_page] = skip_record
        else:
            unparseable_count += 1
    
    console.print(f"[green]Parsed {len(skipped_pdf_pages)} pdf_page identifiers from skipped images[/green]")
    if unparseable_count > 0:
        console.print(f"[yellow]Warning: Could not parse {unparseable_count} image paths[/yellow]")
    
    # Read query file
    console.print("\n[bold cyan]Reading query file...[/bold cyan]")
    try:
        queries = read_query_file(query_file)
    except Exception as e:
        console.print(f"[bold red]Error reading query file:[/bold red] {e}")
        raise typer.Exit(1)
    
    console.print(f"[green]Found {len(queries)} queries[/green]")
    
    # Extract pdf_page identifiers from queries
    query_pdf_pages = get_pdf_pages_from_queries(queries)
    console.print(f"[green]Found {len(query_pdf_pages)} unique pdf_pages in queries[/green]")
    
    # Find intersection
    console.print("\n[bold cyan]Finding intersection...[/bold cyan]")
    intersection = set(skipped_pdf_pages.keys()) & query_pdf_pages
    
    if not intersection:
        console.print("\n[bold green]✓ No skipped pages found in query ground truth![/bold green]")
        console.print("All query pages have embeddings available.")
        return
    
    # Display results
    console.print(f"\n[bold red]⚠ Found {len(intersection)} skipped pages in query ground truth![/bold red]")
    
    # Create detailed table
    table = Table(title="Skipped Pages in Query Ground Truth")
    table.add_column("PDF_Page", style="cyan")
    table.add_column("Image Path", style="yellow")
    table.add_column("Skip Reason", style="red")
    table.add_column("Queries Affected", style="magenta", justify="right")
    
    # Count queries per pdf_page
    query_counts: Dict[str, int] = {}
    for query in queries:
        pdf_page = query.get("pdf_page", "").strip()
        if pdf_page in intersection:
            query_counts[pdf_page] = query_counts.get(pdf_page, 0) + 1
    
    total_queries_affected = 0
    for pdf_page in sorted(intersection):
        skip_record = skipped_pdf_pages[pdf_page]
        image_path = Path(skip_record.get("image_path", "")).name
        reason = skip_record.get("reason", "unknown")
        query_count = query_counts.get(pdf_page, 0)
        total_queries_affected += query_count
        
        table.add_row(
            pdf_page,
            image_path,
            reason,
            str(query_count)
        )
    
    console.print(table)
    
    # Summary statistics
    console.print("\n[bold cyan]═══ Summary ═══[/bold cyan]")
    console.print(f"[yellow]Total skipped pages:[/yellow] {len(skipped_images)}")
    console.print(f"[yellow]Total query pages:[/yellow] {len(query_pdf_pages)}")
    console.print(f"[yellow]Skipped pages in queries:[/yellow] {len(intersection)}")
    console.print(f"[yellow]Queries affected:[/yellow] {total_queries_affected} out of {len(queries)}")
    console.print(f"[yellow]Percentage of queries affected:[/yellow] {total_queries_affected/len(queries)*100:.1f}%")
    
    # Breakdown by skip reason
    console.print("\n[bold cyan]Skip Reasons:[/bold cyan]")
    reason_counts: Dict[str, int] = {}
    for pdf_page in intersection:
        reason = skipped_pdf_pages[pdf_page].get("reason", "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    
    for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
        console.print(f"  - {reason}: {count} pages")
    
    # Recommendations
    console.print("\n[bold yellow]⚠ Recommendations:[/bold yellow]")
    console.print("1. Re-process these pages with more available GPU memory")
    console.print("2. Consider using smaller batch sizes for these pages")
    console.print("3. Temporarily reduce image resolution for problematic pages")
    console.print("4. Exclude affected queries from recall evaluation until resolved")
    
    console.print()
