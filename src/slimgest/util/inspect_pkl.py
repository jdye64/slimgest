import pickle
from pathlib import Path
from typing import Any

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

app = typer.Typer(help="Inspect pickle files and display their contents")

console = Console()


def analyze_object(obj: Any, max_depth: int = 3, current_depth: int = 0) -> dict:
    """Recursively analyze an object to understand its structure."""
    result = {
        "type": type(obj).__name__,
        "repr": repr(obj)[:100] + ("..." if len(repr(obj)) > 100 else ""),
    }
    
    if current_depth >= max_depth:
        return result
    
    # Handle different types
    if isinstance(obj, dict):
        result["length"] = len(obj)
        result["keys"] = list(obj.keys())[:10]  # Show first 10 keys
        if obj:
            sample_key = list(obj.keys())[0]
            result["sample_value"] = analyze_object(obj[sample_key], max_depth, current_depth + 1)
    elif isinstance(obj, (list, tuple)):
        result["length"] = len(obj)
        if obj:
            result["first_element"] = analyze_object(obj[0], max_depth, current_depth + 1)
    elif hasattr(obj, "__dict__"):
        result["attributes"] = list(vars(obj).keys())[:10]
    elif isinstance(obj, (str, int, float, bool)):
        result["value"] = obj
    
    return result


def build_tree(name: str, analysis: dict, tree: Tree = None) -> Tree:
    """Build a rich Tree representation of the analysis."""
    if tree is None:
        tree = Tree(f"[bold cyan]{name}[/bold cyan]")
    
    tree.add(f"[yellow]Type:[/yellow] {analysis['type']}")
    
    if "value" in analysis:
        tree.add(f"[yellow]Value:[/yellow] {analysis['value']}")
    
    if "length" in analysis:
        tree.add(f"[yellow]Length:[/yellow] {analysis['length']}")
    
    if "keys" in analysis:
        keys_branch = tree.add("[yellow]Keys (first 10):[/yellow]")
        for key in analysis["keys"]:
            keys_branch.add(f"{key}")
        
        if "sample_value" in analysis:
            sample_branch = tree.add("[yellow]Sample Value:[/yellow]")
            build_tree("", analysis["sample_value"], sample_branch)
    
    if "first_element" in analysis:
        elem_branch = tree.add("[yellow]First Element:[/yellow]")
        build_tree("", analysis["first_element"], elem_branch)
    
    if "attributes" in analysis:
        attrs_branch = tree.add("[yellow]Attributes (first 10):[/yellow]")
        for attr in analysis["attributes"]:
            attrs_branch.add(f"{attr}")
    
    return tree


@app.command()
def run(
    pickle_file: Path = typer.Argument(..., exists=True, file_okay=True, help="Path to pickle file to inspect"),
    max_depth: int = typer.Option(3, help="Maximum depth to analyze nested structures"),
    show_raw: bool = typer.Option(False, help="Show raw repr() of the object"),
):
    """
    Inspect a pickle file and display an overview of its contents.
    
    This command loads a pickle file and displays information about its structure,
    including types, lengths, keys, and sample values for nested structures.
    """
    console.print(f"\n[bold green]Loading pickle file:[/bold green] {pickle_file}")
    
    try:
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        console.print(f"[bold red]Error loading pickle file:[/bold red] {e}")
        raise typer.Exit(1)
    
    
    breakpoint()
    console.print("[bold green]✓[/bold green] File loaded successfully\n")
    
    # Basic info
    console.print("[bold cyan]═══ Basic Information ═══[/bold cyan]")
    console.print(f"[yellow]Root Type:[/yellow] {type(data).__name__}")
    console.print(f"[yellow]File Size:[/yellow] {pickle_file.stat().st_size:,} bytes\n")
    
    # Show raw representation if requested
    if show_raw:
        console.print("[bold cyan]═══ Raw Representation ═══[/bold cyan]")
        rprint(data)
        console.print()
    
    # Analyze structure
    console.print("[bold cyan]═══ Structure Analysis ═══[/bold cyan]")
    analysis = analyze_object(data, max_depth=max_depth)
    tree = build_tree("Root Object", analysis)
    console.print(tree)
    
    # Additional statistics for common types
    console.print("\n[bold cyan]═══ Summary ═══[/bold cyan]")
    
    if isinstance(data, dict):
        table = Table(title="Dictionary Summary")
        table.add_column("Key", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("Sample/Info", style="green")
        
        for i, (key, value) in enumerate(data.items()):
            if i >= 20:  # Limit to first 20 entries
                table.add_row("...", "...", f"({len(data) - 20} more items)")
                break
            
            value_type = type(value).__name__
            if isinstance(value, (list, tuple, dict, str)):
                info = f"length: {len(value)}"
            elif isinstance(value, (int, float, bool)):
                info = str(value)
            else:
                info = repr(value)[:50]
            
            table.add_row(str(key), value_type, info)
        
        console.print(table)
    
    elif isinstance(data, (list, tuple)):
        console.print(f"[yellow]Total items:[/yellow] {len(data)}")
        if data:
            console.print(f"[yellow]First item type:[/yellow] {type(data[0]).__name__}")
            if len(data) > 1:
                # Check if all items are the same type
                all_same_type = all(type(item) == type(data[0]) for item in data)
                console.print(f"[yellow]All same type:[/yellow] {all_same_type}")
    
    console.print()
