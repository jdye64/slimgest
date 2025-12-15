import typer

app = typer.Typer(help="Prepare release candidate")

@app.command()
def run(
):
    print("TODO: Implement prepare_rc utilities")
