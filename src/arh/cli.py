import typer

app = typer.Typer(help="Agent Reliability Harness: pass^k for MCP tool use.")


@app.callback()
def main() -> None:
    """Agent Reliability Harness: pass^k for MCP tool use."""


@app.command()
def version() -> None:
    """Print the harness version."""
    from arh import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()
