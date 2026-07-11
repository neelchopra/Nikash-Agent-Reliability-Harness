import asyncio
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console

from arh.report import render_table, summarize
from arh.runner.trial import run_task_trials
from arh.store.jsonl import JsonlStore
from arh.tasks import load_task

app = typer.Typer(help="Agent Reliability Harness: pass^k for MCP tool use.")
console = Console()


@app.callback()
def main() -> None:
    """Agent Reliability Harness: pass^k for MCP tool use."""


@app.command()
def version() -> None:
    """Print the harness version."""
    from arh import __version__

    typer.echo(__version__)


@app.command()
def run(
    task: Path = typer.Option(Path("tasks/fs-rename-001.yaml"), help="Task YAML file."),
    model: str = typer.Option("gemini/gemini-2.5-flash", help="LiteLLM model id."),
    n: int = typer.Option(10, help="Trials per task."),
    out: Path = typer.Option(Path("results"), help="Output directory."),
) -> None:
    """Run one task for n trials and print the pass^k report."""
    task_obj = load_task(task)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_file = out / f"{task_obj.id}-{stamp}.jsonl"
    store = JsonlStore(results_file)
    runs_root = out / ".arh_runs"
    done = 0

    def on_result(r) -> None:
        nonlocal done
        done += 1
        mark = "[green]PASS[/green]" if r.success else (
            "[yellow]INFRA[/yellow]" if r.failure_source == "infra" else "[red]FAIL[/red]"
        )
        console.print(f"  trial {done:>2}/{n}  {mark}  {r.grade_detail}")

    console.print(f"[bold]{task_obj.id}[/bold] x {n} trials on [bold]{model}[/bold]")
    results = asyncio.run(
        run_task_trials(task_obj, model, n, runs_root, store, on_result=on_result)
    )
    console.print(render_table(summarize(results)))
    console.print(f"results written to {results_file}")


@app.command()
def report(results_file: Path = typer.Argument(..., help="A results .jsonl file.")) -> None:
    """Recompute and print the pass^k table from a results file."""
    rows = JsonlStore(results_file).load()
    if not rows:
        typer.echo("no results found")
        raise typer.Exit(code=1)
    console.print(render_table(summarize(rows)))


if __name__ == "__main__":
    app()
