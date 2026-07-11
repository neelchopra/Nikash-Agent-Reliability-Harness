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

tasks_app = typer.Typer(help="Task management commands.")
app.add_typer(tasks_app, name="tasks")


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
    model: str = typer.Option("github/gpt-4o-mini", help="LiteLLM model id."),
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


@tasks_app.command("validate")
def tasks_validate(
    tasks_dir: Path = typer.Option(Path("tasks"), help="Directory containing task YAML files."),
) -> None:
    """Validate every task YAML file in a directory."""
    yaml_files = sorted(tasks_dir.glob("*.yaml"))
    if not yaml_files:
        typer.echo(f"no task YAML files found under {tasks_dir}")
        raise typer.Exit(code=1)
    failures = 0
    for path in yaml_files:
        try:
            task = load_task(path)
        except Exception as e:
            failures += 1
            console.print(f"  [red]FAIL[/red]  {path.name}: {e}")
        else:
            console.print(f"  [green]OK[/green]    {path.name}  ({task.id})")
    if failures:
        console.print(f"[red]{failures} task(s) failed validation[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]all {len(yaml_files)} tasks valid[/green]")


if __name__ == "__main__":
    app()
