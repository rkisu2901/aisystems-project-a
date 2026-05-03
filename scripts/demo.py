"""
Interactive demo CLI for Project A.
Run: python scripts/demo.py
"""
import os
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich import box
from langfuse import Langfuse
from dotenv import load_dotenv

load_dotenv()

# Import from same scripts directory
sys.path.insert(0, os.path.dirname(__file__))
from rag import ask

console = Console()
langfuse = Langfuse()

LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")


def run_query(query):
    console.print()
    console.print(Panel(query, title="[bold cyan]User Query[/]", border_style="cyan"))

    with console.status("[bold yellow]Running RAG pipeline...[/]"):
        result = ask(query)

    table = Table(title="Retrieved Chunks", box=box.ROUNDED, show_lines=True, title_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Source", style="cyan", width=28)
    table.add_column("Similarity", justify="center", width=10)
    table.add_column("Content Preview", width=60)

    for i, chunk in enumerate(result["retrieved_chunks"]):
        sim = chunk["similarity"]
        sim_color = "green" if sim > 0.8 else "yellow" if sim > 0.7 else "red"
        table.add_row(
            str(i + 1),
            f"{chunk['doc_name']}\nchunk {chunk['chunk_index']}",
            f"[{sim_color}]{sim:.4f}[/]",
            chunk["content"][:200].replace("\n", " ") + "...",
        )

    console.print(table)
    console.print()
    console.print(Panel(Markdown(result["answer"]), title="[bold green]Answer[/]", border_style="green"))

    trace_url = f"{LANGFUSE_HOST}/trace/{result['trace_id']}"
    console.print(f"\n[dim]Trace:[/] [link={trace_url}]{trace_url}[/link]  |  [dim]Latency:[/] {result['elapsed_seconds']}s\n")
    langfuse.flush()


def main():
    console.print(Panel("[bold]Acmera Knowledge Base Assistant[/]\n[dim]Project A — Naive RAG Pipeline[/]", border_style="blue"))

    while True:
        console.print()
        query = console.input("[bold cyan]Ask a question (or 'q' to quit): [/]").strip()
        if query.lower() == "q":
            break
        if query:
            run_query(query)


if __name__ == "__main__":
    main()
