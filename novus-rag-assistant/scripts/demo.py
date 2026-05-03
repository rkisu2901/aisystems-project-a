"""
demo.py — Rich CLI demo for the Novus Bank RAG assistant.

Displays a color-coded retrieval table (doc, similarity score, preview)
followed by the generated answer. Good for live demos and sanity checks.

Usage:
    python scripts/demo.py
    python scripts/demo.py --query "What is the penalty for breaking an FD early?"
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.rag import ask

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

DEMO_QUERIES = [
    "What is the minimum SIP amount at Novus Bank?",
    "If I report a fraud transaction after 5 days, what is my liability?",
    "What is the penalty for breaking a fixed deposit early?",
    "How do I reactivate a dormant account?",
    "What documents are needed to open an account?",
]

console = Console() if RICH_AVAILABLE else None


def similarity_color(score: float) -> str:
    """Map similarity score to a Rich color name."""
    if score >= 0.80:
        return "bold green"
    if score >= 0.65:
        return "yellow"
    return "red"


def display_result(result: dict) -> None:
    if not RICH_AVAILABLE:
        print(f"\nQ: {result['query']}")
        print(f"A: {result['answer']}")
        for i, c in enumerate(result["retrieved_chunks"], 1):
            print(f"  [{i}] {c['doc_id']} (sim={c['similarity']:.3f})")
        print(f"Elapsed: {result['elapsed_seconds']}s\n")
        return

    console.rule(f"[bold blue]Query")
    console.print(f"  [italic]{result['query']}[/italic]\n")

    # Retrieval table
    table = Table(title="Retrieved Chunks (top-5)", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Source Document", min_width=30)
    table.add_column("Similarity", justify="right", min_width=10)
    table.add_column("Preview", max_width=60)

    for i, chunk in enumerate(result["retrieved_chunks"], 1):
        sim = chunk["similarity"]
        color = similarity_color(sim)
        preview = chunk["content"][:120].replace("\n", " ") + "…"
        table.add_row(
            str(i),
            chunk["doc_id"],
            Text(f"{sim:.4f}", style=color),
            preview,
        )

    console.print(table)

    # Answer panel
    console.print(
        Panel(
            result["answer"],
            title="[bold green]Novus Assist Answer",
            border_style="green",
        )
    )

    trace_info = f"trace_id={result['trace_id']}" if result["trace_id"] else "LangFuse not configured"
    console.print(
        f"  [dim]Elapsed: {result['elapsed_seconds']}s | {trace_info}[/dim]\n"
    )


def run_interactive():
    if RICH_AVAILABLE:
        console.print(
            Panel(
                "[bold]Novus Bank Knowledge Base[/bold]\n"
                "Ask any question about Novus Bank's policies and products.\n"
                "Type [bold red]quit[/bold red] to exit.",
                title="Novus Assist",
                border_style="blue",
            )
        )
    else:
        print("=== Novus Bank RAG Demo ===\nType 'quit' to exit.\n")

    while True:
        try:
            if RICH_AVAILABLE:
                query = console.input("[bold blue]Question:[/bold blue] ").strip()
            else:
                query = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query or query.lower() in {"quit", "exit"}:
            break

        if RICH_AVAILABLE:
            with console.status("[bold green]Thinking…"):
                result = ask(query)
        else:
            result = ask(query)

        display_result(result)


def main():
    parser = argparse.ArgumentParser(description="Novus Bank RAG demo")
    parser.add_argument("--query", "-q", type=str, help="Single query (non-interactive)")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run through all built-in demo queries",
    )
    args = parser.parse_args()

    if args.query:
        result = ask(args.query)
        display_result(result)
    elif args.demo:
        for q in DEMO_QUERIES:
            result = ask(q)
            display_result(result)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
