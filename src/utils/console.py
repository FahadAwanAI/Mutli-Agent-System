from typing import List, Dict, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


def print_header(title: str, subtitle: str = "") -> None:
    console.print(Panel(
        f"[bold white]{title}[/bold white]\n[dim]{subtitle}[/dim]",
        border_style="cyan",
        expand=False,
    ))


def print_section(title: str) -> None:
    console.print(f"\n[bold cyan]── {title} ──[/bold cyan]")


def print_event(title: str, event_id: str, description: str, rules: str, date: str) -> None:
    console.print(Panel(
        f"[bold]{title}[/bold]\n"
        f"[dim]ID: {event_id} | Resolves: {date[:10] if date else 'N/A'}[/dim]\n\n"
        f"{description[:300]}{'...' if len(description) > 300 else ''}\n\n"
        f"[dim]Rules: {rules[:200]}{'...' if len(rules) > 200 else ''}[/dim]",
        title="📋 Event",
        border_style="blue",
    ))


def print_agents_status(active: List[str], inactive: List[str]) -> None:
    console.print("\n[bold]Agent Status:[/bold]")
    for name in active:
        console.print(f"  [green]✓[/green] {name}")
    for name in inactive:
        console.print(f"  [red]✗[/red] {name} [dim](no API key)[/dim]")
    console.print()


def print_prediction(
    agent_name: str,
    prediction: str,
    probability: float,
    rationale: str,
    key_facts: List[Dict],
) -> None:
    color = "green" if prediction == "YES" else "red"
    facts_text = "\n".join(
        f"  • {f['claim']} [dim]({f['source']})[/dim]" for f in key_facts
    )
    console.print(Panel(
        f"[{color}][bold]{prediction}[/bold] — {probability * 100:.0f}% confidence[/{color}]\n\n"
        f"[italic]{rationale}[/italic]\n\n"
        f"[bold]Key Facts:[/bold]\n{facts_text}",
        title=f"🤖 {agent_name}",
        border_style=color,
    ))


def print_predictions_table(predictions: List[Dict]) -> None:
    table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
    table.add_column("Agent")
    table.add_column("Prediction")
    table.add_column("Probability")

    for pred in predictions:
        color = "green" if pred["prediction"] == "YES" else "red"
        table.add_row(
            pred["agent_name"],
            f"[{color}]{pred['prediction']}[/{color}]",
            f"{pred['probability'] * 100:.0f}%",
        )
    console.print(table)


def print_moderator(message: str, is_intro: bool = False) -> None:
    style = "yellow" if is_intro else "dim yellow"
    console.print(f"\n[{style}]⚖️  Moderator: {message}[/{style}]\n")


def print_error(message: str) -> None:
    console.print(f"[red]❌ {message}[/red]")
