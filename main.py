import os
import sys
import warnings

# Suppress noisy third-party warnings before any imports
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import argparse
from dotenv import load_dotenv

from src.logger import get_logger, setup_logging
from src.utils.console import console, print_header, print_section

# Logging is configured before any other module uses it
load_dotenv()
setup_logging()

logger = get_logger(__name__)


# ── Interactive mode ──────────────────────────────────────────────────────────

def interactive_mode() -> None:
    """Prompt the user for an event and mode, then run the full battle."""
    print_header("🎯 AI PREDICTION BATTLE", "Intelligence Benchmark for Tech Predictions")

    console.print("\n[bold cyan]Enter a Polymarket event:[/bold cyan]")
    console.print("[dim]You can use: Event ID, URL, or Slug[/dim]")
    console.print("[dim]Example: 74949 or https://polymarket.com/event/... or event-slug[/dim]\n")

    event_input = console.input("[bold green]Event: [/bold green]").strip()
    if not event_input:
        console.print("[red]❌ No input provided.[/red]")
        return

    console.print("\n[bold cyan]Choose mode:[/bold cyan]")
    console.print("  1. Text debate (default)")
    console.print("  2. Voice debate (agents speak)")
    mode_input = console.input("\n[bold green]Mode (1/2): [/bold green]").strip() or "1"
    use_voice = mode_input == "2"

    logger.info("Interactive battle started.", extra={"event": event_input, "voice": use_voice})

    from src.services.prediction_service import PredictionService

    print_section("PHASE 1: Independent Research & Predictions")
    console.print("[dim]Each agent researches independently with NO cross-leakage.[/dim]\n")

    service = PredictionService()
    predictions, agent_predictions = service.run_battle(event_input)

    if not predictions:
        console.print("[red]❌ No predictions generated. Check your API keys.[/red]")
        return

    resolved_event_id = predictions[0].event_id

    if use_voice:
        print_section("PHASE 2: Voice Debate")
        from src.services.voice_debate_service import VoiceDebateService
        VoiceDebateService(service.all_agents).run_voice_debate(resolved_event_id, agent_predictions)
    else:
        print_section("PHASE 2: Text Debate")
        console.print("[dim]Agents defend locked positions. NO changes allowed.[/dim]\n")
        from src.services.debate_service import DebateService
        DebateService(service.all_agents).run_debate(resolved_event_id, agent_predictions, rounds=2)


# ── CLI mode ──────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) == 1:
        interactive_mode()
        return

    parser = argparse.ArgumentParser(
        description="AI Prediction Battle — Intelligence Benchmark for Tech Predictions"
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Full battle: predictions + debate")
    run_p.add_argument("event_id")
    run_p.add_argument("--rounds", type=int, default=2)
    run_p.add_argument("--voice", action="store_true")

    pred_p = sub.add_parser("predict", help="Predictions only")
    pred_p.add_argument("event_id")

    deb_p = sub.add_parser("debate", help="Text debate only")
    deb_p.add_argument("event_id")
    deb_p.add_argument("--rounds", type=int, default=2)

    voice_p = sub.add_parser("voice", help="Voice debate (V2)")
    voice_p.add_argument("event_id")

    sub.add_parser("discover", help="Discover trending tech events")
    sub.add_parser("test-voice", help="Test voice generation")

    args = parser.parse_args()
    logger.info("CLI command received.", extra={"command": args.command})

    if args.command == "run":
        from src.services.prediction_service import PredictionService
        from src.services.debate_service import DebateService

        print_header("🎯 AI PREDICTION BATTLE", "Intelligence Benchmark for Tech Predictions")
        print_section("PHASE 1: Independent Research & Predictions")
        console.print("[dim]Each agent researches independently with NO cross-leakage.[/dim]\n")

        service = PredictionService()
        predictions, agent_predictions = service.run_battle(args.event_id)

        if not predictions:
            console.print("[red]❌ No predictions generated. Check your API keys.[/red]")
            return

        resolved_event_id = predictions[0].event_id

        if args.voice:
            print_section("PHASE 2: Voice Debate")
            from src.services.voice_debate_service import VoiceDebateService
            VoiceDebateService(service.all_agents).run_voice_debate(resolved_event_id, agent_predictions)
        else:
            print_section("PHASE 2: Text Debate")
            console.print("[dim]Agents defend locked positions. NO changes allowed.[/dim]\n")
            DebateService(service.all_agents).run_debate(resolved_event_id, agent_predictions, rounds=args.rounds)

    elif args.command == "predict":
        from src.services.prediction_service import PredictionService

        print_header("🎯 AI PREDICTION ENGINE", "V0 — Prediction Only")
        service = PredictionService()
        predictions, _ = service.run_battle(args.event_id)
        if predictions:
            console.print("\n[green]✅ Predictions locked and saved.[/green]")

    elif args.command == "debate":
        from src.services.prediction_service import PredictionService
        from src.services.debate_service import DebateService

        print_header("⚔️ AI DEBATE ENGINE", "V1 — Panel Discussion")
        console.print("[dim]Running predictions first...[/dim]\n")
        service = PredictionService()
        predictions, agent_predictions = service.run_battle(args.event_id)
        if agent_predictions:
            DebateService(service.all_agents).run_debate(
                predictions[0].event_id, agent_predictions, rounds=args.rounds
            )
        else:
            console.print("[red]❌ No predictions to debate.[/red]")

    elif args.command == "voice":
        from src.services.prediction_service import PredictionService
        from src.services.voice_debate_service import VoiceDebateService

        print_header("🎙️ VOICE DEBATE", "V2 — AI Agents Speak")
        console.print("[dim]Running predictions first...[/dim]\n")
        service = PredictionService()
        predictions, agent_predictions = service.run_battle(args.event_id)
        if agent_predictions:
            VoiceDebateService(service.all_agents).run_voice_debate(
                predictions[0].event_id, agent_predictions
            )
        else:
            console.print("[red]❌ No predictions to debate.[/red]")

    elif args.command == "discover":
        from rich.table import Table
        from src.services.polymarket_service import PolymarketService

        print_header("🔍 DISCOVER EVENTS", "Trending AI/Tech Events on Polymarket")
        events = PolymarketService.search_tech_events(limit=10)
        if events:
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Event ID", style="dim")
            table.add_column("Title", max_width=50)
            table.add_column("Resolution Date")
            for e in events:
                table.add_row(e.event_id, e.title[:50], e.resolution_date[:10] if e.resolution_date else "N/A")
            console.print(table)
        else:
            console.print("[yellow]No events found.[/yellow]")

    elif args.command == "test-voice":
        from src.utils.voice import test_voice
        print_header("🎙️ VOICE TEST", "Testing Agent Voices")
        test_voice()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
