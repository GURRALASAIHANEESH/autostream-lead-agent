import logging
import sys

from agent.orchestrator import run_turn
from agent.state_manager import AgentState, create_initial_state
from config import get_settings

logger: logging.Logger = logging.getLogger(__name__)


def _print_header() -> None:
    """Print the CLI startup header."""
    print("AutoStream Agent — type 'quit' or 'exit' to stop")
    print("\u2500" * 50)


def _print_lead_captured_footer() -> None:
    """Print the session-complete footer after lead capture."""
    print("\u2500" * 50)
    print("Session complete. Lead successfully captured.")


def main() -> None:
    """Interactive CLI loop that drives the agent one turn at a time."""

    settings = get_settings()
    logger.info(
        "Starting CLI session with provider=%s, model=%s",
        settings.LLM_PROVIDER,
        settings.LLM_MODEL,
    )

    _print_header()

    state: AgentState = create_initial_state()

    # Generate the initial agent greeting so the user sees it immediately.
    try:
        greeting, state = run_turn("hello", state)
    except Exception:
        logger.exception("Failed to generate initial greeting.")
        print("Agent: Hello! I'm the AutoStream assistant. How can I help you today?")
        greeting = ""

    if greeting:
        print(f"Agent: {greeting}")
    print()

    try:
        while True:
            try:
                user_input: str = input("You: ")
            except EOFError:
                print("\nSession ended.")
                break

            if not user_input.strip():
                continue

            if user_input.strip().lower() in ("quit", "exit"):
                print("Goodbye.")
                break

            try:
                response, state = run_turn(user_input.strip(), state)
            except Exception:
                logger.exception("Error during run_turn for input: %.120s", user_input)
                print("Agent: Sorry, something went wrong. Please try again.")
                print()
                continue

            print(f"Agent: {response}")
            print()

            if state["lead_captured"]:
                _print_lead_captured_footer()
                break

    except KeyboardInterrupt:
        print("\nSession ended.")
        sys.exit(0)


if __name__ == "__main__":
    main()