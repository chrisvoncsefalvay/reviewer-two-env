"""A2A server entry point for the Research Plan Purple Agent.

This module creates and runs an A2A-compliant HTTP server that
exposes the research plan generation agent using Claude Sonnet.
"""

import argparse
import logging
import os

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentSkill

from research_plan_env.purple_agent.agent import Executor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_agent_card(host: str, port: int, card_url: str | None) -> AgentCard:
    """Create the A2A agent card describing this Purple Agent.

    Args:
        host: Server host address.
        port: Server port.
        card_url: Optional external URL for the agent card.

    Returns:
        AgentCard with capability and skill definitions.
    """
    base_url = card_url or f"http://{host}:{port}"

    return AgentCard(
        name="Research Plan Generator (Claude Sonnet)",
        description=(
            "Purple Agent that generates comprehensive research plans using "
            "Claude Sonnet. Responds to research goals with structured plans "
            "and iteratively improves based on evaluator feedback."
        ),
        url=base_url,
        version="1.0.0",
        capabilities=AgentCapabilities(
            streaming=True,
            pushNotifications=False,
        ),
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        skills=[
            AgentSkill(
                id="generate_research_plan",
                name="Generate Research Plan",
                description=(
                    "Generate a comprehensive research plan for a given goal. "
                    "Uses Claude Sonnet to create well-structured, actionable "
                    "plans that address methodology, outcomes, and evaluation "
                    "criteria. Iteratively improves based on feedback."
                ),
                tags=["research", "planning", "nlp", "claude", "sonnet"],
                examples=[
                    "Create a research plan for studying ML model robustness",
                    "Generate a plan for investigating neural network interpretability",
                ],
            ),
        ],
    )


def create_app(
    host: str = "127.0.0.1",
    port: int = 9010,
    card_url: str | None = None,
    model: str | None = None,
) -> A2AStarletteApplication:
    """Create the A2A Starlette application.

    Args:
        host: Server host address.
        port: Server port.
        card_url: Optional external URL for the agent card.
        model: Optional Claude model name override.

    Returns:
        Configured A2A Starlette application.
    """
    agent_card = create_agent_card(host, port, card_url)
    task_store = InMemoryTaskStore()
    executor = Executor(model=model)

    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
    )

    return A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=handler,
    )


def main() -> None:
    """Main entry point for the Purple Agent server."""
    parser = argparse.ArgumentParser(
        description="Research Plan Generator Purple Agent (Claude Sonnet)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9010,
        help="Port to bind to (default: 9010)",
    )
    parser.add_argument(
        "--card-url",
        type=str,
        default=None,
        help="External URL for agent card advertisement",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Claude model to use (default: claude-sonnet-4-20250514)",
    )

    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning(
            "ANTHROPIC_API_KEY not set. Set this environment variable "
            "before sending requests to the agent."
        )

    logger.info(
        f"Starting Research Plan Generator Purple Agent on "
        f"{args.host}:{args.port}"
    )
    if args.model:
        logger.info(f"Using model: {args.model}")

    app = create_app(
        host=args.host,
        port=args.port,
        card_url=args.card_url,
        model=args.model,
    )

    uvicorn.run(
        app.build(),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
