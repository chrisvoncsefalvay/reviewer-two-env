"""A2A server entry point for the Research Plan Green Agent.

This module creates and runs an A2A-compliant HTTP server that
exposes the research plan evaluation agent to the AgentBeats platform.
"""

import argparse
import logging

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentSkill

from research_plan_env.green_agent.agent import Executor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_agent_card(host: str, port: int, card_url: str | None) -> AgentCard:
    """Create the A2A agent card describing this Green Agent.

    Args:
        host: Server host address.
        port: Server port.
        card_url: Optional external URL for the agent card.

    Returns:
        AgentCard with capability and skill definitions.
    """
    base_url = card_url or f"http://{host}:{port}"

    return AgentCard(
        name="Research Plan Evaluator",
        description=(
            "Green Agent that evaluates Purple Agents on research plan "
            "generation tasks. Uses the facebook/research-plan-gen dataset "
            "with LLM-based rubric evaluation and progressive hint reveal."
        ),
        url=base_url,
        version="1.0.0",
        capabilities=AgentCapabilities(
            streaming=True,
            pushNotifications=False,
        ),
        defaultInputModes=["text"],
        defaultOutputModes=["text", "data"],
        skills=[
            AgentSkill(
                id="evaluate_research_plan",
                name="Evaluate Research Plan",
                description=(
                    "Evaluate a Purple Agent's ability to generate research "
                    "plans. Sends a research goal, receives plans, and scores "
                    "them against hidden rubric criteria."
                ),
                tags=["evaluation", "research", "nlp", "benchmark"],
                examples=[
                    "Evaluate agent on ML research tasks",
                    "Run research plan benchmark with arxiv subset",
                ],
            ),
        ],
    )


def create_app(
    host: str = "127.0.0.1",
    port: int = 9009,
    card_url: str | None = None,
) -> A2AStarletteApplication:
    """Create the A2A Starlette application.

    Args:
        host: Server host address.
        port: Server port.
        card_url: Optional external URL for the agent card.

    Returns:
        Configured A2A Starlette application.
    """
    agent_card = create_agent_card(host, port, card_url)
    task_store = InMemoryTaskStore()
    executor = Executor()

    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
    )

    return A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=handler,
    )


def main() -> None:
    """Main entry point for the Green Agent server."""
    parser = argparse.ArgumentParser(
        description="Research Plan Evaluator Green Agent"
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
        default=9009,
        help="Port to bind to (default: 9009)",
    )
    parser.add_argument(
        "--card-url",
        type=str,
        default=None,
        help="External URL for agent card advertisement",
    )

    args = parser.parse_args()

    logger.info(
        f"Starting Research Plan Evaluator Green Agent on "
        f"{args.host}:{args.port}"
    )

    app = create_app(
        host=args.host,
        port=args.port,
        card_url=args.card_url,
    )

    uvicorn.run(
        app.build(),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
