"""Test script to run an evaluation between green and purple agents.

This script sends an evaluation request to the green agent, which then
communicates with the purple agent to evaluate research plan generation.
"""

import asyncio
import json
import logging

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Task, TextPart

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

GREEN_AGENT_URL = "http://127.0.0.1:9009"
PURPLE_AGENT_URL = "http://127.0.0.1:9010"


async def run_evaluation():
    """Run a single evaluation request."""
    # Create the evaluation request
    eval_request = {
        "participants": {
            "purple": PURPLE_AGENT_URL,
        },
        "config": {
            "subset": "ml",
            "split": "train",
            "max_attempts": 3,  # Keep it short for testing
            "free_attempts": 1,
            "success_threshold": 0.7,
            "task_index": 0,  # Use first task for reproducibility
        },
    }

    logger.info("=" * 60)
    logger.info("STARTING EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Green Agent: {GREEN_AGENT_URL}")
    logger.info(f"Purple Agent: {PURPLE_AGENT_URL}")
    logger.info(f"Config: {json.dumps(eval_request['config'], indent=2)}")
    logger.info("=" * 60)

    async with httpx.AsyncClient(timeout=600.0) as http_client:
        # Resolve green agent card
        resolver = A2ACardResolver(http_client, GREEN_AGENT_URL)
        card = await resolver.get_agent_card()
        logger.info(f"Connected to Green Agent: {card.name}")

        # Create client using ClientFactory
        config = ClientConfig(httpx_client=http_client, streaming=True)
        factory = ClientFactory(config)
        client = factory.create(card)

        # Create message with eval request
        message = Message(
            kind="message",
            role="user",
            parts=[TextPart(kind="text", text=json.dumps(eval_request))],
            message_id="test-eval-001",
            context_id="test-context-001",
        )

        logger.info("\nSending evaluation request to Green Agent...")
        logger.info("-" * 60)

        # Send message and process streaming response
        response = client.send_message(message)

        result = None
        async for event in response:
            if isinstance(event, Message):
                text_parts = [
                    p.text for p in event.parts if hasattr(p, "text")
                ]
                if text_parts:
                    logger.info(f"\n[Message] {' '.join(text_parts)}")
            elif isinstance(event, tuple) and len(event) == 2:
                task, _ = event
                if isinstance(task, Task):
                    if task.status:
                        state = task.status.state
                        msg = ""
                        if task.status.message and task.status.message.parts:
                            msg_parts = [
                                p.text
                                for p in task.status.message.parts
                                if hasattr(p, "text")
                            ]
                            msg = " ".join(msg_parts)
                        logger.info(f"[Task Status] {state}: {msg}")

                    if task.artifacts:
                        for artifact in task.artifacts:
                            logger.info(f"\n[Artifact: {artifact.name}]")
                            for part in artifact.parts:
                                if hasattr(part, "data"):
                                    result = part.data
                                    logger.info(json.dumps(
                                        part.data, indent=2, default=str
                                    ))
                                elif hasattr(part, "text"):
                                    text = part.text
                                    if len(text) > 500:
                                        text = text[:500] + "..."
                                    logger.info(text)

        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 60)

        if result:
            logger.info("\nFinal Result:")
            logger.info(f"  Winner: {result.get('winner', 'N/A')}")
            detail = result.get("detail", {})
            logger.info(f"  Passed: {detail.get('passed', 'N/A')}")
            logger.info(f"  Best Score: {detail.get('best_score', 'N/A')}")
            logger.info(f"  Best Attempt: {detail.get('best_attempt', 'N/A')}")
            logger.info(
                f"  Total Attempts: {detail.get('total_attempts', 'N/A')}"
            )

            if "scores" in detail:
                logger.info("\n  Score History:")
                for score in detail["scores"]:
                    logger.info(
                        f"    Attempt {score['attempt']}: "
                        f"reward={score['reward']:.3f}, "
                        f"criteria={score['criteria_met']}"
                        f"/{score['rubric_count']}"
                    )

        return result


if __name__ == "__main__":
    result = asyncio.run(run_evaluation())
