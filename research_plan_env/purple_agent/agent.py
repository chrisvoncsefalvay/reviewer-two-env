"""Purple Agent implementation for research plan generation.

This agent receives research goals from the Green Agent evaluator,
uses Claude Sonnet to generate research plans, and iteratively
improves them based on feedback.
"""

import logging
import os
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Artifact,
    Message,
    TaskState,
    TextPart,
)

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert research planner. Your role is to create \
comprehensive, well-structured research plans based on given goals.

When creating a research plan, you should:
1. Analyse the research goal thoroughly
2. Break down the research into clear phases
3. Identify methodology and approaches
4. Define expected outcomes and success criteria
5. Consider potential challenges and mitigation strategies
6. Include timeline considerations and milestones

Your plans should be:
- Well-organised with clear sections and headers
- Comprehensive yet focused on the specific goal
- Practical and actionable
- Between 400-1500 words for optimal scoring

When you receive feedback, carefully analyse what criteria were not met and \
address those specific areas in your improved plan. Pay close attention to \
any revealed requirements or hints.
"""


class ResearchPlanGeneratorAgent:
    """Agent that generates research plans using Claude Sonnet.

    This Purple Agent:
    1. Receives a research goal from the Green Agent
    2. Uses Claude Sonnet to generate an initial research plan
    3. Receives feedback and iteratively improves the plan
    4. Returns progressively better plans until success or max attempts

    Attributes:
        model: The Claude model to use (default: claude-sonnet-4-20250514).
        max_tokens: Maximum tokens for each response.
    """

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = 4096,
    ):
        """Initialise the research plan generator.

        Args:
            model: Claude model name. Defaults to claude-sonnet-4-20250514.
            max_tokens: Maximum tokens for responses.
        """
        self.model = model or os.environ.get(
            "ANTHROPIC_MODEL", "claude-sonnet-4-20250514"
        )
        self.max_tokens = max_tokens
        self._client: anthropic.Anthropic | None = None
        self._conversation_history: list[dict[str, str]] = []

    def _get_client(self) -> anthropic.Anthropic:
        """Get or create the Anthropic client."""
        if not ANTHROPIC_AVAILABLE:
            raise RuntimeError(
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            )

        if self._client is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY environment variable not set"
                )
            self._client = anthropic.Anthropic(api_key=api_key)

        return self._client

    def reset(self) -> None:
        """Reset conversation history for a new evaluation session."""
        self._conversation_history = []

    async def run(
        self,
        message: Message,
        task_updater: TaskUpdater,
    ) -> None:
        """Process an incoming message and generate a research plan.

        Args:
            message: Incoming A2A message containing the research goal or feedback.
            task_updater: Task updater for reporting progress and results.
        """
        try:
            # Extract text from message - unwrap Part union types
            text_parts = []
            for p in message.parts:
                actual_part = getattr(p, 'root', p)
                if isinstance(actual_part, TextPart):
                    text_parts.append(actual_part)
            if not text_parts:
                raise ValueError("No text content in message")

            input_text = text_parts[0].text

            await task_updater.update_status(
                state=TaskState.working,
                message=Message(
                    kind="message",
                    role="agent",
                    parts=[TextPart(kind="text", text="Generating research plan with Claude Sonnet...")],
                    message_id="status-working",
                ),
            )

            # Generate research plan
            logger.info(f"\n{'='*60}\nPURPLE AGENT: Received prompt\n{'='*60}")
            logger.info(f"Input:\n{input_text[:500]}{'...' if len(input_text) > 500 else ''}")

            research_plan = await self._generate_plan(input_text)

            logger.info(f"\n{'='*60}\nPURPLE AGENT: Generated plan\n{'='*60}")
            logger.info(f"Output:\n{research_plan[:500]}{'...' if len(research_plan) > 500 else ''}")

            # Send the plan as response - add artifact then complete
            await task_updater.add_artifact(
                artifact_id="research_plan",
                name="research_plan",
                parts=[TextPart(kind="text", text=research_plan)],
            )
            await task_updater.complete()

        except Exception as e:
            logger.exception("Failed to generate research plan")
            await task_updater.failed(
                message=Message(
                    kind="message",
                    role="agent",
                    parts=[TextPart(kind="text", text=f"Generation failed: {e}")],
                    message_id="status-failed",
                ),
            )

    async def _generate_plan(self, input_text: str) -> str:
        """Generate a research plan using Claude Sonnet.

        Args:
            input_text: The research goal or feedback from the Green Agent.

        Returns:
            The generated research plan text.
        """
        client = self._get_client()

        # Add the input to conversation history
        self._conversation_history.append({
            "role": "user",
            "content": input_text,
        })

        # Generate response
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=self._conversation_history,
        )

        # Extract response text
        assistant_message = response.content[0].text

        # Add to conversation history for multi-turn
        self._conversation_history.append({
            "role": "assistant",
            "content": assistant_message,
        })

        return assistant_message


class Executor(AgentExecutor):
    """A2A executor that manages Purple Agent instances and task processing."""

    def __init__(self, model: str | None = None):
        """Initialise the executor.

        Args:
            model: Optional Claude model name override.
        """
        self.model = model
        self.agents: dict[str, ResearchPlanGeneratorAgent] = {}

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute an A2A request.

        Args:
            context: Request context with task and message info.
            event_queue: Queue for sending events back to client.
        """
        # Get task info - task is created by DefaultRequestHandler
        task = context.current_task

        # Check terminal state if task exists
        if task and task.status and task.status.state in (
            TaskState.completed,
            TaskState.canceled,
            TaskState.failed,
            TaskState.rejected,
        ):
            raise ValueError(
                f"Task in terminal state: {task.status.state}"
            )

        # Get or create agent for this context
        context_id = context.context_id
        if context_id not in self.agents:
            self.agents[context_id] = ResearchPlanGeneratorAgent(
                model=self.model,
            )

        agent = self.agents[context_id]

        # Use task_id and context_id from context
        task_id = context.task_id or (task.id if task else "unknown")
        ctx_id = context.context_id
        task_updater = TaskUpdater(event_queue, task_id, ctx_id)

        # Run the agent
        await agent.run(context.message, task_updater)

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Cancel is not supported."""
        raise NotImplementedError("Cancel not supported")
