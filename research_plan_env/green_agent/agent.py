"""Green Agent implementation for research plan evaluation.

This agent receives evaluation requests from the AgentBeats platform,
communicates with Purple Agents to collect their research plans,
and evaluates them using the reward calculator.
"""

import json
import logging
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Artifact,
    DataPart,
    InvalidRequestError,
    Message,
    TaskState,
    TextPart,
)

from research_plan_env.green_agent.messenger import Messenger
from research_plan_env.green_agent.models import (
    EvalRequest,
    EvalResult,
    TaskConfig,
)
from research_plan_env.server.environment import ResearchPlanEnvironment

logger = logging.getLogger(__name__)


class ResearchPlanEvaluatorAgent:
    """Agent that evaluates Purple Agents on research plan generation.

    This Green Agent:
    1. Receives an EvalRequest with Purple Agent URL and config
    2. Initialises a research plan task from the dataset
    3. Sends the goal to the Purple Agent
    4. Receives their research plan
    5. Optionally allows multiple turns with feedback
    6. Returns evaluation scores

    Attributes:
        required_roles: Roles that must be present in participants.
        required_config_keys: Config keys that must be provided.
    """

    required_roles: list[str] = ["purple"]
    required_config_keys: list[str] = []

    def __init__(self):
        """Initialise the evaluator agent."""
        self.messenger = Messenger(timeout=300.0)
        self._env: ResearchPlanEnvironment | None = None

    def _get_environment(self, config: TaskConfig) -> ResearchPlanEnvironment:
        """Get or create the environment with given config."""
        if self._env is None:
            self._env = ResearchPlanEnvironment(
                subset=config.subset,
                split=config.split,
                max_attempts=config.max_attempts,
                free_attempts=config.free_attempts,
            )
        return self._env

    async def run(
        self,
        message: Message,
        task_updater: TaskUpdater,
    ) -> None:
        """Execute the evaluation workflow.

        Args:
            message: Incoming A2A message containing EvalRequest JSON.
            task_updater: Task updater for reporting progress and results.
        """
        try:
            # Parse the evaluation request
            text_parts = [p for p in message.parts if isinstance(p, TextPart)]
            if not text_parts:
                raise InvalidRequestError("No text content in message")

            request_json = text_parts[0].text
            try:
                request_data = json.loads(request_json)
                eval_request = EvalRequest(**request_data)
            except (json.JSONDecodeError, ValueError) as e:
                raise InvalidRequestError(f"Invalid EvalRequest: {e}") from e

            # Validate required roles
            for role in self.required_roles:
                if role not in eval_request.participants:
                    raise InvalidRequestError(f"Missing required role: {role}")

            # Parse configuration
            config = TaskConfig.from_config(eval_request.config)

            # Get Purple Agent URL
            purple_url = str(eval_request.participants["purple"])

            # Run the evaluation
            result = await self._evaluate_purple_agent(
                purple_url=purple_url,
                config=config,
                task_updater=task_updater,
            )

            # Complete the task with result
            task_updater.complete(
                status_message="Evaluation complete",
                artifacts=[
                    Artifact(
                        name="eval_result",
                        parts=[
                            DataPart(
                                kind="data",
                                data=result.model_dump(),
                            ),
                        ],
                    ),
                ],
            )

        except InvalidRequestError:
            raise
        except Exception as e:
            logger.exception("Evaluation failed")
            task_updater.failed(f"Evaluation failed: {e}")

    async def _evaluate_purple_agent(
        self,
        purple_url: str,
        config: TaskConfig,
        task_updater: TaskUpdater,
    ) -> EvalResult:
        """Run the full evaluation loop with a Purple Agent.

        Args:
            purple_url: URL of the Purple Agent to evaluate.
            config: Task configuration.
            task_updater: For progress updates.

        Returns:
            EvalResult with winner and detailed scores.
        """
        env = self._get_environment(config)

        # Reset environment with optional specific task
        reset_kwargs: dict[str, Any] = {
            "subset": config.subset,
            "split": config.split,
        }
        if config.task_index is not None:
            reset_kwargs["index"] = config.task_index

        obs = env.reset(**reset_kwargs)

        task_updater.update_status(
            state=TaskState.working,
            message=f"Starting evaluation: {obs.rubric_count} criteria",
        )

        # Build initial prompt for Purple Agent
        initial_prompt = self._build_initial_prompt(obs)

        best_score = 0.0
        best_attempt = 0
        all_scores: list[dict[str, Any]] = []

        # Multi-turn evaluation loop
        attempt = 0
        while not obs.done and attempt < config.max_attempts:
            attempt += 1

            task_updater.update_status(
                state=TaskState.working,
                message=f"Attempt {attempt}/{config.max_attempts}",
            )

            # Send prompt to Purple Agent
            if attempt == 1:
                prompt = initial_prompt
            else:
                prompt = self._build_feedback_prompt(obs)

            try:
                response = await self.messenger.talk_to_agent(
                    agent_url=purple_url,
                    text=prompt,
                )
            except Exception as e:
                logger.warning(f"Purple Agent communication failed: {e}")
                break

            research_plan = response.get("text", "")
            if not research_plan:
                logger.warning("Purple Agent returned empty response")
                break

            # Submit to environment for evaluation
            from research_plan_env.models import ActionMode, ResearchPlanAction

            action = ResearchPlanAction(
                mode=ActionMode.SUBMIT,
                research_plan=research_plan,
            )
            obs = env.step(action)

            # Track scores
            score_data = {
                "attempt": attempt,
                "reward": obs.reward,
                "criteria_met": obs.criteria_met,
                "rubric_count": obs.rubric_count,
                "length_score": obs.length_score,
                "format_score": obs.format_score,
                "total_penalty": obs.total_penalty,
            }
            all_scores.append(score_data)

            if obs.reward > best_score:
                best_score = obs.reward
                best_attempt = attempt

            # Check for success
            if obs.reward >= config.success_threshold:
                break

        # Determine winner
        passed = best_score >= config.success_threshold
        winner = "purple" if passed else "none"

        return EvalResult(
            winner=winner,
            detail={
                "passed": passed,
                "best_score": best_score,
                "best_attempt": best_attempt,
                "total_attempts": attempt,
                "success_threshold": config.success_threshold,
                "scores": all_scores,
                "config": {
                    "subset": config.subset,
                    "split": config.split,
                    "max_attempts": config.max_attempts,
                },
            },
        )

    def _build_initial_prompt(self, obs: Any) -> str:
        """Build the initial prompt for the Purple Agent."""
        return f"""You are participating in a research plan generation evaluation.

## Research Goal

{obs.goal}

## Instructions

Write a comprehensive research plan that addresses this goal. Your plan should:
- Be well-structured with clear sections
- Address all aspects of the research goal
- Include methodology, expected outcomes, and evaluation criteria
- Be between 400-1500 words for optimal scoring

You have {obs.max_attempts} attempts. The first {obs.free_attempts} are free, \
then hints will be progressively revealed.

Submit your research plan now."""

    def _build_feedback_prompt(self, obs: Any) -> str:
        """Build a feedback prompt for subsequent attempts."""
        parts = [
            "## Feedback on Previous Attempt",
            "",
            f"Score: {obs.reward:.2f}",
            f"Criteria met: {obs.criteria_met}/{obs.rubric_count}",
            "",
            obs.feedback,
        ]

        if obs.revealed_hints:
            parts.extend([
                "",
                "## Revealed Requirements",
                "Address these specific areas:",
            ])
            for i, hint in enumerate(obs.revealed_hints, 1):
                parts.append(f"{i}. {hint}")

            if obs.hints_ignored_count > 0:
                parts.append(
                    f"\nWarning: {obs.hints_ignored_count} hints were not "
                    "addressed. Ignoring hints doubles the penalty."
                )

        parts.extend([
            "",
            f"Attempts remaining: {obs.max_attempts - obs.attempt_number}",
            "",
            "Submit an improved research plan:",
        ])

        return "\n".join(parts)


class Executor(AgentExecutor):
    """A2A executor that manages agent instances and task processing."""

    def __init__(self):
        """Initialise the executor."""
        self.agents: dict[str, ResearchPlanEvaluatorAgent] = {}

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
        # Get or create task
        task = context.current_task
        if task is None:
            task = context.new_task(context.message)
            event_queue.enqueue_event(task)

        # Check terminal state
        if task.status and task.status.state in (
            TaskState.completed,
            TaskState.canceled,
            TaskState.failed,
            TaskState.rejected,
        ):
            raise InvalidRequestError(
                f"Task in terminal state: {task.status.state}"
            )

        # Get or create agent for this context
        context_id = context.context_id
        if context_id not in self.agents:
            self.agents[context_id] = ResearchPlanEvaluatorAgent()

        agent = self.agents[context_id]
        task_updater = TaskUpdater(event_queue, task.id, task.contextId)

        # Run the agent
        await agent.run(context.message, task_updater)

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Cancel is not supported."""
        from a2a.types import UnsupportedOperationError
        raise UnsupportedOperationError("Cancel not supported")
