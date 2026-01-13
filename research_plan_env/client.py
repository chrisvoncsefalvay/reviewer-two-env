"""Client for interacting with the Research Plan Generation Environment.

Provides a Python client that communicates with the environment server
using the OpenEnv HTTP protocol.
"""

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from research_plan_env.models import (
    ResearchPlanAction,
    ResearchPlanObservation,
    ResearchPlanState,
)


class ResearchPlanEnv(EnvClient[ResearchPlanAction, ResearchPlanObservation, ResearchPlanState]):
    """Client for the Research Plan Generation Environment.

    Usage:
        ```python
        from research_plan_env import ResearchPlanEnv, ResearchPlanAction

        # Connect to the environment server
        env = ResearchPlanEnv(base_url="http://localhost:7860")

        # Reset to get a new research task
        obs = env.reset()
        print(f"Goal: {obs.goal}")
        print(f"Rubric: {obs.rubric}")

        # Generate and submit a research plan
        action = ResearchPlanAction(
            research_plan="My detailed research plan..."
        )
        result = env.step(action)

        print(f"Reward: {result.reward}")
        print(f"Feedback: {result.observation.feedback}")
        ```
    """

    def _step_payload(self, action: ResearchPlanAction) -> dict:
        """Convert action to JSON payload for the server.

        Args:
            action: The research plan action to convert.

        Returns:
            Dictionary payload for the HTTP request.
        """
        return action.model_dump()

    def _parse_result(self, payload: dict) -> StepResult[ResearchPlanObservation]:
        """Parse server response into a StepResult.

        Args:
            payload: The JSON response from the server.

        Returns:
            StepResult containing the observation, reward, and done flag.
        """
        # Extract observation data
        obs_data = payload.get("observation", payload)

        # Build observation
        observation = ResearchPlanObservation(
            done=obs_data.get("done", False),
            reward=obs_data.get("reward", 0.0),
            goal=obs_data.get("goal", ""),
            rubric=obs_data.get("rubric", []),
            rubric_scores=obs_data.get("rubric_scores", {}),
            length_score=obs_data.get("length_score", 0.0),
            format_score=obs_data.get("format_score", 0.0),
            feedback=obs_data.get("feedback", ""),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", observation.reward),
            done=payload.get("done", observation.done),
        )

    def _parse_state(self, payload: dict) -> ResearchPlanState:
        """Parse state response from the server.

        Args:
            payload: The JSON state response.

        Returns:
            ResearchPlanState object.
        """
        return ResearchPlanState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            current_goal=payload.get("current_goal", ""),
            current_rubric=payload.get("current_rubric", []),
            reference_solution=payload.get("reference_solution", ""),
            article_id=payload.get("article_id", ""),
            subset=payload.get("subset", "ml"),
            submitted_plan=payload.get("submitted_plan"),
            total_reward=payload.get("total_reward", 0.0),
        )

    def get_task(self) -> tuple[str, list[str]]:
        """Convenience method to get the current task.

        Returns:
            Tuple of (goal, rubric) for the current task.
        """
        state = self.state
        return state.current_goal, state.current_rubric

    def submit_plan(self, plan: str) -> StepResult[ResearchPlanObservation]:
        """Convenience method to submit a research plan.

        Args:
            plan: The research plan text to submit.

        Returns:
            StepResult with evaluation.
        """
        action = ResearchPlanAction(research_plan=plan)
        return self.step(action)
