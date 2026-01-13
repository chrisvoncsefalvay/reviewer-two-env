"""Data models for Research Plan Generation Environment.

Defines the Action, Observation, and State types for the RL environment.
"""

from enum import Enum

from pydantic import Field
from openenv.core.env_server import Action, Observation, State


class ActionMode(str, Enum):
    """Action mode selection."""

    SUBMIT = "submit"
    RESET = "reset"


class DatasetSubset(str, Enum):
    """Available dataset subsets."""

    ML = "ml"
    ARXIV = "arxiv"
    PUBMED = "pubmed"


class DatasetSplit(str, Enum):
    """Available dataset splits."""

    TRAIN = "train"
    TEST = "test"


class ResearchPlanAction(Action):
    """Agent action: submit a research plan or reset the episode.

    Modes:
    - submit: Submit a research plan for evaluation (default)
    - reset: Reset the episode with optional subset/split selection

    For reset mode, if subset/split are not provided, the current
    configuration is preserved. Initial default is ml:train.
    """

    mode: ActionMode = Field(
        default=ActionMode.SUBMIT,
        description="Action mode: submit a plan or reset the episode",
    )

    # Submit mode field
    research_plan: str = Field(
        default="",
        description="The research plan to submit (required for submit mode)",
    )

    # Reset mode fields (optional - None means keep current value)
    subset: DatasetSubset | None = Field(
        default=None,
        description="Dataset subset for reset (ml, arxiv, pubmed). None keeps current.",
    )
    split: DatasetSplit | None = Field(
        default=None,
        description="Dataset split for reset (train, test). None keeps current.",
    )


class ResearchPlanObservation(Observation):
    """Observation returned after each step.

    Contains the research goal on reset (rubrics are hidden initially),
    and evaluation feedback after plan submission. Rubric hints are
    progressively revealed after free attempts.
    """

    # Task information (provided on reset)
    goal: str = Field(
        default="",
        description="The research task/goal to address",
    )
    rubric_count: int = Field(
        default=0,
        description="Number of evaluation criteria (criteria themselves are hidden)",
    )

    # Attempt tracking
    attempt_number: int = Field(
        default=0,
        description="Current attempt number for this episode",
    )
    max_attempts: int = Field(
        default=10,
        description="Maximum attempts allowed before episode ends",
    )
    free_attempts: int = Field(
        default=2,
        description="Number of attempts before rubric hints start appearing",
    )

    # Evaluation feedback (provided after step)
    criteria_met: int = Field(
        default=0,
        description="Number of criteria satisfied (out of rubric_count)",
    )
    length_score: float = Field(
        default=0.0,
        description="Score based on plan length appropriateness (0.0-1.0)",
    )
    format_score: float = Field(
        default=0.0,
        description="Score based on plan formatting/structure (0.0-1.0)",
    )
    feedback: str = Field(
        default="",
        description="General feedback on the submitted plan",
    )

    # Progressive rubric reveal
    revealed_hints: list[str] = Field(
        default_factory=list,
        description="Hints for revealed rubric criteria (in loose terms)",
    )
    hints_ignored_count: int = Field(
        default=0,
        description="Number of revealed hints not addressed in submission",
    )

    # Penalties
    attempt_penalty: float = Field(
        default=0.0,
        description="Cumulative penalty from multiple attempts",
    )
    compliance_penalty: float = Field(
        default=0.0,
        description="Penalty for ignoring revealed rubric hints",
    )
    total_penalty: float = Field(
        default=0.0,
        description="Total penalty (attempt + compliance)",
    )

    # Observation base fields: done, reward, metadata are inherited


class ResearchPlanState(State):
    """Public state returned to agents (rubrics hidden).

    This state is what agents see via get_state - it intentionally
    excludes the rubric criteria to prevent cheating.
    """

    # State base fields: episode_id, step_count are inherited

    current_goal: str = Field(
        default="",
        description="The current research goal being worked on",
    )
    rubric_count: int = Field(
        default=0,
        description="Number of evaluation criteria (hidden)",
    )
    article_id: str = Field(
        default="",
        description="Identifier of the source article",
    )
    subset: str = Field(
        default="ml",
        description="Dataset subset (ml, arxiv, pubmed)",
    )

    # Multi-turn tracking
    attempt_number: int = Field(
        default=0,
        description="Current attempt number",
    )
    max_attempts: int = Field(
        default=10,
        description="Maximum attempts allowed",
    )
    free_attempts: int = Field(
        default=2,
        description="Number of free attempts before hints",
    )
    revealed_hint_count: int = Field(
        default=0,
        description="Number of rubric hints currently revealed",
    )
    best_score: float = Field(
        default=0.0,
        description="Best score achieved so far",
    )
    cumulative_penalty: float = Field(
        default=0.0,
        description="Cumulative penalty from attempts and compliance",
    )
    total_reward: float = Field(
        default=0.0,
        description="Final reward for this episode",
    )


class InternalState:
    """Internal state for environment (not exposed to agents).

    Contains sensitive data like rubrics and reference solutions.
    """

    def __init__(
        self,
        episode_id: str = "",
        step_count: int = 0,
        current_goal: str = "",
        current_rubric: list[str] | None = None,
        reference_solution: str = "",
        article_id: str = "",
        subset: str = "ml",
        attempt_number: int = 0,
        max_attempts: int = 10,
        free_attempts: int = 2,
        submission_history: list[dict] | None = None,
        best_score: float = 0.0,
        cumulative_penalty: float = 0.0,
        compliance_penalty: float = 0.0,
        total_reward: float = 0.0,
        revealed_rubric_indices: list[int] | None = None,
    ):
        self.episode_id = episode_id
        self.step_count = step_count
        self.current_goal = current_goal
        self.current_rubric = current_rubric or []
        self.reference_solution = reference_solution
        self.article_id = article_id
        self.subset = subset
        self.attempt_number = attempt_number
        self.max_attempts = max_attempts
        self.free_attempts = free_attempts
        self.submission_history = submission_history or []
        self.best_score = best_score
        self.cumulative_penalty = cumulative_penalty
        self.compliance_penalty = compliance_penalty
        self.total_reward = total_reward
        self.revealed_rubric_indices = revealed_rubric_indices or []

    def to_public_state(self) -> ResearchPlanState:
        """Convert to public state (without rubric text)."""
        return ResearchPlanState(
            episode_id=self.episode_id,
            step_count=self.step_count,
            current_goal=self.current_goal,
            rubric_count=len(self.current_rubric),
            article_id=self.article_id,
            subset=self.subset,
            attempt_number=self.attempt_number,
            max_attempts=self.max_attempts,
            free_attempts=self.free_attempts,
            revealed_hint_count=len(self.revealed_rubric_indices),
            best_score=self.best_score,
            cumulative_penalty=self.cumulative_penalty + self.compliance_penalty,
            total_reward=self.total_reward,
        )
