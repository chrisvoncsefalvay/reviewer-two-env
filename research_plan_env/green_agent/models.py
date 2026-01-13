"""Data models for Green Agent evaluation protocol.

Defines the request and response formats for A2A-based evaluation.
"""

from typing import Any

from pydantic import BaseModel, HttpUrl


class EvalRequest(BaseModel):
    """Evaluation request from AgentBeats platform.

    Attributes:
        participants: Mapping of role names to agent endpoint URLs.
            Expected role: "purple" for the agent being evaluated.
        config: Configuration parameters for the evaluation.
            - subset: Dataset subset (ml, arxiv, pubmed). Default: ml
            - split: Dataset split (train, test). Default: train
            - max_attempts: Maximum attempts allowed. Default: 10
            - task_index: Specific task index, or None for random.
    """

    participants: dict[str, HttpUrl]
    config: dict[str, Any]


class EvalResult(BaseModel):
    """Evaluation result returned to AgentBeats platform.

    Attributes:
        winner: Role of the winning participant (or "none" if failed).
        detail: Detailed evaluation breakdown including scores and feedback.
    """

    winner: str
    detail: dict[str, Any]


class TaskConfig(BaseModel):
    """Internal configuration for a single evaluation task.

    Parsed from EvalRequest.config with defaults applied.
    """

    subset: str = "ml"
    split: str = "train"
    max_attempts: int = 10
    free_attempts: int = 2
    task_index: int | None = None
    success_threshold: float = 0.8

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TaskConfig":
        """Create TaskConfig from raw config dict with defaults."""
        return cls(
            subset=config.get("subset", "ml"),
            split=config.get("split", "train"),
            max_attempts=config.get("max_attempts", 10),
            free_attempts=config.get("free_attempts", 2),
            task_index=config.get("task_index"),
            success_threshold=config.get("success_threshold", 0.8),
        )
