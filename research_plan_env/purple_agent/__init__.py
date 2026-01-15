"""Purple Agent for research plan generation.

This A2A-compatible agent uses Claude Sonnet to generate and iteratively
improve research plans based on feedback from the Green Agent evaluator.
"""

from research_plan_env.purple_agent.agent import (
    Executor,
    ResearchPlanGeneratorAgent,
)
from research_plan_env.purple_agent.server import create_app, main

__all__ = [
    "Executor",
    "ResearchPlanGeneratorAgent",
    "create_app",
    "main",
]
