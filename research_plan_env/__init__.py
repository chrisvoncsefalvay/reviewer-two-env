"""Research Plan Generation Environment using OpenEnv."""

# Lazy imports to avoid requiring openenv for subpackages that don't need it
# (e.g., purple_agent only needs a2a-sdk, not the full openenv stack)

__all__ = [
    "ResearchPlanAction",
    "ResearchPlanObservation",
    "ResearchPlanState",
    "ResearchPlanEnv",
]


def __getattr__(name: str):
    """Lazy import attributes to avoid loading openenv unnecessarily."""
    if name in ("ResearchPlanAction", "ResearchPlanObservation", "ResearchPlanState"):
        from research_plan_env.models import (
            ResearchPlanAction,
            ResearchPlanObservation,
            ResearchPlanState,
        )
        return {
            "ResearchPlanAction": ResearchPlanAction,
            "ResearchPlanObservation": ResearchPlanObservation,
            "ResearchPlanState": ResearchPlanState,
        }[name]
    elif name == "ResearchPlanEnv":
        from research_plan_env.client import ResearchPlanEnv
        return ResearchPlanEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
