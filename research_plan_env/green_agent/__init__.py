"""Green Agent wrapper for AgentBeats A2A protocol.

This module provides A2A-compatible evaluation endpoints for the
Research Plan Generation Environment, allowing Purple Agents to
be evaluated on research plan quality.
"""

from research_plan_env.green_agent.models import EvalRequest, EvalResult

__all__ = ["EvalRequest", "EvalResult"]
