"""Research Plan Generation Environment implementation.

This environment presents research tasks from the facebook/research-plan-gen
dataset and evaluates agent-generated research plans against rubric criteria.
"""

import random
import uuid
from typing import Optional

import torch
from datasets import load_dataset
from openenv.core.env_server import Environment

from research_plan_env.models import (
    ActionMode,
    InternalState,
    ResearchPlanAction,
    ResearchPlanObservation,
    ResearchPlanState,
)
from research_plan_env.server.reward_calculator import RewardCalculator


class ResearchPlanEnvironment(Environment):
    """Environment for training agents to generate research plans.

    The environment:
    1. On reset: samples a research task (goal only - rubric is hidden)
    2. On step: receives a research plan and evaluates it against hidden rubric
    3. After free attempts: progressively reveals rubric hints
    4. Penalises ignoring revealed hints (double penalty)

    This is a multi-turn episodic environment - agents can submit multiple
    attempts to improve their score, but each attempt incurs a penalty.
    """

    # Penalty configuration
    ATTEMPT_PENALTY_BASE = 0.02  # Base penalty per attempt
    ATTEMPT_PENALTY_GROWTH = 1.5  # Penalty growth factor
    COMPLIANCE_PENALTY_MULTIPLIER = 2.0  # Double penalty for ignoring hints
    SUCCESS_THRESHOLD = 0.8  # Score threshold to end episode successfully
    FREE_ATTEMPTS = 2  # Attempts before hints start appearing

    def __init__(
        self,
        subset: str = "ml",
        split: str = "train",
        model_name: str = "google/flan-t5-small",
        seed: Optional[int] = None,
        max_attempts: int = 10,
        free_attempts: int = 2,
    ):
        """Initialise the environment.

        Args:
            subset: Dataset subset to use ('ml', 'arxiv', or 'pubmed').
            split: Dataset split to use ('train' or 'test').
            model_name: HuggingFace model for reward calculation.
            seed: Random seed for reproducibility.
            max_attempts: Maximum attempts allowed per episode.
            free_attempts: Attempts before rubric hints start appearing.
        """
        self.subset = subset
        self.split = split
        self.model_name = model_name
        self.max_attempts = max_attempts
        self.free_attempts = free_attempts

        # Load the dataset
        self._dataset = load_dataset(
            "facebook/research-plan-gen", subset
        )[split]
        self._dataset_size = len(self._dataset)

        # Initialise reward calculator
        self._reward_calculator = RewardCalculator(
            model_name=model_name,
        )

        # Random state
        if seed is not None:
            random.seed(seed)
        self._rng = random.Random(seed)

        # Episode state (internal - contains rubrics)
        self._internal_state: Optional[InternalState] = None
        self._episode_count = 0

    def _calculate_attempt_penalty(self, attempt_number: int) -> float:
        """Calculate cumulative penalty based on attempt number."""
        if attempt_number <= 1:
            return 0.0
        return self.ATTEMPT_PENALTY_BASE * (
            self.ATTEMPT_PENALTY_GROWTH ** (attempt_number - 1) - 1
        )

    def _get_rubrics_to_reveal(self, attempt: int) -> list[int]:
        """Calculate which rubric indices should be revealed by this attempt.

        After free_attempts, distribute rubric reveals evenly across
        remaining attempts.
        """
        if self._internal_state is None:
            return []

        rubric_count = len(self._internal_state.current_rubric)
        max_att = self._internal_state.max_attempts
        free_att = self._internal_state.free_attempts

        # No reveals during free attempts
        if attempt <= free_att:
            return []

        # Calculate reveals for attempts after free period
        reveal_attempts = max_att - free_att  # Attempts where reveals happen
        if reveal_attempts <= 0:
            return list(range(rubric_count))

        # How many rubrics to reveal per attempt (distribute evenly)
        rubrics_per_attempt = rubric_count / reveal_attempts
        attempt_in_reveal_phase = attempt - free_att

        # Calculate how many should be revealed by this attempt
        target_revealed = min(
            rubric_count,
            int(rubrics_per_attempt * attempt_in_reveal_phase + 0.5)
        )

        return list(range(target_revealed))

    def _create_loose_hint(self, rubric_criterion: str) -> str:
        """Convert a rubric criterion into a loose hint using LLM.

        Uses the reward calculator's LLM to paraphrase the criterion
        in vaguer terms without giving away the exact requirement.
        """
        prompt = f"""Rephrase this evaluation criterion as a brief, vague hint.
Do not copy the original wording. Use general terms.
Keep it under 15 words.

Criterion: {rubric_criterion}

Hint:"""

        try:
            # Use the reward calculator's model to generate the hint
            inputs = self._reward_calculator.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=256,
                truncation=True,
            ).to(self._reward_calculator.device)

            with torch.no_grad():
                outputs = self._reward_calculator.hf_model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self._reward_calculator.tokenizer.pad_token_id,
                )

            hint = self._reward_calculator.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            ).strip()

            # Ensure hint is not empty
            if not hint or len(hint) < 5:
                return f"Consider aspects related to: {rubric_criterion[:40]}..."

            return hint

        except Exception:
            # Fallback if LLM fails
            return f"Think about: {rubric_criterion[:50]}..."

    def _check_hint_compliance(
        self, plan: str, revealed_indices: list[int], rubric_scores: dict
    ) -> tuple[int, float]:
        """Check if agent addressed revealed hints.

        Returns (ignored_count, compliance_penalty).
        """
        if not revealed_indices:
            return 0, 0.0

        ignored = 0
        for idx in revealed_indices:
            # If score for this criterion is low, consider it ignored
            score = rubric_scores.get(idx, 0.0)
            if score < 0.5:  # Threshold for "addressed"
                ignored += 1

        # Double penalty for each ignored hint
        penalty = ignored * self.ATTEMPT_PENALTY_BASE * self.COMPLIANCE_PENALTY_MULTIPLIER
        return ignored, penalty

    def _generate_feedback(
        self,
        rubric_scores: dict[int, float],
        rubric_avg: float,
        length_score: float,
        format_score: float,
        revealed_hints: list[str],
        hints_ignored: int,
    ) -> str:
        """Generate feedback including revealed hints."""
        criteria_met = sum(1 for s in rubric_scores.values() if s >= 0.7)
        total_criteria = len(rubric_scores)

        parts = []

        # Overall assessment
        if rubric_avg >= 0.9:
            parts.append("Excellent coverage of the research requirements.")
        elif rubric_avg >= 0.7:
            parts.append("Good progress. Your plan addresses most key aspects.")
        elif rubric_avg >= 0.5:
            parts.append(
                "Partial coverage. Consider expanding on core research components."
            )
        else:
            parts.append(
                "Your plan needs significant improvement. "
                "Focus on the fundamental aspects of the research goal."
            )

        parts.append(f"\nCriteria satisfied: {criteria_met}/{total_criteria}")

        # Length/format feedback
        if length_score < 0.5:
            parts.append("\nAdjust plan length for better coverage.")
        if format_score < 0.5:
            parts.append("\nUse multiple paragraphs to organise your ideas.")

        # Revealed hints section
        if revealed_hints:
            parts.append("\n\n--- REVEALED REQUIREMENTS ---")
            parts.append("Address these areas (ignoring them doubles penalty):\n")
            for i, hint in enumerate(revealed_hints, 1):
                parts.append(f"  {i}. {hint}")

            if hints_ignored > 0:
                parts.append(
                    f"\n[!] Warning: {hints_ignored} revealed requirement(s) "
                    "not adequately addressed. Double penalty applied."
                )

        return "\n".join(parts)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> ResearchPlanObservation:
        """Reset the environment with a new research task."""
        if seed is not None:
            self._rng.seed(seed)

        subset = kwargs.get("subset", self.subset)
        split = kwargs.get("split", self.split)
        max_attempts = kwargs.get("max_attempts", self.max_attempts)
        free_attempts = kwargs.get("free_attempts", self.free_attempts)

        # Reload dataset if subset/split changed
        if subset != self.subset or split != self.split:
            self.subset = subset
            self.split = split
            self._dataset = load_dataset(
                "facebook/research-plan-gen", subset
            )[split]
            self._dataset_size = len(self._dataset)

        # Select task
        index = kwargs.get("index")
        if index is None:
            index = self._rng.randint(0, self._dataset_size - 1)
        else:
            index = min(max(0, index), self._dataset_size - 1)

        task = self._dataset[index]

        if episode_id is None:
            episode_id = str(uuid.uuid4())

        self._episode_count += 1

        # Initialise internal state
        self._internal_state = InternalState(
            episode_id=episode_id,
            step_count=0,
            current_goal=task["Goal"],
            current_rubric=task["Rubric"],
            reference_solution=task["Reference solution"],
            article_id=task.get("article_id", ""),
            subset=subset,
            attempt_number=0,
            max_attempts=max_attempts,
            free_attempts=free_attempts,
            submission_history=[],
            best_score=0.0,
            cumulative_penalty=0.0,
            compliance_penalty=0.0,
            total_reward=0.0,
            revealed_rubric_indices=[],
        )

        return ResearchPlanObservation(
            done=False,
            reward=0.0,
            goal=self._internal_state.current_goal,
            rubric_count=len(self._internal_state.current_rubric),
            attempt_number=0,
            max_attempts=max_attempts,
            free_attempts=free_attempts,
            criteria_met=0,
            length_score=0.0,
            format_score=0.0,
            feedback=(
                f"Submit your research plan. You have {free_attempts} free attempts, "
                f"then {max_attempts - free_attempts} more with progressive hints. "
                "Ignoring revealed hints doubles the penalty."
            ),
            revealed_hints=[],
            hints_ignored_count=0,
            attempt_penalty=0.0,
            compliance_penalty=0.0,
            total_penalty=0.0,
            metadata={
                "episode_id": episode_id,
                "article_id": self._internal_state.article_id,
                "subset": subset,
            },
        )

    def _handle_reset_action(
        self, action: ResearchPlanAction
    ) -> ResearchPlanObservation:
        """Handle reset mode action.

        Uses provided subset/split if given, otherwise keeps current values.
        Falls back to ml:train for initial reset.
        """
        # Determine subset: action value > current > default
        if action.subset is not None:
            new_subset = action.subset.value
        elif self._internal_state is not None:
            new_subset = self._internal_state.subset
        else:
            new_subset = self.subset  # Initial default from constructor

        # Determine split: action value > current > default
        if action.split is not None:
            new_split = action.split.value
        elif self._internal_state is not None:
            new_split = self.split  # Use environment's current split
        else:
            new_split = self.split  # Initial default from constructor

        return self.reset(subset=new_subset, split=new_split)

    def step(self, action: ResearchPlanAction, **kwargs) -> ResearchPlanObservation:
        """Process the agent's action - either submit a plan or reset.

        When mode is RESET, resets the episode with optional subset/split.
        When mode is SUBMIT, evaluates the research plan.
        """
        # Handle reset mode
        if action.mode == ActionMode.RESET:
            return self._handle_reset_action(action)

        # Submit mode - validate we have a plan
        if not action.research_plan or not action.research_plan.strip():
            # Return error observation for empty submission
            if self._internal_state is None:
                return self.reset()
            return ResearchPlanObservation(
                done=False,
                reward=0.0,
                goal=self._internal_state.current_goal,
                rubric_count=len(self._internal_state.current_rubric),
                attempt_number=self._internal_state.attempt_number,
                max_attempts=self._internal_state.max_attempts,
                free_attempts=self._internal_state.free_attempts,
                feedback="Please provide a research plan to submit.",
                metadata={"error": "empty_submission"},
            )

        if self._internal_state is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")

        state = self._internal_state
        state.step_count += 1
        state.attempt_number += 1

        # Get hints that were revealed BEFORE this submission
        # (only these should count for compliance - agent hasn't seen new ones yet)
        previously_revealed_indices = self._get_rubrics_to_reveal(
            state.attempt_number - 1
        )

        # Calculate which rubrics to reveal for this attempt (includes new ones)
        state.revealed_rubric_indices = self._get_rubrics_to_reveal(
            state.attempt_number
        )

        # Get revealed hints to show the agent
        revealed_hints = [
            self._create_loose_hint(state.current_rubric[i])
            for i in state.revealed_rubric_indices
        ]

        # Calculate reward
        raw_score, breakdown = self._reward_calculator.calculate_reward(
            plan=action.research_plan,
            rubric=state.current_rubric,
            goal=state.current_goal,
            reference_solution=state.reference_solution,
        )

        # Calculate penalties
        attempt_penalty = self._calculate_attempt_penalty(state.attempt_number)
        # Only check compliance for hints the agent has already seen
        hints_ignored, compliance_penalty = self._check_hint_compliance(
            action.research_plan,
            previously_revealed_indices,
            breakdown["rubric_scores"],
        )

        state.cumulative_penalty = attempt_penalty
        state.compliance_penalty += compliance_penalty
        total_penalty = attempt_penalty + state.compliance_penalty

        # Track best score
        if raw_score > state.best_score:
            state.best_score = raw_score

        # Record submission
        state.submission_history.append({
            "attempt": state.attempt_number,
            "score": raw_score,
            "attempt_penalty": attempt_penalty,
            "compliance_penalty": compliance_penalty,
            "hints_ignored": hints_ignored,
        })

        # Final reward
        final_reward = max(0.0, state.best_score - total_penalty)
        state.total_reward = final_reward

        # Count criteria met
        criteria_met = sum(
            1 for s in breakdown["rubric_scores"].values() if s >= 0.7
        )

        # Done conditions
        done = (
            state.attempt_number >= state.max_attempts
            or raw_score >= self.SUCCESS_THRESHOLD
        )

        # Generate feedback
        feedback = self._generate_feedback(
            breakdown["rubric_scores"],
            breakdown["rubric_avg"],
            breakdown["length_score"],
            breakdown["format_score"],
            revealed_hints,
            hints_ignored,
        )

        if done:
            if raw_score >= self.SUCCESS_THRESHOLD:
                feedback = f"Success! Your plan meets the requirements.\n\n{feedback}"
            else:
                feedback = (
                    f"Episode ended after {state.attempt_number} attempts. "
                    f"Best score: {state.best_score:.2f}\n\n{feedback}"
                )

        return ResearchPlanObservation(
            done=done,
            reward=final_reward,
            goal=state.current_goal,
            rubric_count=len(state.current_rubric),
            attempt_number=state.attempt_number,
            max_attempts=state.max_attempts,
            free_attempts=state.free_attempts,
            criteria_met=criteria_met,
            length_score=breakdown["length_score"],
            format_score=breakdown["format_score"],
            feedback=feedback,
            revealed_hints=revealed_hints,
            hints_ignored_count=hints_ignored,
            attempt_penalty=attempt_penalty,
            compliance_penalty=state.compliance_penalty,
            total_penalty=total_penalty,
            metadata={
                "episode_id": state.episode_id,
                "step_count": state.step_count,
                "raw_score": raw_score,
                "best_score": state.best_score,
                "final_reward": final_reward,
                "attempts_remaining": state.max_attempts - state.attempt_number,
                "revealed_count": len(state.revealed_rubric_indices),
            },
        )

    @property
    def state(self) -> ResearchPlanState:
        """Return the public state (without rubric - prevents cheating)."""
        if self._internal_state is None:
            return ResearchPlanState(episode_id="", step_count=0)
        return self._internal_state.to_public_state()

    def close(self) -> None:
        """Clean up resources."""
        self._internal_state = None
        if hasattr(self._reward_calculator, "hf_model"):
            del self._reward_calculator.hf_model
            del self._reward_calculator.tokenizer

    def get_dataset_info(self) -> dict:
        """Return information about the loaded dataset."""
        return {
            "subset": self.subset,
            "split": self.split,
            "size": self._dataset_size,
            "episode_count": self._episode_count,
            "model_name": self.model_name,
            "max_attempts": self.max_attempts,
            "free_attempts": self.free_attempts,
        }
