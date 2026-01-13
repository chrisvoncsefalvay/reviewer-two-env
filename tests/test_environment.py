"""Tests for the Research Plan Generation Environment."""

import pytest

from research_plan_env.models import (
    ResearchPlanAction,
    ResearchPlanObservation,
    ResearchPlanState,
)
from research_plan_env.server.environment import ResearchPlanEnvironment
from research_plan_env.server.reward_calculator import RewardCalculator


class TestRewardCalculator:
    """Tests for the reward calculator.

    Note: These tests load actual models which can be slow on first run.
    """

    @pytest.fixture(scope="class")
    def calculator(self):
        """Create a reward calculator (loads models - cached for class)."""
        return RewardCalculator()

    def test_length_score_optimal(self, calculator):
        """Test length scoring for optimal length plans."""
        plan = " ".join(["word"] * 800)
        score = calculator._calculate_length_score(plan)
        assert score == 1.0

    def test_length_score_too_short(self, calculator):
        """Test length scoring for too short plans."""
        plan = " ".join(["word"] * 50)
        score = calculator._calculate_length_score(plan)
        assert score < 0.5

    def test_length_score_too_long(self, calculator):
        """Test length scoring for too long plans."""
        plan = " ".join(["word"] * 5000)
        score = calculator._calculate_length_score(plan)
        assert score < 1.0

    def test_format_score_structured(self, calculator):
        """Test format scoring for well-structured plans."""
        plan = """# Introduction

This is the first section.

## Methodology

- Step 1: Data collection
- Step 2: Analysis
- Step 3: Evaluation

## Results

The experiment shows improved performance metrics."""
        score = calculator._calculate_format_score(plan)
        assert score >= 0.5

    def test_format_score_unstructured(self, calculator):
        """Test format scoring for unstructured plans."""
        plan = "This is just a single paragraph with no structure."
        score = calculator._calculate_format_score(plan)
        assert score < 0.5

    def test_calculate_reward(self, calculator):
        """Test full reward calculation."""
        plan = """# Research Plan

## Objective
Address the scalability challenge.

## Methodology
- Use efficient algorithms
- Implement parallel processing
- Optimise memory usage

## Evaluation
Test on standard benchmarks for performance and accuracy metrics."""

        rubric = [
            "The solution should be scalable",
            "The method should use efficient algorithms",
        ]
        goal = "Develop a scalable solution for large-scale data processing"

        reward, breakdown = calculator.calculate_reward(plan, rubric, goal)

        assert 0.0 <= reward <= 1.0
        assert "rubric_scores" in breakdown
        assert "length_score" in breakdown
        assert "format_score" in breakdown
        assert "feedback" in breakdown

    def test_coherence_check_valid_text(self, calculator):
        """Test coherence check with valid English text."""
        text = """This is a coherent research plan that discusses methodology
        and evaluation criteria. The approach involves systematic analysis
        of the data using established techniques from the literature."""
        score = calculator._check_coherence(text)
        assert score > 0.7

    def test_coherence_check_gibberish(self, calculator):
        """Test coherence check with gibberish text."""
        text = "asdf jkl qwer zxcv bnm uiop hjkl tyui ghjk vbnm"
        score = calculator._check_coherence(text)
        assert score < 0.5

    def test_relevance_calculation(self, calculator):
        """Test semantic relevance calculation."""
        plan = "We will use machine learning algorithms to classify the data."
        criterion = "The solution should employ classification techniques."
        score = calculator._calculate_relevance(plan, criterion)
        assert 0.0 <= score <= 1.0
        assert score > 0.3  # Should have some relevance


class TestModels:
    """Tests for the data models."""

    def test_action_creation(self):
        """Test creating a research plan action."""
        action = ResearchPlanAction(research_plan="My research plan")
        assert action.research_plan == "My research plan"

    def test_observation_creation(self):
        """Test creating an observation."""
        obs = ResearchPlanObservation(
            done=False,
            reward=0.5,
            goal="Test goal",
            rubric_count=3,
        )
        assert obs.done is False
        assert obs.reward == 0.5
        assert obs.goal == "Test goal"
        assert obs.rubric_count == 3

    def test_state_creation(self):
        """Test creating a state."""
        state = ResearchPlanState(
            episode_id="test-123",
            step_count=1,
            current_goal="Test goal",
        )
        assert state.episode_id == "test-123"
        assert state.step_count == 1


class TestEnvironment:
    """Tests for the environment.

    Note: These tests load models and datasets which can be slow.
    """

    @pytest.fixture(scope="class")
    def env(self):
        """Create an environment (cached for class)."""
        return ResearchPlanEnvironment(
            subset="ml",
            split="test",
            seed=42,
        )

    def test_reset(self, env):
        """Test environment reset."""
        obs = env.reset()

        assert obs.done is False
        assert obs.reward == 0.0
        assert len(obs.goal) > 0
        assert obs.rubric_count > 0

    def test_step(self, env):
        """Test environment step.

        Note: The environment is multi-turn. A single step only ends the
        episode if the score reaches the success threshold (0.8) or max
        attempts are reached.
        """
        env.reset()

        action = ResearchPlanAction(
            research_plan="""# Research Plan

## Approach
This plan addresses the research goal by proposing a novel methodology
that combines multiple techniques for improved performance.

## Methods
- Data preprocessing and augmentation
- Model architecture design
- Training procedure optimisation

## Evaluation
Comprehensive experiments on standard benchmarks."""
        )

        obs = env.step(action)

        # Episode continues unless score >= 0.8 or max attempts reached
        assert 0.0 <= obs.reward <= 1.0
        assert len(obs.feedback) > 0
        assert obs.attempt_number == 1

    def test_state_property(self, env):
        """Test state property after reset."""
        env.reset()
        state = env.state

        assert len(state.episode_id) > 0
        assert state.step_count == 0
        assert len(state.current_goal) > 0

    def test_dataset_info(self, env):
        """Test dataset info method."""
        info = env.get_dataset_info()

        assert info["subset"] == "ml"
        assert info["split"] == "test"
        assert info["size"] > 0
