"""Tests for the Purple Agent A2A conformance and functionality.

These tests verify that the Purple Agent:
1. Conforms to A2A protocol requirements
2. Generates valid research plans
3. Handles multi-turn conversations correctly
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from a2a.types import Message, TextPart


@pytest.fixture
def mock_anthropic():
    """Mock the Anthropic client for testing without API calls."""
    with patch(
        "research_plan_env.purple_agent.agent.ANTHROPIC_AVAILABLE", True
    ):
        with patch("research_plan_env.purple_agent.agent.anthropic") as mock:
            # Create mock response
            mock_response = MagicMock()
            mock_response.content = [
                MagicMock(text="# Research Plan\n\nTest plan content.")
            ]

            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock.Anthropic.return_value = mock_client

            yield mock


@pytest.fixture
def sample_message():
    """Create a sample A2A message for testing."""
    return Message(
        kind="message",
        role="user",
        parts=[
            TextPart(
                kind="text",
                text="Create a research plan for studying ML robustness.",
            )
        ],
        messageId="test-message-id",
        contextId="test-context-id",
    )


class TestResearchPlanGeneratorAgent:
    """Tests for the ResearchPlanGeneratorAgent class."""

    @pytest.mark.asyncio
    async def test_agent_initialisation(self, mock_anthropic):
        """Test that agent initialises correctly."""
        os.environ["ANTHROPIC_API_KEY"] = "test-key"

        from research_plan_env.purple_agent.agent import (
            ResearchPlanGeneratorAgent,
        )

        agent = ResearchPlanGeneratorAgent()
        assert agent.model == "claude-sonnet-4-20250514"
        assert agent.max_tokens == 4096
        assert agent._conversation_history == []

    @pytest.mark.asyncio
    async def test_agent_reset(self, mock_anthropic):
        """Test that agent reset clears conversation history."""
        os.environ["ANTHROPIC_API_KEY"] = "test-key"

        from research_plan_env.purple_agent.agent import (
            ResearchPlanGeneratorAgent,
        )

        agent = ResearchPlanGeneratorAgent()
        agent._conversation_history = [{"role": "user", "content": "test"}]
        agent.reset()
        assert agent._conversation_history == []

    @pytest.mark.asyncio
    async def test_generate_plan(self, mock_anthropic):
        """Test research plan generation."""
        os.environ["ANTHROPIC_API_KEY"] = "test-key"

        from research_plan_env.purple_agent.agent import (
            ResearchPlanGeneratorAgent,
        )

        agent = ResearchPlanGeneratorAgent()
        plan = await agent._generate_plan("Test research goal")

        assert "Research Plan" in plan
        assert len(agent._conversation_history) == 2  # User + assistant


class TestExecutor:
    """Tests for the A2A Executor class."""

    @pytest.mark.asyncio
    async def test_executor_creates_agent_per_context(self, mock_anthropic):
        """Test that executor creates separate agents for each context."""
        os.environ["ANTHROPIC_API_KEY"] = "test-key"

        from research_plan_env.purple_agent.agent import Executor

        executor = Executor()
        assert len(executor.agents) == 0

        # After processing a request, an agent should be created
        # This is tested indirectly through the full integration tests


class TestAgentCard:
    """Tests for A2A agent card configuration."""

    def test_agent_card_creation(self):
        """Test that agent card is created with correct metadata."""
        from research_plan_env.purple_agent.server import create_agent_card

        card = create_agent_card("localhost", 9010, None)

        assert card.name == "Research Plan Generator (Claude Sonnet)"
        assert "claude" in card.description.lower()
        assert card.version == "1.0.0"
        assert card.capabilities.streaming is True
        assert len(card.skills) == 1
        assert card.skills[0].id == "generate_research_plan"

    def test_agent_card_custom_url(self):
        """Test agent card with custom URL."""
        from research_plan_env.purple_agent.server import create_agent_card

        card = create_agent_card("localhost", 9010, "https://example.com")
        assert card.url == "https://example.com"


class TestA2AConformance:
    """Tests for A2A protocol conformance."""

    def test_app_creation(self, mock_anthropic):
        """Test that A2A application can be created."""
        os.environ["ANTHROPIC_API_KEY"] = "test-key"

        from research_plan_env.purple_agent.server import create_app

        app = create_app(host="localhost", port=9010)
        assert app is not None

    @pytest.mark.asyncio
    async def test_message_handling(self, mock_anthropic, sample_message):
        """Test that agent handles A2A messages correctly."""
        os.environ["ANTHROPIC_API_KEY"] = "test-key"

        from research_plan_env.purple_agent.agent import (
            ResearchPlanGeneratorAgent,
        )

        agent = ResearchPlanGeneratorAgent()

        # Create mock task updater
        mock_updater = MagicMock()
        mock_updater.update_status = MagicMock()
        mock_updater.complete = MagicMock()
        mock_updater.add_artifact = MagicMock()
        mock_updater.failed = MagicMock()

        await agent.run(sample_message, mock_updater)

        # Verify status was updated
        mock_updater.update_status.assert_called()

        # Verify artifact was added with the research plan
        mock_updater.add_artifact.assert_called()

        # Verify task was completed
        mock_updater.complete.assert_called()


class TestEnvironmentVariables:
    """Tests for environment variable handling."""

    def test_missing_api_key_warning(self, mock_anthropic):
        """Test that missing API key is handled gracefully at startup."""
        # Remove API key if set
        os.environ.pop("ANTHROPIC_API_KEY", None)

        from research_plan_env.purple_agent.agent import (
            ResearchPlanGeneratorAgent,
        )

        agent = ResearchPlanGeneratorAgent()

        # Should not raise during init
        assert agent._client is None

    def test_custom_model_from_env(self, mock_anthropic):
        """Test that custom model can be set via environment variable."""
        os.environ["ANTHROPIC_MODEL"] = "claude-3-opus-20240229"
        os.environ["ANTHROPIC_API_KEY"] = "test-key"

        from research_plan_env.purple_agent.agent import (
            ResearchPlanGeneratorAgent,
        )

        agent = ResearchPlanGeneratorAgent()
        assert agent.model == "claude-3-opus-20240229"

        # Clean up
        del os.environ["ANTHROPIC_MODEL"]
