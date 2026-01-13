"""A2A messaging utilities for Purple Agent communication.

Provides functions to send messages to Purple Agents and receive
their research plan submissions via the A2A protocol.
"""

import uuid
from typing import Any

import httpx
from a2a.client import A2ACardResolver, Client
from a2a.types import (
    DataPart,
    Message,
    MessageSendParams,
    Part,
    Task,
    TextPart,
)


def create_message(
    role: str,
    text: str,
    context_id: str | None = None,
) -> Message:
    """Create an A2A message with text content.

    Args:
        role: The role of the message sender (e.g., "user", "agent").
        text: The text content of the message.
        context_id: Optional context ID for conversation threading.

    Returns:
        A2A Message object ready to send.
    """
    return Message(
        kind="message",
        role=role,
        parts=[TextPart(kind="text", text=text)],
        messageId=uuid.uuid4().hex,
        contextId=context_id or uuid.uuid4().hex,
    )


def merge_parts(parts: list[Part]) -> str:
    """Merge message parts into a single text string.

    Args:
        parts: List of A2A Part objects (TextPart or DataPart).

    Returns:
        Concatenated text from all parts.
    """
    result = []
    for part in parts:
        if isinstance(part, TextPart):
            result.append(part.text)
        elif isinstance(part, DataPart):
            result.append(str(part.data))
    return "\n".join(result)


async def send_message(
    agent_url: str,
    message: Message,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Send a message to a Purple Agent and await response.

    Args:
        agent_url: Base URL of the Purple Agent's A2A endpoint.
        message: The message to send.
        timeout: Request timeout in seconds.

    Returns:
        Dictionary with response text, context_id, and optional status.

    Raises:
        httpx.HTTPError: If the request fails.
        ValueError: If the response format is unexpected.
    """
    async with httpx.AsyncClient(timeout=timeout) as http_client:
        resolver = A2ACardResolver(http_client, agent_url)
        card = await resolver.get_agent_card()

        client = Client(
            http_client=http_client,
            agent_card=card,
        )

        response = await client.send_message(
            MessageSendParams(message=message),
        )

        # Process the response
        result: dict[str, Any] = {"context_id": message.contextId}

        async for event in response:
            if isinstance(event, Message):
                result["text"] = merge_parts(event.parts)
                result["context_id"] = event.contextId
            elif isinstance(event, tuple) and len(event) == 2:
                task, _ = event
                if isinstance(task, Task) and task.status:
                    result["status"] = task.status.state
                    if task.status.message:
                        result["text"] = merge_parts(task.status.message.parts)
                    if task.artifacts:
                        result["artifacts"] = task.artifacts

        return result


class Messenger:
    """Manages conversations with Purple Agents.

    Maintains context IDs for multi-turn conversations and provides
    a simple interface for sending evaluation prompts.
    """

    def __init__(self, timeout: float = 300.0):
        """Initialise the messenger.

        Args:
            timeout: Default timeout for requests in seconds.
        """
        self.timeout = timeout
        self._contexts: dict[str, str] = {}

    async def talk_to_agent(
        self,
        agent_url: str,
        text: str,
        context_id: str | None = None,
    ) -> dict[str, Any]:
        """Send a message to a Purple Agent.

        Args:
            agent_url: Base URL of the Purple Agent.
            text: Message text to send.
            context_id: Optional context ID to continue a conversation.

        Returns:
            Response dictionary with text, context_id, and status.
        """
        # Use stored context if available and none provided
        if context_id is None and agent_url in self._contexts:
            context_id = self._contexts[agent_url]

        message = create_message(
            role="user",
            text=text,
            context_id=context_id,
        )

        response = await send_message(
            agent_url=agent_url,
            message=message,
            timeout=self.timeout,
        )

        # Store context for future turns
        if "context_id" in response:
            self._contexts[agent_url] = response["context_id"]

        return response

    def reset(self) -> None:
        """Clear all stored conversation contexts."""
        self._contexts.clear()
