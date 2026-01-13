# Green Agent implementation

This document describes the Green Agent implementation for the Reviewer Two environment, providing AgentBeats-compatible evaluation via the A2A protocol.

## Overview

The Green Agent evaluates Purple Agents on their ability to generate research plans. It uses the facebook/research-plan-gen dataset and scores submissions against hidden rubric criteria using an LLM-based evaluation system.

## Architecture

```
research_plan_env/green_agent/
    __init__.py      # Package exports
    models.py        # EvalRequest, EvalResult, TaskConfig
    messenger.py     # A2A communication utilities
    agent.py         # Evaluation agent and executor
    server.py        # A2A HTTP server entry point
```

## Components

### Models (models.py)

Defines the data structures for A2A communication:

- **EvalRequest**: Incoming evaluation request from AgentBeats
  - `participants`: Dict mapping role names to agent URLs
  - `config`: Evaluation configuration parameters

- **EvalResult**: Evaluation outcome returned to AgentBeats
  - `winner`: Role of the winning participant ("purple" or "none")
  - `detail`: Detailed scoring breakdown

- **TaskConfig**: Internal configuration with defaults
  - `subset`: Dataset subset (ml, arxiv, pubmed)
  - `split`: Dataset split (train, test)
  - `max_attempts`: Maximum evaluation attempts
  - `free_attempts`: Attempts before hints appear
  - `task_index`: Specific task index or None for random
  - `success_threshold`: Score required to pass (default: 0.8)

### Messenger (messenger.py)

Provides utilities for A2A protocol communication:

- **create_message()**: Creates A2A Message objects with text content
- **merge_parts()**: Concatenates message parts into text
- **send_message()**: Async function to send messages to agents
- **Messenger**: Class managing multi-turn conversations with context tracking

### Agent (agent.py)

Implements the evaluation logic:

- **ResearchPlanEvaluatorAgent**: Main evaluation agent
  - Validates incoming EvalRequests
  - Manages multi-turn evaluation with Purple Agents
  - Generates initial prompts and feedback prompts
  - Calculates final scores and determines winner

- **Executor**: A2A AgentExecutor implementation
  - Manages agent instances per conversation context
  - Handles task state transitions
  - Processes incoming A2A requests

### Server (server.py)

Entry point for the A2A HTTP server:

- **create_agent_card()**: Generates the A2A agent card
- **create_app()**: Creates the Starlette application
- **main()**: CLI entry point with argument parsing

## Evaluation flow

1. Green Agent receives EvalRequest with Purple Agent URL
2. Initialises environment with configured subset/split
3. Resets environment to get a research task
4. Sends initial prompt with goal to Purple Agent
5. Receives research plan response
6. Submits to environment for evaluation
7. If not passed and attempts remain:
   - Generates feedback prompt with hints
   - Sends to Purple Agent for improved submission
   - Repeats evaluation
8. Returns EvalResult with final scores

## Configuration options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| subset | string | "ml" | Dataset subset |
| split | string | "train" | Dataset split |
| max_attempts | int | 10 | Maximum attempts allowed |
| free_attempts | int | 2 | Attempts before hints |
| task_index | int/null | null | Specific task or random |
| success_threshold | float | 0.8 | Score to pass |

## Scoring

Purple Agents are scored on:

- **Rubric coverage (60%)**: LLM-evaluated criterion satisfaction
- **Length score (20%)**: Optimal range 400-1500 words
- **Format score (20%)**: Structure and technical content

Penalties are applied for:
- Multiple attempts (exponentially increasing)
- Ignoring revealed hints (2x penalty multiplier)

## Running the server

### Command line options

```
research-plan-green-agent [OPTIONS]

Options:
  --host TEXT      Host address (default: 127.0.0.1)
  --port INT       Port number (default: 9009)
  --card-url TEXT  External URL for agent card
```

### Environment variables

The Green Agent inherits configuration from the OpenEnv environment:

- `RPG_SUBSET`: Default dataset subset
- `RPG_SPLIT`: Default dataset split
- `RPG_MODEL`: Evaluation model (default: google/flan-t5-small)

## Docker deployment

Build the container:

```bash
docker build -f Dockerfile.green-agent -t research-plan-green-agent .
```

Run the container:

```bash
docker run -p 9009:9009 research-plan-green-agent
```

The agent card will be available at:
```
http://localhost:9009/.well-known/agent.json
```

## A2A endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/.well-known/agent.json` | GET | Agent card with capabilities |
| `/` | POST | A2A JSON-RPC endpoint |

## Example usage

### Sending an evaluation request

```python
import httpx
import json

request = {
    "participants": {
        "purple": "http://purple-agent:8080"
    },
    "config": {
        "subset": "ml",
        "split": "test",
        "max_attempts": 5,
        "success_threshold": 0.75
    }
}

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:9009/",
        json={
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": json.dumps(request)}]
                }
            },
            "id": "1"
        }
    )
```

### Building a Purple Agent

Purple Agents must:

1. Implement A2A protocol endpoints
2. Accept text messages with research goals
3. Return text messages with research plans
4. Handle multi-turn feedback conversations

Recommended response structure:

```
[Context and motivation]

[Specific questions to address]

[Detailed approach]

[Anticipated results]

[How success will be measured]

[Implementation plan]
```

## Testing

The GitHub Actions workflow automatically:

1. Builds the Docker image
2. Starts the container
3. Verifies the agent card endpoint
4. Publishes to GitHub Container Registry on successful builds

To test locally:

```bash
# Build and run
docker build -f Dockerfile.green-agent -t green-agent-test .
docker run -d -p 9009:9009 --name test-agent green-agent-test

# Verify agent card
curl http://localhost:9009/.well-known/agent.json

# Clean up
docker stop test-agent && docker rm test-agent
```
