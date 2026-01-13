---
title: Reviewer Two
emoji: "📝"
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# Reviewer Two

OpenEnv RL environment for training agents to generate research plans. Uses the [facebook/research-plan-gen](https://huggingface.co/datasets/facebook/research-plan-gen) dataset with LLM-based rubric evaluation.

This environment supports two (and a half) interfaces:
1. **OpenEnv API** - Standard RL environment interface for agent training, with the HumanAgent web GUI
2. **Green Agent (A2A)** - AgentBeats-compatible evaluation endpoint for benchmarking

## What's the point?

[See here for a detailed explanation](), but basically: if we're to have AI co-scientists, we need to have a way to elicit their research plans and guide them, the same way we guide coding agents. They in turn need to be able to learn from our idea of what a good research plan for a given topic is. We must, in short, turn our agents into diligent grad students with good research habits and no need for ramen.

## How does it work?

The fundamental principle is multi-turn adaptively penalised disclosure guidance. The agent is given a research topic, and must create a coherent research plan. The green agent/tasker has a rubric of criteria of what a good research plan looks like, but keeps this to themselves. The purple agent/learner can make multiple attempts and receives a reward signal, together with an evaluation. Initially, this is non-specific but also penalty-free. After two free attempts, the green agent starts giving specific hints, but also starts penalising the purple agent for ignoring these hints. This way, the purple agent can learn both from trial-and-error and from guided feedback. More importantly: it has to learn that there are stakes to this game.


## Features

- Multi-turn episodic environment with progressive hint reveal
- Hidden rubrics to prevent reward hacking
- LLM-based evaluation using FLAN-T5
- Compliance penalties for ignoring revealed hints
- Support for ML, ArXiv, and PubMed research domains

## Environment mechanics

### Action space

Actions have two modes:
- **submit**: Submit a research plan for evaluation
- **reset**: Reset the episode with optional subset/split selection

```json
{
  "mode": "submit",
  "research_plan": "Your comprehensive research plan..."
}
```

Or to reset with configuration:
```json
{
  "mode": "reset",
  "subset": "arxiv",
  "split": "test"
}
```

### Observation space

- `goal`: The research task to address
- `rubric_count`: Number of hidden evaluation criteria
- `attempt_number`: Current attempt (1-indexed)
- `revealed_hints`: Hints for rubric criteria (after free attempts)
- `feedback`: Evaluation feedback and scores
- `reward`: Final reward (0.0-1.0)
- `done`: Whether the episode has ended

### Reward structure

- **Rubric coverage** (60%): LLM-evaluated criterion satisfaction using FLAN-T5 with semantic relevance and coherence checking
- **Length score** (20%): Optimal range is 400-1500 words (minimum 200, maximum 3000)
- **Format score** (20%): Paragraphs, section headers, and lists

### Penalties

We're applying a two-free then progressively revealing but penalising system, as a compromise against length gaming but allowing learning opportunities:

- **Attempt penalty**: Exponentially increasing after first attempt
- **Compliance penalty**: 2x penalty for ignoring revealed hints

## OpenEnv API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Get a new research task |
| `/step` | POST | Submit action and receive observation |
| `/state` | GET | Get current environment state |
| `/health` | GET | Health check |
| `/info` | GET | Environment information |
| `/config` | GET/POST | View/update runtime configuration |
| `/web` | GET | Interactive web interface |

## Configuration

Environment variables:
- `RPG_SUBSET`: Dataset subset (`ml`, `arxiv`, `pubmed`) - default: `ml`
- `RPG_SPLIT`: Dataset split (`train`, `test`) - default: `train`
- `RPG_MODEL`: Evaluation model - default: `google/flan-t5-small`
- `RPG_SEED`: Random seed for reproducibility

## Running the OpenEnv server

### Local development

```bash
pip install -e .
uvicorn research_plan_env.server.app:app --host 0.0.0.0 --port 7860
```

### Docker (Hugging Face Spaces)

The environment is deployed at: https://huggingface.co/spaces/chrisvoncsefalvay/reviewer-two-env

---

## Green Agent (AgentBeats A2A protocol)

This environment also functions as a **Green Agent** for the [AgentBeats](https://rdi.berkeley.edu/agentx-agentbeats) paradigm, enabling evaluation of Purple Agents on research plan generation tasks via the [A2A protocol](https://a2a-protocol.org/).

### What is a Green Agent?

In the AgentBeats paradigm, a **Green Agent** is an automated evaluator that benchmarks **Purple Agents** (the agents being tested). This inverts traditional benchmarking - instead of adapting agents to fit benchmarks, the benchmark itself becomes an agent. Kinda neat, if you ask me.

### How it works

1. The Green Agent receives an `EvalRequest` with the Purple Agent's URL
2. It initialises a research task from the dataset
3. Sends the research goal to the Purple Agent
4. Receives the Purple Agent's research plan
5. Evaluates using the hidden rubric criteria
6. Optionally allows multiple attempts with progressive hints
7. Returns an `EvalResult` with scores and winner determination

### A2A endpoints

The Green Agent exposes the standard A2A protocol endpoints:

| Endpoint | Description |
|----------|-------------|
| `/.well-known/agent.json` | Agent card with capabilities |
| `/` | A2A JSON-RPC endpoint |

### EvalRequest format

```json
{
  "participants": {
    "purple": "http://purple-agent:8080"
  },
  "config": {
    "subset": "ml",
    "split": "train",
    "max_attempts": 10,
    "task_index": null,
    "success_threshold": 0.8
  }
}
```

### EvalResult format

```json
{
  "winner": "purple",
  "detail": {
    "passed": true,
    "best_score": 0.85,
    "best_attempt": 3,
    "total_attempts": 3,
    "success_threshold": 0.8,
    "scores": [
      {
        "attempt": 1,
        "reward": 0.62,
        "criteria_met": 3,
        "rubric_count": 5
      }
    ]
  }
}
```

### Running the Green Agent

#### Local development

```bash
pip install -e ".[green-agent]"
research-plan-green-agent --host 0.0.0.0 --port 9009
```

#### Docker

```bash
docker build -f Dockerfile.green-agent -t research-plan-green-agent .
docker run -p 9009:9009 research-plan-green-agent
```

### Building a Purple Agent

To compete against this Green Agent, your Purple Agent must:

1. Implement the A2A protocol (use [a2a-sdk](https://pypi.org/project/a2a-sdk/))
2. Accept text messages containing research goals
3. Return text messages with comprehensive research plans
4. Optionally handle multi-turn feedback for improvement (it will be rather boring without it -- trust me: you can just go and do DPO on the dataset)

### Evaluation criteria

Purple Agents are scored on:

- **Rubric coverage**: How well the plan addresses hidden criteria
- **Length appropriateness**: 400-1500 words optimal
- **Structure**: Paragraphs, section headers, and lists
- **Hint compliance**: Addressing revealed requirements (after free attempts)

### Success threshold

By default, a score of 0.8 or higher is required to pass. The Purple Agent wins if it achieves this threshold within the allowed attempts.

## License

MIT
