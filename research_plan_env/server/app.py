"""FastAPI application for the Research Plan Generation Environment.

Creates a web interface and API endpoints for the RL environment
using OpenEnv's create_web_interface_app factory.
"""

import os

from fastapi import HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from openenv.core.env_server import create_web_interface_app
from pydantic import BaseModel

from research_plan_env.models import (
    DatasetSplit,
    DatasetSubset,
    ResearchPlanAction,
    ResearchPlanObservation,
)
from research_plan_env.server.environment import ResearchPlanEnvironment


# Available dataset options (derived from enums)
AVAILABLE_SUBSETS = [s.value for s in DatasetSubset]
AVAILABLE_SPLITS = [s.value for s in DatasetSplit]

# Mutable configuration (can be changed at runtime)
runtime_config = {
    "subset": os.environ.get("RPG_SUBSET", "ml"),
    "split": os.environ.get("RPG_SPLIT", "train"),
    "model_name": os.environ.get("RPG_MODEL", "google/flan-t5-small"),
    "seed": int(os.environ.get("RPG_SEED")) if os.environ.get("RPG_SEED") else None,
}


class ConfigUpdate(BaseModel):
    """Schema for configuration updates."""

    subset: str | None = None
    split: str | None = None


class DynamicEnvironment(ResearchPlanEnvironment):
    """Environment that uses runtime configuration.

    This class is passed to create_web_interface_app as a factory.
    """

    def __init__(self):
        super().__init__(
            subset=runtime_config["subset"],
            split=runtime_config["split"],
            model_name=runtime_config["model_name"],
            seed=runtime_config["seed"],
        )


# Create the FastAPI application with web interface
app = create_web_interface_app(
    DynamicEnvironment,
    ResearchPlanAction,
    ResearchPlanObservation,
    env_name="research_plan_env",
)


@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect root to the web interface."""
    return RedirectResponse(url="/web")


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    return {
        "status": "healthy",
        "environment": "research_plan_env",
        "config": {
            "subset": runtime_config["subset"],
            "split": runtime_config["split"],
            "model": runtime_config["model_name"],
        },
    }


@app.get("/config")
async def get_config():
    """Get current configuration."""
    return {
        "current": {
            "subset": runtime_config["subset"],
            "split": runtime_config["split"],
        },
        "available": {
            "subsets": AVAILABLE_SUBSETS,
            "splits": AVAILABLE_SPLITS,
        },
    }


@app.post("/config")
async def update_config(update: ConfigUpdate):
    """Update runtime configuration.

    Note: Changes take effect on the next environment reset.
    """
    if update.subset is not None:
        if update.subset not in AVAILABLE_SUBSETS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid subset. Must be one of: {AVAILABLE_SUBSETS}",
            )
        runtime_config["subset"] = update.subset

    if update.split is not None:
        if update.split not in AVAILABLE_SPLITS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid split. Must be one of: {AVAILABLE_SPLITS}",
            )
        runtime_config["split"] = update.split

    return {
        "status": "updated",
        "config": {
            "subset": runtime_config["subset"],
            "split": runtime_config["split"],
        },
        "message": "Changes will take effect on next environment reset.",
    }


@app.get("/config/ui", response_class=HTMLResponse)
async def config_ui():
    """Simple HTML UI for configuration."""
    current_subset = runtime_config["subset"]
    current_split = runtime_config["split"]

    subset_options = "\n".join(
        f'<option value="{s}" {"selected" if s == current_subset else ""}>{s}</option>'
        for s in AVAILABLE_SUBSETS
    )
    split_options = "\n".join(
        f'<option value="{s}" {"selected" if s == current_split else ""}>{s}</option>'
        for s in AVAILABLE_SPLITS
    )

    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Environment Configuration</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 600px;
            margin: 50px auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 24px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            margin-top: 0;
            color: #333;
        }}
        label {{
            display: block;
            margin: 16px 0 8px;
            font-weight: 500;
            color: #555;
        }}
        select {{
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 16px;
        }}
        button {{
            margin-top: 24px;
            padding: 12px 24px;
            background: #0066cc;
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 16px;
            cursor: pointer;
        }}
        button:hover {{
            background: #0052a3;
        }}
        .status {{
            margin-top: 16px;
            padding: 12px;
            border-radius: 4px;
            display: none;
        }}
        .status.success {{
            background: #d4edda;
            color: #155724;
            display: block;
        }}
        .status.error {{
            background: #f8d7da;
            color: #721c24;
            display: block;
        }}
        .nav {{
            margin-bottom: 16px;
        }}
        .nav a {{
            color: #0066cc;
            text-decoration: none;
        }}
        .nav a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="nav">
        <a href="/web">&larr; Back to Environment</a>
    </div>
    <div class="card">
        <h1>Dataset Configuration</h1>
        <p>Select the dataset subset and split for research plan tasks.</p>

        <label for="subset">Dataset Subset</label>
        <select id="subset">
            {subset_options}
        </select>

        <label for="split">Dataset Split</label>
        <select id="split">
            {split_options}
        </select>

        <button onclick="updateConfig()">Update Configuration</button>

        <div id="status" class="status"></div>
    </div>

    <script>
        async function updateConfig() {{
            const subset = document.getElementById('subset').value;
            const split = document.getElementById('split').value;
            const statusEl = document.getElementById('status');

            try {{
                const response = await fetch('/config', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ subset, split }})
                }});

                const data = await response.json();

                if (response.ok) {{
                    statusEl.className = 'status success';
                    statusEl.textContent = 'Configuration updated. Reset the environment to apply changes.';
                }} else {{
                    statusEl.className = 'status error';
                    statusEl.textContent = data.detail || 'Failed to update configuration.';
                }}
            }} catch (err) {{
                statusEl.className = 'status error';
                statusEl.textContent = 'Network error: ' + err.message;
            }}
        }}
    </script>
</body>
</html>
"""


@app.get("/info")
async def environment_info():
    """Return information about the environment configuration."""
    return {
        "name": "Research Plan Generation Environment",
        "description": (
            "Multi-turn RL environment for training agents to generate "
            "research plans. Features progressive rubric reveal and "
            "compliance penalties."
        ),
        "dataset": {
            "name": "facebook/research-plan-gen",
            "subset": runtime_config["subset"],
            "split": runtime_config["split"],
            "available_subsets": AVAILABLE_SUBSETS,
            "available_splits": AVAILABLE_SPLITS,
        },
        "mechanics": {
            "free_attempts": 2,
            "max_attempts": 10,
            "progressive_hints": True,
            "compliance_penalty": "2x for ignoring revealed hints",
        },
        "config_url": "/config/ui",
    }


@app.get("/settings", response_class=HTMLResponse)
async def settings_redirect():
    """Redirect to config UI."""
    return RedirectResponse(url="/config/ui")
