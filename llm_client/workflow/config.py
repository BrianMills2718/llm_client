"""Workflow configuration -- stage-level model, retry, and prompt settings.

Loads from YAML so operational policy lives in config, not code. Each stage
can specify its model, retry policy, and prompt path. The workflow-level
config carries the shared task_prefix and max_budget.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StageRetryConfig(BaseModel):
    """Retry configuration for a single stage."""

    max_retries: int = 2
    base_delay: float = 1.0
    max_delay: float = 30.0


class StageConfig(BaseModel):
    """Configuration for a single workflow stage.

    Specifies the model, retry policy, and optional prompt path.
    All fields are optional -- nodes can override or ignore them.
    """

    model: str | None = None
    prompt: str | None = None
    retry: StageRetryConfig = Field(default_factory=StageRetryConfig)
    fallback_models: list[str] | None = None


class WorkflowConfig(BaseModel):
    """Top-level workflow configuration loaded from YAML.

    Carries the shared task_prefix, max_budget, default model, and
    per-stage overrides. Loaded via ``WorkflowConfig.from_yaml(path)``.
    """

    task_prefix: str = "workflow"
    max_budget: float = 0.0
    default_model: str | None = None
    stages: dict[str, StageConfig] = Field(default_factory=dict)

    def stage(self, name: str) -> StageConfig:
        """Get config for a stage, returning defaults if not specified."""
        return self.stages.get(name, StageConfig())

    def model_for_stage(self, name: str) -> str | None:
        """Resolve model for a stage: stage override -> default_model -> None."""
        stage_cfg = self.stage(name)
        return stage_cfg.model or self.default_model

    @classmethod
    def from_yaml(cls, path: str | Path) -> WorkflowConfig:
        """Load workflow config from a YAML file.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            RuntimeError: If the file is not valid YAML or doesn't match schema.
        """
        from importlib import import_module
        yaml = import_module("yaml")

        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Workflow config not found: {p}")

        try:
            raw = yaml.safe_load(p.read_text())
        except Exception as exc:
            raise RuntimeError(f"Invalid YAML in {p}: {exc}") from exc

        if not isinstance(raw, dict):
            raise RuntimeError(
                f"Workflow config {p}: expected dict, got {type(raw).__name__}"
            )

        return cls(**raw)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowConfig:
        """Create config from a plain dict (for testing or inline use)."""
        return cls(**data)
