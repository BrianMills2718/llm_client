"""Rubric-based output scoring for the improvement flywheel.

Load YAML rubrics, score task outputs via LLM-as-judge, and log results
to the observability DB for analysis by prompt_eval, analyzer, and evolve.

Usage:
    from llm_client import score_output, ascore_output, load_rubric

    result = await ascore_output(
        output="<task output text>",
        rubric="research_quality",
        context={"query": "Palantir contracts"},
        task="sam_gov_research",
    )
    print(result.overall_score)  # 0.0 - 1.0
    print(result.dimensions)     # {"completeness": 4, "accuracy": 3, ...}
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Rubric resolution directories (checked in order):
# 1. Project-local rubrics/ (cwd)
# 2. Built-in llm_client/rubrics/
_BUILTIN_RUBRICS_DIR = Path(__file__).parent / "rubrics"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class RubricCriterion(BaseModel):
    """A single scoring dimension within a rubric."""

    name: str
    weight: float
    description: str
    scale: int = 5


class Rubric(BaseModel):
    """A scoring rubric loaded from YAML."""

    name: str
    version: int = 1
    description: str = ""
    dimensions: list[RubricCriterion]

    @property
    def total_weight(self) -> float:
        return sum(d.weight for d in self.dimensions)


class CriterionScore(BaseModel):
    """Score for a single criterion."""

    criterion: str
    score: int
    reasoning: str = ""


class ScoreResult(BaseModel):
    """Result of scoring a task output against a rubric."""

    rubric: str
    overall_score: float = Field(description="Weighted score normalized to 0.0 - 1.0")
    dimensions: dict[str, int] = Field(description="Per-criterion raw scores")
    reasoning: dict[str, str] = Field(
        default_factory=dict, description="Per-criterion reasoning"
    )
    judge_model: str = ""
    method: str = "llm_judge"
    cost: float = 0.0
    latency_s: float = 0.0
    git_commit: str | None = None


# Pydantic model for structured judge output
class _JudgeOutput(BaseModel):
    scores: list[CriterionScore]


# ---------------------------------------------------------------------------
# Rubric loading
# ---------------------------------------------------------------------------


def load_rubric(name_or_path: str) -> Rubric:
    """Load a rubric by name or path.

    Resolution order:
    1. If name_or_path is an existing file path, load it directly
    2. Project-local rubrics/ directory (cwd)
    3. Built-in llm_client/rubrics/ directory

    Args:
        name_or_path: Rubric name (e.g., "research_quality") or path to YAML file

    Returns:
        Rubric model

    Raises:
        FileNotFoundError: If rubric not found in any location
    """
    # Direct path?
    p = Path(name_or_path)
    if p.is_file():
        return _load_rubric_file(p)

    # Ensure .yaml extension
    fname = name_or_path if name_or_path.endswith(".yaml") else f"{name_or_path}.yaml"

    # Project-local
    local = Path.cwd() / "rubrics" / fname
    if local.is_file():
        return _load_rubric_file(local)

    # Built-in
    builtin = _BUILTIN_RUBRICS_DIR / fname
    if builtin.is_file():
        return _load_rubric_file(builtin)

    raise FileNotFoundError(
        f"Rubric {name_or_path!r} not found. Searched:\n"
        f"  {local}\n"
        f"  {builtin}"
    )


def list_rubrics() -> list[str]:
    """List available built-in rubric names."""
    rubrics = []
    if _BUILTIN_RUBRICS_DIR.is_dir():
        for f in sorted(_BUILTIN_RUBRICS_DIR.glob("*.yaml")):
            rubrics.append(f.stem)
    return rubrics


def _load_rubric_file(path: Path) -> Rubric:
    """Parse a rubric YAML file into a Rubric model."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return Rubric(**data)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


async def ascore_output(
    output: str,
    rubric: str | Rubric,
    *,
    context: dict[str, Any] | str = "",
    task: str | None = None,
    trace_id: str | None = None,
    method: str = "llm_judge",
    judge_model: str | None = None,
    output_model: str | None = None,
    agent_spec: str | None = None,
    prompt_id: str | None = None,
    git_commit: str | None = None,
) -> ScoreResult:
    """Score a task output against a rubric.

    Args:
        output: The task output text to score
        rubric: Rubric name (e.g., "research_quality") or Rubric object
        context: Task context (query, config, etc.) — shown to the judge
        task: Task tag for observability DB
        trace_id: Trace ID to correlate with the execution that produced the output
        method: Scoring method — "llm_judge" (default)
        judge_model: Override judge model (default: get_model("judging"))
        output_model: Model that produced the output (for attribution)
        agent_spec: Agent architecture that produced the output (for attribution)
        prompt_id: Prompt version that produced the output (for attribution)

    Returns:
        ScoreResult with overall_score (0.0-1.0), per-criterion scores, reasoning
    """
    # Load rubric if string
    if isinstance(rubric, str):
        rubric_obj = load_rubric(rubric)
    else:
        rubric_obj = rubric

    # Resolve judge model
    if judge_model is None:
        from llm_client.models import get_model

        judge_model = get_model("judging")

    # Build judge prompt
    context_str = (
        json.dumps(context, indent=2) if isinstance(context, dict) else str(context)
    )

    from llm_client.prompts import render_prompt

    prompt_path = Path(__file__).parent / "prompts" / "rubric_judge.yaml"
    messages = render_prompt(
        str(prompt_path),
        output=output[:10000],  # Cap output shown to judge
        context=context_str[:2000],
        rubric_name=rubric_obj.name,
        rubric_description=rubric_obj.description,
        criteria=[
            {
                "name": c.name,
                "weight": c.weight,
                "description": c.description,
                "scale": c.scale,
            }
            for c in rubric_obj.dimensions
        ],
    )

    # Call judge LLM
    t0 = time.monotonic()
    from llm_client.client import acall_llm_structured

    parsed, meta = await acall_llm_structured(
        judge_model,
        messages,
        response_model=_JudgeOutput,
        task="scoring",
        trace_id=trace_id,
    )
    latency = time.monotonic() - t0
    judge_cost = meta.cost or 0.0

    # Compute weighted score
    scores_by_name: dict[str, CriterionScore] = {s.criterion: s for s in parsed.scores}
    total_weighted = 0.0
    total_weight = 0.0
    dimensions: dict[str, int] = {}
    reasoning: dict[str, str] = {}

    for criterion in rubric_obj.dimensions:
        cs = scores_by_name.get(criterion.name)
        if cs is not None:
            dimensions[criterion.name] = cs.score
            reasoning[criterion.name] = cs.reasoning
            # Normalize to 0-1 for this criterion, then weight
            normalized = (cs.score - 1) / max(criterion.scale - 1, 1)
            total_weighted += normalized * criterion.weight
            total_weight += criterion.weight

    overall = total_weighted / total_weight if total_weight > 0 else 0.0

    result = ScoreResult(
        rubric=rubric_obj.name,
        overall_score=round(overall, 4),
        dimensions=dimensions,
        reasoning=reasoning,
        judge_model=judge_model,
        method=method,
        cost=judge_cost,
        latency_s=round(latency, 3),
        git_commit=git_commit,
    )

    # Log to observability DB
    from llm_client import io_log

    io_log.log_score(
        rubric=rubric_obj.name,
        method=method,
        overall_score=result.overall_score,
        dimensions=dimensions,
        reasoning=json.dumps(reasoning),
        output_model=output_model,
        judge_model=judge_model,
        agent_spec=agent_spec,
        prompt_id=prompt_id,
        cost=judge_cost,
        latency_s=latency,
        task=task,
        trace_id=trace_id,
        git_commit=git_commit,
    )

    return result


def score_output(
    output: str,
    rubric: str | Rubric,
    **kwargs: Any,
) -> ScoreResult:
    """Sync wrapper for ascore_output. See ascore_output for full docs."""
    return asyncio.get_event_loop().run_until_complete(
        ascore_output(output, rubric, **kwargs)
    )
