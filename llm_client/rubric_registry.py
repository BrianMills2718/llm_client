"""Rubric registry for categorical scoring of LLM task outputs.

Loads YAML rubric definitions that use categorical scales (not continuous 0-1)
per Brian's preference. Each dimension has named categories with descriptions
and numeric scores, giving judges clear decision boundaries.

The registry supports two-tier resolution: shared rubrics ship in
``llm_client/rubrics/``, and projects can override or extend with a local
``rubrics/`` directory. Programmatic registration is also supported.

Compatibility: the ``Rubric`` model here can be converted to
``prompt_eval.scoring.Rubric`` via ``to_prompt_eval_rubric()`` for use
with the existing LLM-judge scoring pipeline.

Usage::

    from llm_client.rubric_registry import RubricRegistry

    registry = RubricRegistry()
    rubric = registry.get("extraction_quality")
    for dim in rubric.dimensions:
        print(dim.name, [c.name for c in dim.categories])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Default rubric directories (checked in order).
_PACKAGE_RUBRICS_DIR = Path(__file__).resolve().parent.parent / "rubrics"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Category(BaseModel):
    """A named scoring category within a dimension.

    Each category has a human-readable name, a description that anchors
    the judge's decision, and a numeric score for aggregation.
    """

    name: str = Field(description="Short identifier, e.g. 'complete', 'partial'")
    description: str = Field(description="What this category means — shown to the judge")
    score: float = Field(description="Numeric score for aggregation (e.g. 1.0, 0.5, 0.0)")


class Dimension(BaseModel):
    """A single scoring dimension within a rubric.

    Uses categorical scale: the judge picks one of the named categories
    rather than choosing a number on a continuous scale.
    """

    name: str = Field(description="Dimension identifier, e.g. 'completeness'")
    description: str = Field(description="What this dimension measures")
    weight: float = Field(default=1.0, description="Relative weight for aggregation")
    scale: str = Field(default="categorical", description="Scale type — always 'categorical' for this format")
    categories: list[Category] = Field(description="Ordered scoring categories (best to worst)")

    @property
    def max_score(self) -> float:
        """Maximum possible score across categories."""
        if not self.categories:
            return 0.0
        return max(c.score for c in self.categories)

    @property
    def min_score(self) -> float:
        """Minimum possible score across categories."""
        if not self.categories:
            return 0.0
        return min(c.score for c in self.categories)

    def category_by_name(self, name: str) -> Category | None:
        """Look up a category by its name."""
        for cat in self.categories:
            if cat.name == name:
                return cat
        return None


class Rubric(BaseModel):
    """A categorical scoring rubric loaded from YAML or built programmatically.

    Rubrics define *what quality means* for a task type. Each dimension
    has named categories with descriptions so LLM judges have clear
    decision boundaries rather than vague numeric scales.
    """

    name: str = Field(description="Rubric identifier, e.g. 'extraction_quality'")
    version: str = Field(default="1.0", description="Semantic version string")
    description: str = Field(default="", description="What this rubric evaluates")
    dimensions: list[Dimension] = Field(description="Scoring dimensions")

    @property
    def dimension_names(self) -> list[str]:
        """List of dimension names in definition order."""
        return [d.name for d in self.dimensions]

    @property
    def total_weight(self) -> float:
        """Sum of all dimension weights."""
        return sum(d.weight for d in self.dimensions)

    def get_dimension(self, name: str) -> Dimension | None:
        """Get a dimension by name."""
        for d in self.dimensions:
            if d.name == name:
                return d
        return None

    def to_prompt_eval_rubric(self) -> Any:
        """Convert to prompt_eval.scoring.Rubric for LLM-judge scoring.

        Maps categorical dimensions to the numeric-scale format that
        prompt_eval's ascore_output() expects. Each category's description
        is appended to the dimension description as anchor text.

        Returns:
            prompt_eval.scoring.Rubric instance

        Raises:
            ImportError: If prompt_eval is not installed.
        """
        from prompt_eval.scoring import Rubric as PERubric, RubricCriterion

        criteria = []
        for dim in self.dimensions:
            # Build anchor text from categories
            anchor_lines = []
            for cat in dim.categories:
                anchor_lines.append(f"  - {cat.name} ({cat.score}): {cat.description}")
            desc_with_anchors = dim.description
            if anchor_lines:
                desc_with_anchors += "\nCategories:\n" + "\n".join(anchor_lines)

            # Scale = number of categories (maps to 1..N for the judge)
            scale = len(dim.categories) if dim.categories else 5

            criteria.append(
                RubricCriterion(
                    name=dim.name,
                    weight=dim.weight,
                    description=desc_with_anchors,
                    scale=scale,
                )
            )

        return PERubric(
            name=self.name,
            version=int(self.version.split(".")[0]) if "." in self.version else int(self.version),
            description=self.description,
            dimensions=criteria,
        )

    def score_categorical(self, selections: dict[str, str]) -> float:
        """Compute weighted overall score from category selections.

        Args:
            selections: Map of dimension_name -> category_name chosen by judge.

        Returns:
            Weighted score normalized to 0.0 - 1.0.

        Raises:
            ValueError: If a dimension or category name is not found.
        """
        total_weighted = 0.0
        total_weight = 0.0

        for dim in self.dimensions:
            cat_name = selections.get(dim.name)
            if cat_name is None:
                continue

            cat = dim.category_by_name(cat_name)
            if cat is None:
                raise ValueError(
                    f"Unknown category {cat_name!r} for dimension {dim.name!r}. "
                    f"Valid: {[c.name for c in dim.categories]}"
                )

            # Normalize category score to 0-1 range within this dimension
            score_range = dim.max_score - dim.min_score
            if score_range > 0:
                normalized = (cat.score - dim.min_score) / score_range
            else:
                normalized = cat.score

            total_weighted += normalized * dim.weight
            total_weight += dim.weight

        if total_weight == 0:
            return 0.0
        return total_weighted / total_weight


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class RubricRegistry:
    """Registry for YAML rubric definitions.

    Loads rubrics from one or more directories on disk and supports
    programmatic registration. Provides name-based lookup and listing.

    Resolution order for rubric directories:
    1. Any extra directories passed to the constructor (project-local)
    2. ``llm_client/rubrics/`` (shared library, ships with llm_client)
    """

    def __init__(self, rubric_dirs: list[Path] | None = None) -> None:
        """Initialize the registry and load rubrics from directories.

        Args:
            rubric_dirs: Additional directories to search (checked first).
                Pass project-local ``rubrics/`` paths here.
        """
        self._rubrics: dict[str, Rubric] = {}
        self._sources: dict[str, Path] = {}

        # Build search path: user dirs first, then built-in
        search_dirs: list[Path] = []
        if rubric_dirs:
            search_dirs.extend(rubric_dirs)
        search_dirs.append(_PACKAGE_RUBRICS_DIR)

        # Load in reverse order so that earlier dirs override later ones
        for d in reversed(search_dirs):
            self._load_dir(d)

    def _load_dir(self, directory: Path) -> None:
        """Load all YAML rubric files from a directory."""
        if not directory.is_dir():
            logger.debug("Rubric directory does not exist: %s", directory)
            return

        for path in sorted(directory.glob("*.yaml")):
            try:
                rubric = _load_rubric_file(path)
                self._rubrics[rubric.name] = rubric
                self._sources[rubric.name] = path
            except Exception:
                logger.warning("Failed to load rubric %s", path, exc_info=True)

    def get(self, name: str) -> Rubric:
        """Get a rubric by name.

        Args:
            name: Rubric identifier (e.g. "extraction_quality").

        Returns:
            The Rubric model.

        Raises:
            KeyError: If no rubric with that name is registered.
        """
        if name not in self._rubrics:
            raise KeyError(
                f"Rubric {name!r} not found. Available: {self.list()}"
            )
        return self._rubrics[name]

    def list(self) -> list[str]:
        """List available rubric names, sorted alphabetically."""
        return sorted(self._rubrics.keys())

    def register(self, rubric: Rubric) -> None:
        """Register a rubric programmatically.

        Overwrites any existing rubric with the same name.

        Args:
            rubric: The Rubric to register.
        """
        self._rubrics[rubric.name] = rubric
        self._sources[rubric.name] = Path("<programmatic>")
        logger.debug("Registered rubric %r (programmatic)", rubric.name)

    def source(self, name: str) -> Path | None:
        """Return the file path a rubric was loaded from, or None."""
        return self._sources.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._rubrics

    def __len__(self) -> int:
        return len(self._rubrics)

    def __repr__(self) -> str:
        return f"RubricRegistry({self.list()})"


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------


def _load_rubric_file(path: Path) -> Rubric:
    """Parse a rubric YAML file into a Rubric model.

    Handles both the new categorical format (with ``categories`` lists)
    and a lightweight format that only specifies ``scale: categorical``.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed Rubric model.
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    # Ensure version is a string
    if "version" in data and not isinstance(data["version"], str):
        data["version"] = str(data["version"])

    return Rubric(**data)


def load_rubric(name_or_path: str, extra_dirs: list[Path] | None = None) -> Rubric:
    """Convenience function: load a single rubric by name or file path.

    Args:
        name_or_path: Rubric name (e.g. "extraction_quality") or path to YAML.
        extra_dirs: Additional directories to search.

    Returns:
        Parsed Rubric model.

    Raises:
        FileNotFoundError: If rubric not found.
    """
    # Direct file path?
    p = Path(name_or_path)
    if p.is_file():
        return _load_rubric_file(p)

    # Ensure .yaml extension for name-based lookup
    fname = name_or_path if name_or_path.endswith(".yaml") else f"{name_or_path}.yaml"

    # Search extra dirs first, then built-in
    search_dirs: list[Path] = []
    if extra_dirs:
        search_dirs.extend(extra_dirs)
    # Project-local (cwd)
    search_dirs.append(Path.cwd() / "rubrics")
    # Built-in
    search_dirs.append(_PACKAGE_RUBRICS_DIR)

    for d in search_dirs:
        candidate = d / fname
        if candidate.is_file():
            return _load_rubric_file(candidate)

    searched = "\n  ".join(str(d / fname) for d in search_dirs)
    raise FileNotFoundError(
        f"Rubric {name_or_path!r} not found. Searched:\n  {searched}"
    )
