"""Tests for the rubric registry — categorical scoring rubrics.

Validates YAML loading, registry operations, scoring arithmetic,
and compatibility with the prompt_eval scoring bridge.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_client.rubric_registry import (
    Category,
    Dimension,
    Rubric,
    RubricRegistry,
    _load_rubric_file,
    load_rubric,
)

# Path to the built-in rubrics shipped with llm_client
_RUBRICS_DIR = Path(__file__).resolve().parent.parent / "rubrics"


# ---------------------------------------------------------------------------
# Model unit tests
# ---------------------------------------------------------------------------


class TestCategory:
    """Category model basics."""

    def test_create(self) -> None:
        cat = Category(name="good", description="It is good", score=1.0)
        assert cat.name == "good"
        assert cat.score == 1.0


class TestDimension:
    """Dimension model with categories."""

    @pytest.fixture()
    def dim(self) -> Dimension:
        return Dimension(
            name="accuracy",
            description="Is it accurate?",
            weight=1.0,
            scale="categorical",
            categories=[
                Category(name="accurate", description="all correct", score=1.0),
                Category(name="partial", description="some errors", score=0.5),
                Category(name="wrong", description="mostly wrong", score=0.0),
            ],
        )

    def test_max_score(self, dim: Dimension) -> None:
        assert dim.max_score == 1.0

    def test_min_score(self, dim: Dimension) -> None:
        assert dim.min_score == 0.0

    def test_category_by_name(self, dim: Dimension) -> None:
        cat = dim.category_by_name("partial")
        assert cat is not None
        assert cat.score == 0.5

    def test_category_by_name_missing(self, dim: Dimension) -> None:
        assert dim.category_by_name("nonexistent") is None


class TestRubric:
    """Rubric model and scoring."""

    @pytest.fixture()
    def rubric(self) -> Rubric:
        return Rubric(
            name="test_rubric",
            version="1.0",
            description="A test rubric",
            dimensions=[
                Dimension(
                    name="quality",
                    description="Overall quality",
                    weight=2.0,
                    categories=[
                        Category(name="high", description="high quality", score=1.0),
                        Category(name="medium", description="medium quality", score=0.5),
                        Category(name="low", description="low quality", score=0.0),
                    ],
                ),
                Dimension(
                    name="style",
                    description="Style adherence",
                    weight=1.0,
                    categories=[
                        Category(name="good", description="good style", score=1.0),
                        Category(name="bad", description="bad style", score=0.0),
                    ],
                ),
            ],
        )

    def test_dimension_names(self, rubric: Rubric) -> None:
        assert rubric.dimension_names == ["quality", "style"]

    def test_total_weight(self, rubric: Rubric) -> None:
        assert rubric.total_weight == 3.0

    def test_get_dimension(self, rubric: Rubric) -> None:
        dim = rubric.get_dimension("quality")
        assert dim is not None
        assert dim.weight == 2.0

    def test_get_dimension_missing(self, rubric: Rubric) -> None:
        assert rubric.get_dimension("nonexistent") is None

    def test_score_categorical_all_best(self, rubric: Rubric) -> None:
        """All best categories should produce 1.0."""
        score = rubric.score_categorical({"quality": "high", "style": "good"})
        assert score == pytest.approx(1.0)

    def test_score_categorical_all_worst(self, rubric: Rubric) -> None:
        """All worst categories should produce 0.0."""
        score = rubric.score_categorical({"quality": "low", "style": "bad"})
        assert score == pytest.approx(0.0)

    def test_score_categorical_mixed(self, rubric: Rubric) -> None:
        """Mixed selections: quality=medium (0.5, weight 2), style=good (1.0, weight 1).

        Weighted average: (0.5*2 + 1.0*1) / 3 = 2.0/3 ~ 0.6667
        """
        score = rubric.score_categorical({"quality": "medium", "style": "good"})
        assert score == pytest.approx(2.0 / 3.0)

    def test_score_categorical_partial_dims(self, rubric: Rubric) -> None:
        """Supplying only one dimension should score just that dimension."""
        score = rubric.score_categorical({"quality": "high"})
        assert score == pytest.approx(1.0)

    def test_score_categorical_unknown_category(self, rubric: Rubric) -> None:
        with pytest.raises(ValueError, match="Unknown category"):
            rubric.score_categorical({"quality": "unknown"})

    def test_score_categorical_empty(self, rubric: Rubric) -> None:
        """No selections should return 0.0."""
        assert rubric.score_categorical({}) == 0.0


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


class TestYAMLLoading:
    """Loading rubric files from disk."""

    def test_load_extraction_quality(self) -> None:
        rubric = _load_rubric_file(_RUBRICS_DIR / "extraction_quality.yaml")
        assert rubric.name == "extraction_quality"
        assert rubric.version == "1.0"
        assert len(rubric.dimensions) == 4
        # Check categories exist on first dimension
        assert len(rubric.dimensions[0].categories) == 3

    def test_load_research_quality(self) -> None:
        rubric = _load_rubric_file(_RUBRICS_DIR / "research_quality.yaml")
        assert rubric.name == "research_quality"
        assert len(rubric.dimensions) == 5

    def test_load_code_quality(self) -> None:
        rubric = _load_rubric_file(_RUBRICS_DIR / "code_quality.yaml")
        assert rubric.name == "code_quality"
        assert len(rubric.dimensions) == 4

    def test_load_summary_quality(self) -> None:
        rubric = _load_rubric_file(_RUBRICS_DIR / "summary_quality.yaml")
        assert rubric.name == "summary_quality"
        assert len(rubric.dimensions) == 4

    def test_all_rubrics_score_perfectly(self) -> None:
        """Every rubric should score 1.0 when all best categories are picked."""
        for yaml_file in _RUBRICS_DIR.glob("*.yaml"):
            rubric = _load_rubric_file(yaml_file)
            best_selections = {}
            for dim in rubric.dimensions:
                # Find the category with the highest score
                best_cat = max(dim.categories, key=lambda c: c.score)
                best_selections[dim.name] = best_cat.name
            score = rubric.score_categorical(best_selections)
            assert score == pytest.approx(1.0), f"{rubric.name} best-case != 1.0"

    def test_all_rubrics_score_zero(self) -> None:
        """Every rubric should score 0.0 when all worst categories are picked."""
        for yaml_file in _RUBRICS_DIR.glob("*.yaml"):
            rubric = _load_rubric_file(yaml_file)
            worst_selections = {}
            for dim in rubric.dimensions:
                worst_cat = min(dim.categories, key=lambda c: c.score)
                worst_selections[dim.name] = worst_cat.name
            score = rubric.score_categorical(worst_selections)
            assert score == pytest.approx(0.0), f"{rubric.name} worst-case != 0.0"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRubricRegistry:
    """Registry loading and lookup."""

    @pytest.fixture()
    def registry(self) -> RubricRegistry:
        return RubricRegistry()

    def test_list_includes_builtin(self, registry: RubricRegistry) -> None:
        names = registry.list()
        assert "extraction_quality" in names
        assert "research_quality" in names
        assert "code_quality" in names
        assert "summary_quality" in names

    def test_get(self, registry: RubricRegistry) -> None:
        rubric = registry.get("extraction_quality")
        assert rubric.name == "extraction_quality"
        assert len(rubric.dimensions) > 0

    def test_get_missing_raises(self, registry: RubricRegistry) -> None:
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent_rubric")

    def test_contains(self, registry: RubricRegistry) -> None:
        assert "extraction_quality" in registry
        assert "nonexistent" not in registry

    def test_len(self, registry: RubricRegistry) -> None:
        assert len(registry) >= 4

    def test_register_programmatic(self, registry: RubricRegistry) -> None:
        custom = Rubric(
            name="custom_test",
            version="1.0",
            description="test",
            dimensions=[
                Dimension(
                    name="x",
                    description="x",
                    categories=[
                        Category(name="yes", description="yes", score=1.0),
                        Category(name="no", description="no", score=0.0),
                    ],
                )
            ],
        )
        registry.register(custom)
        assert "custom_test" in registry
        assert registry.get("custom_test").name == "custom_test"

    def test_register_overrides(self, registry: RubricRegistry) -> None:
        """Programmatic registration should override loaded rubrics."""
        override = Rubric(
            name="extraction_quality",
            version="99.0",
            description="overridden",
            dimensions=[],
        )
        registry.register(override)
        assert registry.get("extraction_quality").version == "99.0"

    def test_source_builtin(self, registry: RubricRegistry) -> None:
        src = registry.source("extraction_quality")
        assert src is not None
        assert "rubrics" in str(src)

    def test_source_programmatic(self, registry: RubricRegistry) -> None:
        custom = Rubric(name="prog", version="1.0", dimensions=[])
        registry.register(custom)
        assert registry.source("prog") == Path("<programmatic>")

    def test_extra_dirs_override(self, tmp_path: Path) -> None:
        """Rubrics from extra_dirs should override built-in ones."""
        # Write a minimal rubric that overrides extraction_quality
        rubric_file = tmp_path / "extraction_quality.yaml"
        rubric_file.write_text(
            "name: extraction_quality\n"
            "version: '99.0'\n"
            "description: overridden\n"
            "dimensions: []\n"
        )
        registry = RubricRegistry(rubric_dirs=[tmp_path])
        assert registry.get("extraction_quality").version == "99.0"


# ---------------------------------------------------------------------------
# Convenience load_rubric function
# ---------------------------------------------------------------------------


class TestLoadRubric:
    """The standalone load_rubric() function."""

    def test_load_by_name(self) -> None:
        rubric = load_rubric("extraction_quality")
        assert rubric.name == "extraction_quality"

    def test_load_by_name_with_extension(self) -> None:
        rubric = load_rubric("extraction_quality.yaml")
        assert rubric.name == "extraction_quality"

    def test_load_by_path(self) -> None:
        path = str(_RUBRICS_DIR / "code_quality.yaml")
        rubric = load_rubric(path)
        assert rubric.name == "code_quality"

    def test_load_missing_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_rubric("this_rubric_does_not_exist")

    def test_load_from_extra_dir(self, tmp_path: Path) -> None:
        rubric_file = tmp_path / "custom.yaml"
        rubric_file.write_text(
            "name: custom\n"
            "version: '1.0'\n"
            "description: test\n"
            "dimensions:\n"
            "  - name: x\n"
            "    description: x\n"
            "    categories:\n"
            "      - name: 'okay'\n"
            "        description: 'it is okay'\n"
            "        score: 1.0\n"
        )
        rubric = load_rubric("custom", extra_dirs=[tmp_path])
        assert rubric.name == "custom"


# ---------------------------------------------------------------------------
# prompt_eval bridge
# ---------------------------------------------------------------------------


class TestPromptEvalBridge:
    """Test conversion to prompt_eval.scoring.Rubric format."""

    def test_to_prompt_eval_rubric(self) -> None:
        """Should convert categorical dimensions to prompt_eval's numeric format."""
        try:
            from prompt_eval.scoring import Rubric as PERubric
        except ImportError:
            pytest.skip("prompt_eval not installed")

        rubric = load_rubric("extraction_quality")
        pe_rubric = rubric.to_prompt_eval_rubric()

        assert isinstance(pe_rubric, PERubric)
        assert pe_rubric.name == "extraction_quality"
        assert len(pe_rubric.dimensions) == len(rubric.dimensions)

        # Verify each dimension has anchor text from categories
        for pe_dim in pe_rubric.dimensions:
            assert "Categories:" in pe_dim.description
