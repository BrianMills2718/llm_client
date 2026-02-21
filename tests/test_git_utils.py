"""Tests for llm_client.git_utils."""

from llm_client.git_utils import (
    CODE_CHANGE,
    CONFIG_CHANGE,
    PROMPT_CHANGE,
    RUBRIC_CHANGE,
    TEST_CHANGE,
    classify_diff_files,
    get_git_head,
    get_working_tree_files,
    is_git_dirty,
)


# ---------------------------------------------------------------------------
# classify_diff_files — pure logic, no mocking needed
# ---------------------------------------------------------------------------


def test_classify_prompt_files():
    files = ["prompts/extract.yaml", "prompts/judge.yaml"]
    assert classify_diff_files(files) == {PROMPT_CHANGE}


def test_classify_rubric_files():
    files = ["rubrics/research_quality.yaml"]
    assert classify_diff_files(files) == {RUBRIC_CHANGE}


def test_classify_code_files():
    files = ["llm_client/analyzer.py", "llm_client/client.py"]
    assert classify_diff_files(files) == {CODE_CHANGE}


def test_classify_config_files():
    files = ["config.yaml", "pyproject.toml", "settings.json"]
    assert classify_diff_files(files) == {CONFIG_CHANGE}


def test_classify_test_files():
    files = ["tests/test_analyzer.py", "test_utils.py"]
    assert classify_diff_files(files) == {TEST_CHANGE}


def test_classify_mixed():
    files = [
        "prompts/extract.yaml",
        "llm_client/analyzer.py",
        "tests/test_analyzer.py",
        "config.yaml",
    ]
    result = classify_diff_files(files)
    assert result == {PROMPT_CHANGE, CODE_CHANGE, TEST_CHANGE, CONFIG_CHANGE}


def test_classify_empty():
    assert classify_diff_files([]) == set()


def test_classify_nested_prompts():
    files = ["llm_client/prompts/rubric_judge.yaml"]
    assert classify_diff_files(files) == {PROMPT_CHANGE}


# ---------------------------------------------------------------------------
# get_git_head — run in actual llm_client repo
# ---------------------------------------------------------------------------


def test_get_git_head_returns_string():
    """llm_client is a git repo, so this should return a short SHA."""
    result = get_git_head()
    assert result is not None
    assert len(result) >= 7  # short SHA is typically 7+ chars


def test_get_git_head_nonexistent_dir():
    result = get_git_head(cwd="/nonexistent/path/that/does/not/exist")
    assert result is None


def test_get_working_tree_files_type():
    files = get_working_tree_files()
    assert isinstance(files, list)
    assert all(isinstance(f, str) for f in files)


def test_get_working_tree_files_nonexistent_dir():
    files = get_working_tree_files(cwd="/nonexistent/path/that/does/not/exist")
    assert files == []


def test_is_git_dirty_returns_bool():
    dirty = is_git_dirty()
    assert isinstance(dirty, bool)


def test_is_git_dirty_nonexistent_dir():
    dirty = is_git_dirty(cwd="/nonexistent/path/that/does/not/exist")
    assert dirty is False
