"""Audit repos for raw model literals and llm_client policy bypasses."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel

CODE_EXTENSIONS = {
    ".py",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".ini",
}

EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    "dist",
    "build",
}

DOC_DIR_NAMES = {
    "docs",
    "investigations",
}

TEST_DIR_NAMES = {
    "tests",
    "testdata",
    "fixtures",
}

LOW_SIGNAL_DIR_NAMES = {
    "archive",
    "PROJECTS_DEFERRED",
    "worktrees",
    "deprecated",
}

ALLOW_LINE_TOKENS = (
    "model-policy: allow-raw-model",
    "override_model",
    "fallback_model",
    "fallback_models",
    "benchmark_model",
)

CALL_RE = re.compile(
    r"\b(?:a?call_llm(?:_structured|_with_tools|_batch)?|stream_llm|astream_llm)"
    r"\(\s*(['\"])(?P<model>[^'\"]+)\1"
)

MODEL_LITERAL_RE = re.compile(
    r"(['\"])(?P<model>[^'\"]+)\1"
)


class PolicyViolation(BaseModel):
    path: str
    line: int
    kind: str
    model: str
    text: str


def _looks_like_model_id(value: str) -> bool:
    lower = value.lower()
    provider_prefixes = (
        "openrouter/",
        "openai/",
        "anthropic/",
        "gemini/",
        "google/",
        "ollama/",
        "x-ai/",
        "codex/",
        "claude-code/",
    )
    bare_prefixes = (
        "gpt-",
        "gemini-",
        "claude-",
        "grok-",
        "deepseek-",
        "o1-",
        "o3-",
    )
    if any(lower.startswith(prefix) and len(lower) > len(prefix) for prefix in provider_prefixes):
        return True
    return lower.startswith(bare_prefixes)


def _should_skip_file(
    path: Path,
    *,
    include_docs: bool,
    include_tests: bool,
) -> bool:
    parts = set(path.parts)
    if path.suffix not in CODE_EXTENSIONS:
        return True
    if parts & EXCLUDED_DIR_NAMES:
        return True
    if not include_docs and parts & DOC_DIR_NAMES:
        return True
    if not include_tests and parts & TEST_DIR_NAMES:
        return True
    if parts & LOW_SIGNAL_DIR_NAMES:
        return True
    return False


def _iter_candidate_files(
    roots: Iterable[Path],
    *,
    include_docs: bool,
    include_tests: bool,
) -> Iterable[Path]:
    for root in roots:
        if root.is_file():
            if not _should_skip_file(root, include_docs=include_docs, include_tests=include_tests):
                yield root
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if _should_skip_file(path, include_docs=include_docs, include_tests=include_tests):
                continue
            yield path


def scan_paths(
    roots: Iterable[Path],
    *,
    include_docs: bool = False,
    include_tests: bool = False,
) -> list[PolicyViolation]:
    violations: list[PolicyViolation] = []
    for path in _iter_candidate_files(
        roots,
        include_docs=include_docs,
        include_tests=include_tests,
    ):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(lines, start=1):
            if any(token in line for token in ALLOW_LINE_TOKENS):
                continue
            code_segment = line.split("#", 1)[0]
            stripped = code_segment.strip()
            if not stripped:
                continue
            call_match = CALL_RE.search(code_segment)
            if call_match and _looks_like_model_id(call_match.group("model")):
                violations.append(
                    PolicyViolation(
                        path=str(path),
                        line=line_no,
                        kind="direct_call_literal",
                        model=call_match.group("model"),
                        text=stripped,
                    )
                )
                continue
            for literal_match in MODEL_LITERAL_RE.finditer(code_segment):
                model = literal_match.group("model")
                if not _looks_like_model_id(model):
                    continue
                violations.append(
                    PolicyViolation(
                        path=str(path),
                        line=line_no,
                        kind="raw_model_literal",
                        model=model,
                        text=stripped,
                    )
                )
                break
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Repo paths to audit")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on violations")
    parser.add_argument("--include-docs", action="store_true", help="Scan docs/investigations")
    parser.add_argument("--include-tests", action="store_true", help="Scan tests and fixtures")
    args = parser.parse_args(argv)

    roots = [Path(item).resolve() for item in args.paths]
    violations = scan_paths(
        roots,
        include_docs=args.include_docs,
        include_tests=args.include_tests,
    )
    if not violations:
        print("MODEL POLICY OK")
        return 0

    print("MODEL POLICY VIOLATIONS")
    for violation in violations:
        print(
            f"{violation.path}:{violation.line}: {violation.kind}: "
            f"{violation.model} :: {violation.text}"
        )
    return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
