"""Tests for log maintenance script (rotation, stats, cleanup)."""

import gzip
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# Import the module under test
sys_path_entry = str(Path(__file__).resolve().parent.parent / "scripts")
import sys

sys.path.insert(0, sys_path_entry)
import log_maintenance


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Create a realistic log directory structure for testing."""
    # Project A: large monolithic calls.jsonl (will exceed threshold)
    proj_a = tmp_path / "project-a" / "project-a_llm_client_data"
    proj_a.mkdir(parents=True)
    _write_calls(proj_a / "calls.jsonl", n_lines=100, line_size=1024)

    # Project B: small calls.jsonl (under threshold)
    proj_b = tmp_path / "project-b" / "project-b_llm_client_data"
    proj_b.mkdir(parents=True)
    _write_calls(proj_b / "calls.jsonl", n_lines=5, line_size=100)

    # Project C: date-partitioned files only (no monolithic calls.jsonl)
    proj_c = tmp_path / "project-c" / "project-c_llm_client_data"
    proj_c.mkdir(parents=True)
    _write_calls(proj_c / "calls_2026-03-21.jsonl", n_lines=10, line_size=200)
    _write_calls(proj_c / "calls_2026-03-22.jsonl", n_lines=10, line_size=200)

    # Project D: has experiments.jsonl and foundation_events.jsonl too
    proj_d = tmp_path / "project-d" / "project-d_llm_client_data"
    proj_d.mkdir(parents=True)
    _write_calls(proj_d / "calls.jsonl", n_lines=10, line_size=200)
    _write_calls(proj_d / "experiments.jsonl", n_lines=5, line_size=100)

    return tmp_path


def _write_calls(path: Path, n_lines: int, line_size: int) -> None:
    """Write synthetic JSONL lines with timestamps and padding."""
    base_time = datetime(2026, 3, 1, tzinfo=timezone.utc)
    with open(path, "w") as f:
        for i in range(n_lines):
            ts = (base_time + timedelta(hours=i)).isoformat()
            record = {
                "timestamp": ts,
                "model": "test-model",
                "task": "test-task",
                "padding": "x" * max(0, line_size - 120),
            }
            line = json.dumps(record)
            f.write(line + "\n")


# ── Stats tests ─────────────────────────────────────────────────────────────


class TestStats:
    def test_stats_finds_all_projects(self, data_dir: Path, capsys: pytest.CaptureFixture) -> None:
        log_maintenance.cmd_stats(data_dir, verbose=True)
        out = capsys.readouterr().out
        assert "project-a" in out
        assert "project-b" in out
        assert "project-c" in out
        assert "project-d" in out
        assert "TOTAL" in out

    def test_stats_shows_totals(self, data_dir: Path, capsys: pytest.CaptureFixture) -> None:
        log_maintenance.cmd_stats(data_dir, verbose=True)
        out = capsys.readouterr().out
        assert "4 projects" in out

    def test_stats_empty_dir(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        log_maintenance.cmd_stats(tmp_path)
        out = capsys.readouterr().out
        assert "No log directories" in out


# ── Rotate tests ────────────────────────────────────────────────────────────


class TestRotate:
    def test_rotate_large_file(self, data_dir: Path) -> None:
        calls_file = data_dir / "project-a" / "project-a_llm_client_data" / "calls.jsonl"
        original_size = calls_file.stat().st_size
        # Set threshold low enough to trigger rotation
        threshold_mb = (original_size / (1024 * 1024)) * 0.5  # half the file size

        rotated = log_maintenance.cmd_rotate(data_dir, max_size_mb=threshold_mb)

        # Original file should now be empty (recreated)
        assert calls_file.exists()
        assert calls_file.stat().st_size == 0

        # Should have created a .gz file
        assert len(rotated) == 1
        assert rotated[0].suffix == ".gz"
        assert rotated[0].exists()

    def test_rotate_skips_small_files(self, data_dir: Path) -> None:
        rotated = log_maintenance.cmd_rotate(data_dir, max_size_mb=9999)
        assert len(rotated) == 0

    def test_rotate_dry_run_no_changes(self, data_dir: Path) -> None:
        calls_file = data_dir / "project-a" / "project-a_llm_client_data" / "calls.jsonl"
        original_size = calls_file.stat().st_size

        log_maintenance.cmd_rotate(data_dir, max_size_mb=0.001, dry_run=True)

        # File should be unchanged
        assert calls_file.stat().st_size == original_size

    def test_rotate_does_not_touch_date_partitioned(self, data_dir: Path) -> None:
        """Date-partitioned files (calls_2026-03-21.jsonl) should not be rotated."""
        rotated = log_maintenance.cmd_rotate(data_dir, max_size_mb=0.001)

        # Only project-a and project-d have monolithic calls.jsonl
        # project-c only has date-partitioned files
        rotated_parents = {r.parent.parent.name for r in rotated}
        assert "project-c" not in rotated_parents

    def test_rotate_prunes_old_archives(self, data_dir: Path) -> None:
        log_dir = data_dir / "project-a" / "project-a_llm_client_data"

        # Create some fake old archives
        for i in range(7):
            fake_gz = log_dir / f"calls.2026010{i}_000000.jsonl.gz"
            fake_gz.write_bytes(b"fake")

        # Rotate with max_rotated=2
        log_maintenance.cmd_rotate(data_dir, max_size_mb=0.001, max_rotated=2)

        # Should have at most 2 .gz files remaining
        gz_files = list(log_dir.glob("calls.*.jsonl.gz"))
        assert len(gz_files) <= 2

    def test_rotated_gz_is_valid(self, data_dir: Path) -> None:
        """Verify the gzipped content can be decompressed and parsed."""
        calls_file = data_dir / "project-a" / "project-a_llm_client_data" / "calls.jsonl"

        # Read original content before rotation
        with open(calls_file, "r") as f:
            original_lines = f.readlines()

        rotated = log_maintenance.cmd_rotate(data_dir, max_size_mb=0.001)
        assert len(rotated) >= 1

        gz_path = rotated[0]
        with gzip.open(gz_path, "rt") as f:
            restored_lines = f.readlines()

        assert len(restored_lines) == len(original_lines)
        # Verify first record parses
        first_record = json.loads(restored_lines[0])
        assert "timestamp" in first_record


# ── Cleanup tests ───────────────────────────────────────────────────────────


class TestCleanup:
    def test_cleanup_archives_old_files(self, data_dir: Path) -> None:
        # Make project-b's file appear old
        calls_file = data_dir / "project-b" / "project-b_llm_client_data" / "calls.jsonl"
        # Ensure it's above the 10KB skip threshold
        _write_calls(calls_file, n_lines=50, line_size=500)
        old_time = time.time() - (100 * 86400)  # 100 days ago
        os.utime(calls_file, (old_time, old_time))

        result = log_maintenance.cmd_cleanup(data_dir, archive_days=90)

        assert len(result["archived"]) >= 1
        # Original should be gone
        assert not calls_file.exists()
        # .gz should exist
        gz_path = calls_file.with_suffix(".jsonl.gz")
        assert gz_path.exists()

    def test_cleanup_skips_recent_files(self, data_dir: Path) -> None:
        result = log_maintenance.cmd_cleanup(data_dir, archive_days=90)
        # All test files are freshly created, so nothing should be archived
        assert len(result["archived"]) == 0

    def test_cleanup_skips_tiny_files(self, data_dir: Path) -> None:
        # Make a tiny old file
        log_dir = data_dir / "project-b" / "project-b_llm_client_data"
        tiny = log_dir / "tiny.jsonl"
        tiny.write_text('{"test": true}\n')
        old_time = time.time() - (200 * 86400)
        os.utime(tiny, (old_time, old_time))

        result = log_maintenance.cmd_cleanup(data_dir, archive_days=90)
        # Tiny file should not be archived
        assert tiny.exists()

    def test_cleanup_delete_phase_opt_in(self, data_dir: Path) -> None:
        # Create an old .gz archive
        log_dir = data_dir / "project-a" / "project-a_llm_client_data"
        old_gz = log_dir / "calls.20250101_000000.jsonl.gz"
        with gzip.open(old_gz, "wt") as f:
            f.write('{"test": true}\n')
        old_time = time.time() - (200 * 86400)
        os.utime(old_gz, (old_time, old_time))

        # Without delete_days: archive stays
        result = log_maintenance.cmd_cleanup(data_dir, archive_days=90, delete_days=None)
        assert old_gz.exists()

        # With delete_days: archive gets deleted
        result = log_maintenance.cmd_cleanup(data_dir, archive_days=90, delete_days=180)
        assert len(result["deleted"]) >= 1
        assert not old_gz.exists()

    def test_cleanup_dry_run(self, data_dir: Path) -> None:
        calls_file = data_dir / "project-b" / "project-b_llm_client_data" / "calls.jsonl"
        _write_calls(calls_file, n_lines=50, line_size=500)
        old_time = time.time() - (100 * 86400)
        os.utime(calls_file, (old_time, old_time))

        original_size = calls_file.stat().st_size
        result = log_maintenance.cmd_cleanup(data_dir, archive_days=90, dry_run=True)

        # File should be unchanged
        assert calls_file.exists()
        assert calls_file.stat().st_size == original_size


# ── Helper function tests ──────────────────────────────────────────────────


class TestHelpers:
    def test_find_log_dirs(self, data_dir: Path) -> None:
        dirs = log_maintenance._find_log_dirs(data_dir)
        names = {d.name for d in dirs}
        assert "project-a_llm_client_data" in names
        assert "project-b_llm_client_data" in names

    def test_line_count(self, data_dir: Path) -> None:
        calls = data_dir / "project-a" / "project-a_llm_client_data" / "calls.jsonl"
        assert log_maintenance._line_count(calls) == 100

    def test_date_range(self, data_dir: Path) -> None:
        calls = data_dir / "project-a" / "project-a_llm_client_data" / "calls.jsonl"
        first, last = log_maintenance._date_range(calls)
        assert first is not None
        assert last is not None
        assert "2026-03-01" in first
        # Last entry is 99 hours later
        assert first < last

    def test_format_size(self) -> None:
        assert "1.0KB" == log_maintenance._format_size(1024)
        assert "1.0MB" == log_maintenance._format_size(1024 * 1024)
        assert "1.0GB" == log_maintenance._format_size(1024 * 1024 * 1024)
        assert "500B" == log_maintenance._format_size(500)

    def test_project_name_from_dir(self, data_dir: Path) -> None:
        log_dir = data_dir / "project-a" / "project-a_llm_client_data"
        assert log_maintenance._project_name_from_dir(log_dir) == "project-a"


# ── CLI tests ──────────────────────────────────────────────────────────────


class TestCLI:
    def test_cli_stats(self, data_dir: Path) -> None:
        assert log_maintenance.main(["--data-dir", str(data_dir), "stats"]) == 0

    def test_cli_rotate_dry_run(self, data_dir: Path) -> None:
        assert log_maintenance.main(["--data-dir", str(data_dir), "rotate", "--dry-run"]) == 0

    def test_cli_cleanup_dry_run(self, data_dir: Path) -> None:
        assert log_maintenance.main(
            ["--data-dir", str(data_dir), "cleanup", "--days", "90", "--dry-run"]
        ) == 0

    def test_cli_rotate_custom_threshold(self, data_dir: Path) -> None:
        assert log_maintenance.main(
            ["--data-dir", str(data_dir), "rotate", "--max-size", "50"]
        ) == 0
