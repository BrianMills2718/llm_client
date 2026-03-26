#!/usr/bin/env python3
"""Log maintenance for llm_client observability logs.

Handles rotation, statistics, and cleanup of JSONL logs that accumulate
in ~/projects/data/{project}/{project}_llm_client_data/calls.jsonl.

Heavy-use projects (e.g., Digimon at 441MB) grow unbounded without this.
"""

import argparse
import gzip
import json
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Defaults (all overridable via CLI) ──────────────────────────────────────

DATA_DIR = Path(os.environ.get("LLM_CLIENT_DATA_DIR", str(Path.home() / "projects" / "data")))
MAX_SIZE_MB = int(os.environ.get("LOG_MAX_SIZE_MB", "100"))
MAX_ROTATED = int(os.environ.get("LOG_MAX_ROTATED", "5"))
ARCHIVE_DAYS = int(os.environ.get("LOG_ARCHIVE_DAYS", "90"))
DELETE_DAYS = int(os.environ.get("LOG_DELETE_DAYS", "180"))

# Pattern: {project}_llm_client_data/calls.jsonl (monolithic)
# Pattern: {project}_llm_client_data/calls_{date}.jsonl (date-partitioned)
CALLS_GLOB = "calls*.jsonl"


def _find_log_dirs(data_dir: Path) -> list[Path]:
    """Find all *_llm_client_data directories under data_dir."""
    return sorted(data_dir.rglob("*_llm_client_data"))


def _find_jsonl_files(log_dir: Path) -> list[Path]:
    """Find all JSONL files in a log directory (calls, experiments, foundation_events)."""
    return sorted(log_dir.glob("*.jsonl"))


def _find_calls_files(log_dir: Path) -> list[Path]:
    """Find calls JSONL files (both monolithic and date-partitioned)."""
    return sorted(log_dir.glob(CALLS_GLOB))


def _file_size_mb(path: Path) -> float:
    """File size in megabytes."""
    return path.stat().st_size / (1024 * 1024)


def _line_count(path: Path) -> int:
    """Count lines in a file without loading it all into memory."""
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


def _date_range(path: Path) -> tuple[str | None, str | None]:
    """Extract first and last timestamp from a JSONL file.

    Reads only the first and last lines to avoid loading the whole file.
    """
    first_ts = None
    last_ts = None

    try:
        with open(path, "r") as f:
            # First line
            first_line = f.readline().strip()
            if first_line:
                try:
                    first_ts = json.loads(first_line).get("timestamp", "")[:19]
                except (json.JSONDecodeError, KeyError):
                    pass

            # Seek to end and scan backwards for last line
            f.seek(0, 2)
            pos = f.tell()
            if pos == 0:
                return first_ts, first_ts

            # Read backwards to find last newline
            pos -= 1
            while pos > 0:
                f.seek(pos)
                ch = f.read(1)
                if ch == "\n" and pos < f.seek(0, 2) - 1:
                    break
                pos -= 1

            if pos > 0:
                f.seek(pos + 1)
            else:
                f.seek(0)

            last_line = f.readline().strip()
            if last_line:
                try:
                    last_ts = json.loads(last_line).get("timestamp", "")[:19]
                except (json.JSONDecodeError, KeyError):
                    pass
    except (OSError, IOError):
        pass

    return first_ts, last_ts


def _project_name_from_dir(log_dir: Path) -> str:
    """Extract project name from the _llm_client_data directory path."""
    # Parent dir is the project dir under data/
    return log_dir.parent.name


def _format_size(size_bytes: int) -> str:
    """Human-readable file size."""
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / (1024**3):.1f}GB"
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024**2):.1f}MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f}KB"
    return f"{size_bytes}B"


# ── Commands ────────────────────────────────────────────────────────────────


def cmd_stats(data_dir: Path, *, verbose: bool = False) -> None:
    """Report log file sizes, line counts, and date ranges per project."""
    log_dirs = _find_log_dirs(data_dir)
    if not log_dirs:
        print(f"No log directories found under {data_dir}")
        return

    total_bytes = 0
    total_lines = 0
    total_files = 0
    rows: list[tuple[str, int, int, str | None, str | None, int]] = []

    for log_dir in log_dirs:
        jsonl_files = _find_jsonl_files(log_dir)
        if not jsonl_files:
            continue

        project = _project_name_from_dir(log_dir)
        dir_bytes = sum(f.stat().st_size for f in jsonl_files)
        dir_lines = 0
        earliest = None
        latest = None

        for f in jsonl_files:
            lines = _line_count(f)
            dir_lines += lines

            if f.name.startswith("calls"):
                first, last = _date_range(f)
                if first and (earliest is None or first < earliest):
                    earliest = first
                if last and (latest is None or last > latest):
                    latest = last

        total_bytes += dir_bytes
        total_lines += dir_lines
        total_files += len(jsonl_files)
        rows.append((project, dir_bytes, dir_lines, earliest, latest, len(jsonl_files)))

    # Sort by size descending
    rows.sort(key=lambda r: r[1], reverse=True)

    # Print header
    print(f"{'Project':<50} {'Size':>10} {'Lines':>10} {'Files':>6} {'Date Range'}")
    print("-" * 110)

    # In non-verbose mode, only show projects > 1MB or top 20
    display_rows = rows if verbose else [r for r in rows if r[1] >= 1024 * 1024]
    if not verbose and len(display_rows) < 20:
        display_rows = rows[:20]

    for project, size, lines, earliest, latest, n_files in display_rows:
        date_range = ""
        if earliest and latest:
            date_range = f"{earliest[:10]} .. {latest[:10]}"
        elif earliest:
            date_range = f"{earliest[:10]}"

        print(f"{project:<50} {_format_size(size):>10} {lines:>10,} {n_files:>6} {date_range}")

    if not verbose and len(rows) > len(display_rows):
        print(f"  ... {len(rows) - len(display_rows)} smaller projects omitted (use --verbose)")

    # Totals
    print("-" * 110)
    print(f"{'TOTAL':<50} {_format_size(total_bytes):>10} {total_lines:>10,} {total_files:>6}")
    print(f"\n{len(rows)} projects, {total_files} files, {_format_size(total_bytes)} total")

    # Compressed file stats
    gz_files = list(data_dir.rglob("*.jsonl.gz"))
    if gz_files:
        gz_bytes = sum(f.stat().st_size for f in gz_files)
        print(f"{len(gz_files)} compressed archives, {_format_size(gz_bytes)} total")


def cmd_rotate(
    data_dir: Path,
    *,
    max_size_mb: float = MAX_SIZE_MB,
    max_rotated: int = MAX_ROTATED,
    dry_run: bool = False,
) -> list[Path]:
    """Rotate oversized calls.jsonl files.

    For each calls.jsonl exceeding max_size_mb:
    1. Rename to calls.{timestamp}.jsonl
    2. Compress with gzip
    3. Create empty calls.jsonl
    4. Prune old rotated files beyond max_rotated

    Returns list of files that were rotated.
    """
    rotated: list[Path] = []
    log_dirs = _find_log_dirs(data_dir)

    for log_dir in log_dirs:
        # Only rotate monolithic calls.jsonl (not date-partitioned ones)
        calls_file = log_dir / "calls.jsonl"
        if not calls_file.exists():
            continue

        size_mb = _file_size_mb(calls_file)
        if size_mb < max_size_mb:
            continue

        project = _project_name_from_dir(log_dir)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rotated_name = f"calls.{ts}.jsonl"
        rotated_path = log_dir / rotated_name
        gz_path = log_dir / f"{rotated_name}.gz"

        print(f"[rotate] {project}: {_format_size(calls_file.stat().st_size)} -> {gz_path.name}")

        if not dry_run:
            # Rename current file
            shutil.move(str(calls_file), str(rotated_path))

            # Compress
            with open(rotated_path, "rb") as f_in:
                with gzip.open(gz_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Remove uncompressed rotated copy
            rotated_path.unlink()

            # Create fresh empty file
            calls_file.touch()

            print(f"         compressed: {_format_size(gz_path.stat().st_size)}")
            rotated.append(gz_path)

        # Prune old rotated files
        existing_gz = sorted(log_dir.glob("calls.*.jsonl.gz"), reverse=True)
        for old_gz in existing_gz[max_rotated:]:
            print(f"  [prune] {project}: removing old archive {old_gz.name}")
            if not dry_run:
                old_gz.unlink()

    if not rotated and not dry_run:
        print(f"No files exceed {max_size_mb}MB threshold. Nothing to rotate.")

    return rotated


def cmd_cleanup(
    data_dir: Path,
    *,
    archive_days: int = ARCHIVE_DAYS,
    delete_days: int | None = None,
    dry_run: bool = False,
) -> dict[str, list[Path]]:
    """Archive old log files and optionally delete aged archives.

    Phase 1 (archive): JSONL files with last-modified > archive_days ago
    get compressed to .jsonl.gz alongside the original, then the original
    is removed.

    Phase 2 (delete, opt-in): .jsonl.gz files older than delete_days get
    removed. Only runs if --delete-days is explicitly passed.

    Returns dict with 'archived' and 'deleted' lists.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=archive_days)
    result: dict[str, list[Path]] = {"archived": [], "deleted": []}

    log_dirs = _find_log_dirs(data_dir)

    for log_dir in log_dirs:
        project = _project_name_from_dir(log_dir)

        for jsonl_file in _find_jsonl_files(log_dir):
            # Skip already-tiny files (< 10KB)
            if jsonl_file.stat().st_size < 10240:
                continue

            # Skip if a .gz already exists for this file
            gz_path = jsonl_file.with_suffix(jsonl_file.suffix + ".gz")
            if gz_path.exists():
                continue

            # Check modification time
            mtime = datetime.fromtimestamp(jsonl_file.stat().st_mtime, tz=timezone.utc)
            if mtime >= cutoff:
                continue

            print(f"[archive] {project}/{jsonl_file.name}: "
                  f"{_format_size(jsonl_file.stat().st_size)}, "
                  f"last modified {mtime.strftime('%Y-%m-%d')}")

            if not dry_run:
                with open(jsonl_file, "rb") as f_in:
                    with gzip.open(gz_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                jsonl_file.unlink()
                print(f"         -> {gz_path.name} ({_format_size(gz_path.stat().st_size)})")
                result["archived"].append(gz_path)

    # Phase 2: delete old archives (opt-in only)
    if delete_days is not None:
        delete_cutoff = datetime.now(timezone.utc) - timedelta(days=delete_days)
        for gz_file in sorted(data_dir.rglob("*.jsonl.gz")):
            mtime = datetime.fromtimestamp(gz_file.stat().st_mtime, tz=timezone.utc)
            if mtime >= delete_cutoff:
                continue

            project = _project_name_from_dir(gz_file.parent)
            print(f"[delete] {project}/{gz_file.name}: "
                  f"{_format_size(gz_file.stat().st_size)}, "
                  f"last modified {mtime.strftime('%Y-%m-%d')}")

            if not dry_run:
                gz_file.unlink()
                result["deleted"].append(gz_file)

    if not result["archived"] and not result["deleted"] and not dry_run:
        print(f"No files older than {archive_days} days found. Nothing to archive.")

    return result


# ── CLI ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Log maintenance for llm_client observability JSONL logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s stats                      Show log sizes per project
  %(prog)s stats --verbose            Include all projects (even tiny ones)
  %(prog)s rotate                     Rotate files > 100MB (default)
  %(prog)s rotate --max-size 50       Rotate files > 50MB
  %(prog)s rotate --dry-run           Show what would be rotated
  %(prog)s cleanup --days 90          Archive logs older than 90 days
  %(prog)s cleanup --days 90 --delete-days 180  Also delete archives > 180 days
""",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Root data directory (default: {DATA_DIR})",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # stats
    p_stats = sub.add_parser("stats", help="Show log file sizes and statistics")
    p_stats.add_argument("--verbose", "-v", action="store_true", help="Show all projects")

    # rotate
    p_rotate = sub.add_parser("rotate", help="Rotate oversized log files")
    p_rotate.add_argument(
        "--max-size",
        type=float,
        default=MAX_SIZE_MB,
        help=f"Max file size in MB before rotation (default: {MAX_SIZE_MB})",
    )
    p_rotate.add_argument(
        "--max-rotated",
        type=int,
        default=MAX_ROTATED,
        help=f"Max rotated archives to keep (default: {MAX_ROTATED})",
    )
    p_rotate.add_argument("--dry-run", "-n", action="store_true", help="Show what would happen")

    # cleanup
    p_cleanup = sub.add_parser("cleanup", help="Archive old logs, optionally delete aged archives")
    p_cleanup.add_argument(
        "--days",
        type=int,
        default=ARCHIVE_DAYS,
        help=f"Archive files older than N days (default: {ARCHIVE_DAYS})",
    )
    p_cleanup.add_argument(
        "--delete-days",
        type=int,
        default=None,
        help="Delete compressed archives older than N days (opt-in, no default)",
    )
    p_cleanup.add_argument("--dry-run", "-n", action="store_true", help="Show what would happen")

    args = parser.parse_args(argv)

    if args.command == "stats":
        cmd_stats(args.data_dir, verbose=args.verbose)
    elif args.command == "rotate":
        if args.dry_run:
            print("[DRY RUN] No files will be modified.\n")
        cmd_rotate(
            args.data_dir,
            max_size_mb=args.max_size,
            max_rotated=args.max_rotated,
            dry_run=args.dry_run,
        )
    elif args.command == "cleanup":
        if args.dry_run:
            print("[DRY RUN] No files will be modified.\n")
        cmd_cleanup(
            args.data_dir,
            archive_days=args.days,
            delete_days=args.delete_days,
            dry_run=args.dry_run,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
