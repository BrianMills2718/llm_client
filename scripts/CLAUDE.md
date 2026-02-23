# Scripts Directory

Utility scripts for development and CI. All scripts support `--help` for options.

## Core Scripts

| Script | Purpose |
|--------|---------|
| `scripts/meta/check_plan_tests.py` | Verify/run plan test requirements |
| `scripts/meta/check_plan_blockers.py` | Validate blocked-plan dependencies |
| `scripts/meta/complete_plan.py` | Mark plan complete |
| `scripts/meta/sync_plan_status.py` | Sync plan status |
| `scripts/meta/merge_pr.py` | Merge PRs via GitHub CLI |
| `scripts/meta/parse_plan.py` | Parse plan metadata |
| `scripts/meta/generate_quiz.py` | Generate comprehension quiz prompts |

## Common Commands

```bash
# Plan tests
python scripts/meta/check_plan_tests.py --plan N        # Run tests for plan
python scripts/meta/check_plan_tests.py --plan N --tdd  # See what tests to write

# Plan completion
python scripts/meta/complete_plan.py --plan N           # Mark complete

# Blocked-plan checks
python scripts/meta/check_plan_blockers.py --strict
```

## Worktree Coordination Scripts (opt-in)

If using the worktree coordination module, these additional scripts are available
in `scripts/worktree-coordination/`:

| Script | Purpose |
|--------|---------|
| `check_claims.py` | Manage active work claims |
| `meta_status.py` | Dashboard: claims, PRs, progress |
| `finish_pr.py` | Complete PR lifecycle: merge + cleanup |
| `safe_worktree_remove.py` | Safely remove worktrees |
| `check_messages.py` | Inter-CC messaging inbox |
| `send_message.py` | Send messages to other CC instances |

## Configuration

Edit config files in repo root to customize behavior:
- `meta-process.yaml` - Meta-process settings
- `docs/plans/CLAUDE.md` - plan index
