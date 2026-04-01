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
| `scripts/meta/check_required_reading.py` | Enforce required docs read before editing coupled source files |
| `scripts/meta/validate_relationships.py` | Validate relationships/read-gate config integrity |
| `scripts/meta/worktree-coordination/check_claims.py` | Claim, inspect, and release sanctioned worktree ownership |
| `scripts/meta/worktree-coordination/create_worktree.py` | Create sanctioned worktrees under the repo's default worktree root |
| `scripts/meta/worktree-coordination/safe_worktree_remove.py` | Remove sanctioned worktrees safely |

## Common Commands

```bash
# Plan tests
python scripts/meta/check_plan_tests.py --plan N        # Run tests for plan
python scripts/meta/check_plan_tests.py --plan N --tdd  # See what tests to write

# Plan completion
python scripts/meta/complete_plan.py --plan N           # Mark complete

# Blocked-plan checks
python scripts/meta/check_plan_blockers.py --strict

# Required-reading gate check (used by .claude/hooks/gate-edit.sh)
python scripts/meta/check_required_reading.py llm_client/client.py

# Relax gate temporarily without code changes
LLM_CLIENT_READ_GATE_MODE=warn python scripts/meta/check_required_reading.py llm_client/client.py

# Read-gate smoke tests (strict/warn/off behavior)
pytest -q tests/test_required_reading_gate.py

# Validate relationships config before CI
python scripts/meta/validate_relationships.py --strict

# Sanctioned worktree coordination
make worktree BRANCH=plan-22-example TASK="Describe the task" PLAN=22
make worktree-list
make worktree-remove BRANCH=plan-22-example
```

## Configuration

Edit config files in repo root to customize behavior:
- `docs/plans/CLAUDE.md` - plan index
- `docs/ops/CAPABILITY_DECOMPOSITION.md` - repo-local capability ownership source of record
- `scripts/relationships.yaml` - source/doc couplings and required-reading defaults
