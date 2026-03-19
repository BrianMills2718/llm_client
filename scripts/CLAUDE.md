# Scripts Directory

Utility scripts for development and CI. All scripts support `--help` for options.

## Core Scripts

| Script | Purpose |
|--------|---------|
| `scripts/meta/check_plan_tests.py` | Verify/run plan test requirements |
| `scripts/meta/check_plan_blockers.py` | Validate blocked-plan dependencies |
| `scripts/meta/complete_plan.py` | Mark plan complete |
| `scripts/meta/sync_plan_status.py` | Sync plan status |
| `scripts/meta/file_context.py` | Resolve required reading and doc/ADR context for repo files |
| `scripts/meta/merge_pr.py` | Merge PRs via GitHub CLI |
| `scripts/meta/parse_plan.py` | Parse plan metadata |
| `scripts/meta/generate_quiz.py` | Generate comprehension quiz prompts |
| `scripts/meta/hook_log.py` | Append structured JSONL entries for Claude hook activity |
| `scripts/meta/validate_plan.py` | Validate plan references against the documentation graph |
| `scripts/meta/check_required_reading.py` | Enforce required docs read before editing coupled source files |
| `scripts/meta/validate_relationships.py` | Validate relationships/read-gate config integrity |
| `scripts/check_markdown_links.py` | Validate local markdown links and anchors in governance docs |

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
python scripts/check_required_reading.py llm_client/client.py

# Relax gate temporarily without code changes
LLM_CLIENT_READ_GATE_MODE=warn python scripts/check_required_reading.py llm_client/client.py

# Read-gate smoke tests (strict/warn/off behavior)
pytest -q tests/test_required_reading_gate.py tests/test_read_gate_hooks.py

# Validate relationships config before CI
python scripts/meta/validate_relationships.py --strict

# Validate a plan against the documentation graph
python scripts/meta/validate_plan.py --plan-file docs/plans/07_governed_repo_contract_alignment.md

# Resolve required-reading context for a target file
python scripts/meta/file_context.py llm_client/agents.py --json

# Check governance-doc links before closing a slice
python scripts/check_markdown_links.py CLAUDE.md docs/plans/CLAUDE.md scripts/CLAUDE.md
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
- `scripts/relationships.yaml` - source/doc couplings and required-reading defaults

Required-reading gate controls:
- `meta_process.quality.required_reading.enabled`
- `meta_process.quality.required_reading.mode` (`strict` | `warn` | `off`)
- `meta_process.quality.required_reading.uncoupled_mode` (`strict` | `warn` | `off`)
- `meta_process.quality.required_reading.show_success`

Current project default:
- Coupled source files: `strict`
- Uncoupled source files: `strict`
- Hook decisions are logged to `.claude/hook_log.jsonl` for operator review.
