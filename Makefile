# llm_client Makefile — consumer interface for observability and development
#
# Usage: make help

SHELL := /bin/bash
.DEFAULT_GOAL := help
PYTHON := python
DAYS ?= 7
PROJECT ?=
LIMIT ?= 20

# ─── Observability ───────────────────────────────────────────────────────────

.PHONY: cost cost-by-project cost-by-model cost-by-task errors recent traces summary

cost:  ## Total spend (DAYS=7 default, PROJECT= optional)
	@$(PYTHON) -m llm_client cost --group-by project --days $(DAYS) \
		$(if $(PROJECT),--project $(PROJECT))

cost-by-project:  ## Spend per project (DAYS=7)
	@$(PYTHON) -m llm_client cost --group-by project --days $(DAYS)

cost-by-model:  ## Spend per model (DAYS=7, PROJECT= optional)
	@$(PYTHON) -m llm_client cost --group-by model --days $(DAYS) \
		$(if $(PROJECT),--project $(PROJECT))

cost-by-task:  ## Spend per task (DAYS=7, PROJECT= optional)
	@$(PYTHON) -m llm_client cost --group-by task --days $(DAYS) \
		$(if $(PROJECT),--project $(PROJECT))

errors:  ## Error breakdown by model (DAYS=7, PROJECT= optional)
	@$(PYTHON) -c "\
	import sqlite3, os; \
	from pathlib import Path; \
	db = sqlite3.connect(os.environ.get('LLM_CLIENT_DB_PATH', str(Path.home() / 'projects/data/llm_observability.db')), timeout=10); \
	from datetime import datetime, timedelta, timezone; \
	cutoff = (datetime.now(timezone.utc) - timedelta(days=$(DAYS))).isoformat(); \
	pf = \"AND project = '$(PROJECT)'\" if '$(PROJECT)' else ''; \
	rows = db.execute(f\"\"\"SELECT model, COUNT(*) as total, \
		SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as errors, \
		ROUND(100.0 * SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*), 1) as error_pct \
		FROM llm_calls WHERE task != 'test' AND timestamp >= ? {pf} \
		GROUP BY model ORDER BY errors DESC LIMIT 20\"\"\", (cutoff,)).fetchall(); \
	print(f\"{'Model':<55} {'Total':>7} {'Errors':>7} {'Rate':>7}\"); \
	print('-' * 80); \
	[print(f'{r[0]:<55} {r[1]:>7} {r[2]:>7} {r[3]:>6}%') for r in rows]; \
	total = sum(r[1] for r in rows); errs = sum(r[2] for r in rows); \
	print('-' * 80); \
	print(f\"{'TOTAL':<55} {total:>7} {errs:>7} {(100*errs/total if total else 0):>6.1f}%\")"

recent:  ## Last N calls (LIMIT=20, PROJECT= optional)
	@$(PYTHON) -c "\
	import sqlite3, os; \
	from pathlib import Path; \
	db = sqlite3.connect(os.environ.get('LLM_CLIENT_DB_PATH', str(Path.home() / 'projects/data/llm_observability.db')), timeout=10); \
	pf = \"WHERE project = '$(PROJECT)'\" if '$(PROJECT)' else ''; \
	rows = db.execute(f\"\"\"SELECT timestamp, model, task, \
		ROUND(COALESCE(marginal_cost, cost), 4) as cost, \
		total_tokens, ROUND(latency_s, 1) as latency, \
		CASE WHEN error IS NOT NULL THEN 'ERR' ELSE 'ok' END as status \
		FROM llm_calls WHERE task != 'test' {pf} ORDER BY timestamp DESC LIMIT $(LIMIT)\"\"\").fetchall(); \
	print(f\"{'Time':<20} {'Model':<40} {'Task':<25} {'Cost':>8} {'Tokens':>8} {'Lat':>6} {'St':>4}\"); \
	print('-' * 115); \
	[print(f'{r[0][11:19]:<20} {(r[1] or \"?\")[:39]:<40} {(r[2] or \"-\")[:24]:<25} \$${ r[3] or 0:>7.4f} {r[4] or 0:>8} {r[5] or 0:>5.1f}s {r[6]:>4}') for r in rows]"

traces:  ## Recent traces with cost rollup (DAYS=3, PROJECT= optional)
	@$(PYTHON) -m llm_client traces --days $(DAYS) \
		$(if $(PROJECT),--project $(PROJECT))

summary:  ## Quick dashboard: spend, calls, errors, top models (DAYS=7)
	@echo "=== llm_client Summary (last $(DAYS) days) ==="
	@echo ""
	@$(PYTHON) -c "\
	import sqlite3, os; \
	from pathlib import Path; \
	db = sqlite3.connect(os.environ.get('LLM_CLIENT_DB_PATH', str(Path.home() / 'projects/data/llm_observability.db')), timeout=10); \
	from datetime import datetime, timedelta, timezone; \
	cutoff = (datetime.now(timezone.utc) - timedelta(days=$(DAYS))).isoformat(); \
	r = db.execute(\"\"\"SELECT COUNT(*), ROUND(COALESCE(SUM(COALESCE(marginal_cost, cost)), 0), 2), \
		SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END), \
		COUNT(DISTINCT project), COUNT(DISTINCT model), \
		COALESCE(SUM(total_tokens), 0) \
		FROM llm_calls WHERE task != 'test' AND timestamp >= ?\"\"\", (cutoff,)).fetchone(); \
	print(f'  Calls:    {r[0]:,}'); \
	print(f'  Spend:    \$${r[1]:,.2f}'); \
	print(f'  Errors:   {r[2]:,} ({100*r[2]/r[0] if r[0] else 0:.1f}%)'); \
	print(f'  Projects: {r[3]}'); \
	print(f'  Models:   {r[4]}'); \
	print(f'  Tokens:   {r[5]:,}'); \
	print(); \
	print('Top projects by spend:'); \
	rows = db.execute(\"\"\"SELECT COALESCE(project,'unknown'), \
		ROUND(COALESCE(SUM(COALESCE(marginal_cost, cost)), 0), 2), COUNT(*) \
		FROM llm_calls WHERE task != 'test' AND timestamp >= ? GROUP BY project ORDER BY 2 DESC LIMIT 5\"\"\", (cutoff,)).fetchall(); \
	[print(f'  \$${r[1]:>8.2f}  {r[2]:>6} calls  {r[0]}') for r in rows]; \
	print(); \
	print('Top models by spend:'); \
	rows = db.execute(\"\"\"SELECT model, \
		ROUND(COALESCE(SUM(COALESCE(marginal_cost, cost)), 0), 2), COUNT(*) \
		FROM llm_calls WHERE task != 'test' AND timestamp >= ? GROUP BY model ORDER BY 2 DESC LIMIT 5\"\"\", (cutoff,)).fetchall(); \
	[print(f'  \$${r[1]:>8.2f}  {r[2]:>6} calls  {r[0]}') for r in rows]"

# ─── Development ─────────────────────────────────────────────────────────────

.PHONY: test test-quick check install

test:  ## Run full test suite
	@pytest tests/ -q --tb=short

test-quick:  ## Run tests (minimal output)
	@pytest tests/ -q --tb=no

check:  ## Run tests + mypy
	@echo "Running tests..."
	@pytest tests/ -q --tb=short
	@echo ""
	@echo "Running mypy..."
	@mypy llm_client --ignore-missing-imports
	@echo ""
	@echo "All checks passed!"

install:  ## Install llm_client in editable mode
	@pip install -e .

# ─── Maintenance ─────────────────────────────────────────────────────────────

.PHONY: api-docs models log-stats log-rotate log-cleanup

api-docs:  ## Regenerate API reference docs
	@$(PYTHON) scripts/meta/generate_api_reference.py --write

models:  ## Show model registry
	@$(PYTHON) -m llm_client models

log-stats:  ## Show JSONL log sizes per project
	@$(PYTHON) scripts/log_maintenance.py stats

log-rotate:  ## Rotate oversized JSONL logs (MAX_SIZE_MB=100, DRY_RUN= for preview)
	@$(PYTHON) scripts/log_maintenance.py rotate \
		--max-size $(or $(MAX_SIZE_MB),100) \
		$(if $(DRY_RUN),--dry-run)

log-cleanup:  ## Archive old JSONL logs (ARCHIVE_DAYS=90, DELETE_DAYS= optional, DRY_RUN= for preview)
	@$(PYTHON) scripts/log_maintenance.py cleanup \
		--days $(or $(ARCHIVE_DAYS),90) \
		$(if $(DELETE_DAYS),--delete-days $(DELETE_DAYS)) \
		$(if $(DRY_RUN),--dry-run)

# ─── Help ────────────────────────────────────────────────────────────────────

.PHONY: help

help:  ## Show this help
	@echo "llm_client — runtime substrate for multi-provider LLM calls"
	@echo ""
	@echo "Observability:"
	@grep -E '^[a-z].*:.*## ' $(MAKEFILE_LIST) | grep -E '^(cost|errors|recent|traces|summary)' | \
		awk -F ':.*## ' '{printf "  make %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "Development:"
	@grep -E '^[a-z].*:.*## ' $(MAKEFILE_LIST) | grep -E '^(test|check|install)' | \
		awk -F ':.*## ' '{printf "  make %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "Maintenance:"
	@grep -E '^[a-z].*:.*## ' $(MAKEFILE_LIST) | grep -E '^(api-docs|models|log-)' | \
		awk -F ':.*## ' '{printf "  make %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "Options: DAYS=7 PROJECT= LIMIT=20 MAX_SIZE_MB=100 ARCHIVE_DAYS=90 DRY_RUN= DELETE_DAYS="
