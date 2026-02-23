# === META-PROCESS TARGETS ===
# Added by meta-process install.sh

# Configuration
SCRIPTS_META := scripts/meta
PLANS_DIR := docs/plans
READS_FILE ?= /tmp/.claude_session_reads

# --- Session Start ---
.PHONY: status

status:  ## Show git status
	@git status --short --branch

# --- During Implementation ---
.PHONY: test test-quick check read-gate-check read-gate-check-warn

test:  ## Run pytest
	pytest tests/ -v

test-quick:  ## Run pytest (no traceback)
	pytest tests/ -q --tb=no

check:  ## Run all checks (test, mypy)
	@echo "Running tests..."
	@pytest tests/ -q --tb=short
	@echo ""
	@echo "Running mypy..."
	@mypy llm_client --ignore-missing-imports
	@echo ""
	@echo "All checks passed!"

read-gate-check:  ## Check required-reading gate (FILE=llm_client/client.py)
ifndef FILE
	$(error FILE is required. Usage: make read-gate-check FILE=llm_client/client.py)
endif
	@python $(SCRIPTS_META)/check_required_reading.py "$(FILE)" --reads-file "$(READS_FILE)"

read-gate-check-warn:  ## Check gate in warn mode (FILE=llm_client/client.py)
ifndef FILE
	$(error FILE is required. Usage: make read-gate-check-warn FILE=llm_client/client.py)
endif
	@LLM_CLIENT_READ_GATE_MODE=warn \
		python $(SCRIPTS_META)/check_required_reading.py "$(FILE)" --reads-file "$(READS_FILE)"

# --- PR Workflow ---
.PHONY: pr-ready pr merge finish

pr-ready:  ## Rebase on main and push
	@git fetch origin main
	@git rebase origin/main
	@git push -u origin HEAD

pr:  ## Create PR (opens browser)
	@gh pr create --fill --web

merge:  ## Merge PR (PR=number required)
ifndef PR
	$(error PR is required. Usage: make merge PR=123)
endif
	@python $(SCRIPTS_META)/merge_pr.py $(PR)

finish:  ## Merge PR + cleanup branch (BRANCH=name PR=number required)
ifndef BRANCH
	$(error BRANCH is required. Usage: make finish BRANCH=plan-42-feature PR=123)
endif
ifndef PR
	$(error PR is required. Usage: make finish BRANCH=plan-42-feature PR=123)
endif
	@gh pr merge $(PR) --squash --delete-branch
	@git checkout main && git pull --ff-only
	@git branch -d $(BRANCH) 2>/dev/null || true

# --- Plans ---
.PHONY: plan-tests plan-complete

plan-tests:  ## Check plan's required tests (PLAN=N required)
ifndef PLAN
	$(error PLAN is required. Usage: make plan-tests PLAN=42)
endif
	@python $(SCRIPTS_META)/check_plan_tests.py --plan $(PLAN)

plan-complete:  ## Mark plan complete with verification (PLAN=N required)
ifndef PLAN
	$(error PLAN is required. Usage: make plan-complete PLAN=42)
endif
	@python $(SCRIPTS_META)/complete_plan.py --plan $(PLAN)

# --- Help ---
.PHONY: help-meta

help-meta:  ## Show meta-process targets
	@echo "Meta-Process Targets:"
	@echo ""
	@echo "  Session:"
	@echo "    status               Show git status"
	@echo ""
	@echo "  Development:"
	@echo "    test                 Run tests"
	@echo "    check                Run tests + mypy"
	@echo "    read-gate-check      Check required reading (FILE=...)"
	@echo "    read-gate-check-warn Check gate in warn mode (FILE=...)"
	@echo ""
	@echo "  PR Workflow:"
	@echo "    pr-ready             Rebase + push"
	@echo "    pr                   Create PR"
	@echo "    merge                Merge PR (PR=number)"
	@echo "    finish               Merge + cleanup (BRANCH=name PR=number)"
	@echo ""
	@echo "  Plans:"
	@echo "    plan-tests           Check plan tests (PLAN=N)"
	@echo "    plan-complete        Complete plan (PLAN=N)"
