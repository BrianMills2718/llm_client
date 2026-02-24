# === META-PROCESS TARGETS ===
# Added by meta-process install.sh

# Configuration
SCRIPTS_META := scripts/meta
PLANS_DIR := docs/plans
READS_FILE ?= /tmp/.claude_session_reads
GITHUB_ACCOUNT ?= BrianMills2718
PR_AUTO_EXPECTED_REPO ?= llm_client
CLAIMS_SCRIPT ?= scripts/meta/worktree-coordination/check_claims.py
META_STATUS_SCRIPT ?= scripts/meta/worktree-coordination/meta_status.py
SAFE_WORKTREE_REMOVE_SCRIPT ?= scripts/meta/worktree-coordination/safe_worktree_remove.py

# --- Session Start ---
.PHONY: status worktree worktree-remove claim release claims meta-status

status:  ## Show git status
	@git status --short --branch

worktree:  ## Create worktree from origin/main (BRANCH=... TASK="..." optional)
ifndef BRANCH
	$(error BRANCH is required. Usage: make worktree BRANCH=plan-NN-description TASK="...optional...")
endif
	@git fetch origin main
	@mkdir -p worktrees
	@if git show-ref --verify --quiet "refs/heads/$(BRANCH)"; then \
		echo "ERROR: branch '$(BRANCH)' already exists locally."; \
		exit 1; \
	fi
	@git worktree add "worktrees/$(BRANCH)" -b "$(BRANCH)" origin/main
	@echo "Worktree created: worktrees/$(BRANCH)"
	@echo "Next: cd worktrees/$(BRANCH)"
	@if [ -n "$(TASK)" ] && [ -f "$(CLAIMS_SCRIPT)" ]; then \
		python "$(CLAIMS_SCRIPT)" --claim --id "$(BRANCH)" --task "$(TASK)"; \
	fi
	@if [ -z "$(TASK)" ]; then \
		echo "Optional claim: python $(CLAIMS_SCRIPT) --claim --id \"$(BRANCH)\" --task \"...\""; \
	fi

worktree-remove:  ## Safely remove worktree (BRANCH=...)
ifndef BRANCH
	$(error BRANCH is required. Usage: make worktree-remove BRANCH=plan-NN-description)
endif
	@if [ -f "$(SAFE_WORKTREE_REMOVE_SCRIPT)" ]; then \
		python "$(SAFE_WORKTREE_REMOVE_SCRIPT)" "worktrees/$(BRANCH)"; \
	else \
		git worktree remove "worktrees/$(BRANCH)"; \
	fi

claim:  ## Create/update active claim for current branch (TASK="...")
ifndef TASK
	$(error TASK is required. Usage: make claim TASK="description" [ID=branch-name])
endif
	@if [ ! -f "$(CLAIMS_SCRIPT)" ]; then \
		echo "ERROR: claims script not found at $(CLAIMS_SCRIPT)"; \
		exit 1; \
	fi
	@CLAIM_ID="$${ID:-$$(git rev-parse --abbrev-ref HEAD)}"; \
	python "$(CLAIMS_SCRIPT)" --claim --id "$$CLAIM_ID" --task "$(TASK)"

release:  ## Release active claim for current branch
	@if [ ! -f "$(CLAIMS_SCRIPT)" ]; then \
		echo "ERROR: claims script not found at $(CLAIMS_SCRIPT)"; \
		exit 1; \
	fi
	@CLAIM_ID="$${ID:-$$(git rev-parse --abbrev-ref HEAD)}"; \
	python "$(CLAIMS_SCRIPT)" --release --id "$$CLAIM_ID"

claims:  ## List active claims/worktrees
	@if [ ! -f "$(CLAIMS_SCRIPT)" ]; then \
		echo "ERROR: claims script not found at $(CLAIMS_SCRIPT)"; \
		exit 1; \
	fi
	@python "$(CLAIMS_SCRIPT)" --list

meta-status:  ## Coordination dashboard (claims/PRs/worktrees)
	@if [ -f "$(META_STATUS_SCRIPT)" ]; then \
		python "$(META_STATUS_SCRIPT)"; \
	elif [ -f "$(CLAIMS_SCRIPT)" ]; then \
		python "$(CLAIMS_SCRIPT)" --list; \
	else \
		echo "ERROR: no coordination status script found."; \
		exit 1; \
	fi

# --- During Implementation ---
.PHONY: test test-quick check adoption-gate read-gate-check read-gate-check-warn

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

adoption-gate:  ## Run local long-thinking adoption gate (cron/CI friendly)
	./scripts/adoption_gate.sh

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
.PHONY: pr-ready pr merge finish pr-auto-check pr-auto

pr-ready:  ## Rebase on main and push
	@git fetch origin main
	@git rebase origin/main
	@git push -u origin HEAD

pr:  ## Create PR (opens browser)
	@gh pr create --fill --web

pr-auto-check:  ## Autonomous PR preflight (branch/clean tree/origin/account)
	@python $(SCRIPTS_META)/pr_auto.py --preflight-only --expected-origin-repo $(PR_AUTO_EXPECTED_REPO) --account $(GITHUB_ACCOUNT)

pr-auto:  ## Autonomous PR create + auto-merge request (non-interactive)
	@python $(SCRIPTS_META)/pr_auto.py --expected-origin-repo $(PR_AUTO_EXPECTED_REPO) --account $(GITHUB_ACCOUNT) --fill --auto-merge

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
	@echo "    worktree             Create worktree from main (BRANCH=... TASK=...)"
	@echo "    worktree-remove      Safely remove worktree (BRANCH=...)"
	@echo "    claim                Claim current branch (TASK=...)"
	@echo "    release              Release current claim"
	@echo "    claims               List active claims/worktrees"
	@echo "    meta-status          Coordination dashboard"
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
	@echo "    pr-auto-check        Preflight autonomous PR flow"
	@echo "    pr-auto              Non-interactive PR + auto-merge request"
	@echo "    merge                Merge PR (PR=number)"
	@echo "    finish               Merge + cleanup (BRANCH=name PR=number)"
	@echo ""
	@echo "  Plans:"
	@echo "    plan-tests           Check plan tests (PLAN=N)"
	@echo "    plan-complete        Complete plan (PLAN=N)"
