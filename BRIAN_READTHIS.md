# BRIAN READ THIS: Automatic Daily Check (No GitHub Actions)

## What this is
This runs one daily safety check to catch model-routing regressions early.

It runs:
- `./scripts/adoption_gate.sh`

That script checks if long-thinking/background behavior stays above your threshold.
By default it now checks all runs (no run-id prefix filter).

## Why this matters
- If behavior drifts, this catches it fast.
- It runs on your machine/server with cron.
- No GitHub Actions, no GitHub Actions cost.

## New safety lock (worktree enforcement)
- Direct pushes to `main` are now blocked by local git `pre-push`.
- Main-directory edits are blocked in Claude hooks (use a claimed worktree).

Emergency bypasses (use sparingly):
- `ALLOW_MAIN_PUSH=1 git push ...` (bypass main push block)
- `SKIP_CLAIM_VERIFY=1 git push ...` (bypass claim verification)

## One-time setup (copy/paste)

### Step 1: install the cron job
Run this once:

```bash
(crontab -l 2>/dev/null; echo '15 6 * * * cd /home/brian/projects/llm_client && PYTHON_BIN=/home/brian/projects/llm_client/.venv/bin/python LLM_CLIENT_ADOPTION_WARN_ONLY=1 LLM_CLIENT_ADOPTION_SINCE_DAYS=7 ./scripts/adoption_gate.sh >> /home/brian/projects/llm_client/adoption_gate.log 2>&1') | crontab -
```

This means:
- run every day at `06:15`
- use your local repo path
- use your local virtualenv Python
- run in warn-only mode first (no non-zero exit)
- use rolling window: last 7 days of records
- append output to `adoption_gate.log`

### Step 2: verify it is installed

```bash
crontab -l | grep adoption_gate.sh
```

If you see a line, you are done.

## Run it now (manual test)

```bash
cd /home/brian/projects/llm_client
./scripts/adoption_gate.sh --format table
```

If you want report-only mode (never fail exit code), run:

```bash
cd /home/brian/projects/llm_client
LLM_CLIENT_ADOPTION_WARN_ONLY=1 ./scripts/adoption_gate.sh --format table
```

## Read results
- Log file: `/home/brian/projects/llm_client/adoption_gate.log`
- Look for:
  - `Gate: PASS` = good
  - `Gate: FAIL` = investigate
  - `Reason: insufficient_samples` = not enough data yet

## If it fails too often at first
Use less strict settings temporarily:

```bash
cd /home/brian/projects/llm_client
LLM_CLIENT_ADOPTION_MIN_SAMPLES=5 LLM_CLIENT_ADOPTION_MIN_RATE=0.80 ./scripts/adoption_gate.sh --format table
```

Then tighten back later.

## Current concern (documented)
- Right now your logs have records but almost no `reasoning_effort/background_mode` dimensions.
- This causes `insufficient_samples` or `missing_reasoning_effort_dimension`.
- If we run strict mode immediately, you get daily noisy failures that are not actionable.
- That is why cron is currently in warn-only mode.

## One-command fix to start collecting real samples
Run this once (or occasionally):

```bash
cd /home/brian/projects/llm_client
./scripts/adoption_probe.sh
```

What it does:
- makes one real long-thinking call (`gpt-5.2-pro` + `reasoning_effort=high`)
- appends one fresh task-graph experiment row with routing trace + background mode
- gives the adoption gate real data to evaluate

Requirements:
- `OPENROUTER_API_KEY` must be set (default path)
- for direct Gemini models (`gemini/...`), use `GEMINI_API_KEY` instead
- this incurs real API cost (small, but non-zero)

## When to switch to strict mode
Once logs show enough data (at least 20 matching reasoning samples), switch cron to strict:

```bash
(crontab -l 2>/dev/null | sed 's/ LLM_CLIENT_ADOPTION_WARN_ONLY=1//') | crontab -
```

Then verify:

```bash
crontab -l | grep adoption_gate.sh
```

Current cron line should include both:
- `LLM_CLIENT_ADOPTION_WARN_ONLY=1`
- `LLM_CLIENT_ADOPTION_SINCE_DAYS=7`

Optional filter (only if you need it later):
- `LLM_CLIENT_ADOPTION_RUN_ID_PREFIX=nightly_`
