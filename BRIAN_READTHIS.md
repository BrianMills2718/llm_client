# BRIAN READ THIS: Automatic Daily Check (No GitHub Actions)

## What this is
This runs one daily safety check to catch model-routing regressions early.

It runs:
- `./scripts/adoption_gate.sh`

That script checks if long-thinking/background behavior stays above your threshold.

## Why this matters
- If behavior drifts, this catches it fast.
- It runs on your machine/server with cron.
- No GitHub Actions, no GitHub Actions cost.

## One-time setup (copy/paste)

### Step 1: install the cron job
Run this once:

```bash
(crontab -l 2>/dev/null; echo '15 6 * * * cd /home/brian/projects/llm_client && PYTHON_BIN=/home/brian/projects/llm_client/.venv/bin/python ./scripts/adoption_gate.sh >> /home/brian/projects/llm_client/adoption_gate.log 2>&1') | crontab -
```

This means:
- run every day at `06:15`
- use your local repo path
- use your local virtualenv Python
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
