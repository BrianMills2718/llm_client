#!/usr/bin/env bash
set -euo pipefail

EXPECTED_ACCOUNT="${EXPECTED_ACCOUNT:-BrianMills2718}"
EXPECTED_EMAIL="${EXPECTED_EMAIL:-brianmills2718@gmail.com}"
EXPECTED_HOOK_LABEL="${EXPECTED_HOOK_LABEL:-com.brian.openclaw.personal}"
EXPECTED_GH_CONFIG_DIR="${EXPECTED_GH_CONFIG_DIR:-$HOME/.config/gh-personal}"

GREEN=""
RED=""
YELLOW=""
RESET=""
if [[ -t 1 ]]; then
  GREEN="$(printf '\033[32m')"
  RED="$(printf '\033[31m')"
  YELLOW="$(printf '\033[33m')"
  RESET="$(printf '\033[0m')"
fi

PASS_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0

pass() {
  PASS_COUNT=$((PASS_COUNT + 1))
  printf "%sPASS%s: %s\n" "$GREEN" "$RESET" "$1"
}

warn() {
  WARN_COUNT=$((WARN_COUNT + 1))
  printf "%sWARN%s: %s\n" "$YELLOW" "$RESET" "$1"
}

fail() {
  FAIL_COUNT=$((FAIL_COUNT + 1))
  printf "%sFAIL%s: %s\n" "$RED" "$RESET" "$1"
}

check_equals() {
  local actual="$1"
  local expected="$2"
  local msg="$3"
  if [[ "$actual" == "$expected" ]]; then
    pass "$msg ($actual)"
  else
    fail "$msg (expected: $expected, got: $actual)"
  fi
}

echo "Verifying personal-only GitHub/OpenClaw setup..."
echo "Expected account: $EXPECTED_ACCOUNT"
echo "Expected email:   $EXPECTED_EMAIL"
echo ""

if [[ "$(uname -s)" != "Darwin" ]]; then
  warn "Not running on macOS (Darwin). launchd checks skipped."
fi

if command -v gh >/dev/null 2>&1; then
  GH_PATH="$(command -v gh)"
  pass "gh found at $GH_PATH"
else
  fail "gh CLI not found in PATH"
fi

if [[ -x "$HOME/.local/bin/gh" ]]; then
  if rg -n "gh-personal|GH_CONFIG_DIR" "$HOME/.local/bin/gh" >/dev/null 2>&1; then
    pass "gh wrapper exists and references personal config"
  else
    fail "gh wrapper exists but does not reference personal config"
  fi
else
  fail "gh wrapper missing at ~/.local/bin/gh"
fi

EMAIL="$(git config --global user.email || true)"
check_equals "$EMAIL" "$EXPECTED_EMAIL" "Global git email"

NAME="$(git config --global user.name || true)"
if [[ -n "$NAME" ]]; then
  pass "Global git user.name set ($NAME)"
else
  fail "Global git user.name is not set"
fi

HOOKS_PATH="$(git config --global core.hooksPath || true)"
if [[ -n "$HOOKS_PATH" && -f "$HOOKS_PATH/pre-push" ]]; then
  if rg -n "aisteno|brian-steno" "$HOOKS_PATH/pre-push" >/dev/null 2>&1; then
    pass "pre-push hook blocks work remotes ($HOOKS_PATH/pre-push)"
  else
    fail "pre-push hook found but does not block work remotes"
  fi
else
  fail "Global pre-push hook not installed via core.hooksPath"
fi

GH_HELPERS="$(git config --global --get-all credential.https://github.com.helper || true)"
if [[ "$GH_HELPERS" == *"gh-auth-git-credential-personal"* ]]; then
  pass "GitHub credential helper uses personal helper"
else
  fail "GitHub credential helper is not personal helper"
fi
if [[ "$GH_HELPERS" == *"gh-auth-git-credential-work"* ]]; then
  fail "GitHub credential helper still references work helper"
fi

if [[ -d "$HOME/.config/gh-work" ]]; then
  fail "~/.config/gh-work exists (work credentials present)"
else
  pass "~/.config/gh-work absent"
fi

if [[ -f "$HOME/.gitconfig-work" ]]; then
  fail "~/.gitconfig-work exists (work git include present)"
else
  pass "~/.gitconfig-work absent"
fi

if [[ -d "$HOME/projects" ]]; then
  LOGIN="$(cd "$HOME/projects" && gh api user --jq .login 2>/dev/null || true)"
  if [[ -n "$LOGIN" ]]; then
    check_equals "$LOGIN" "$EXPECTED_ACCOUNT" "Effective gh account in ~/projects"
  else
    fail "Unable to query gh account in ~/projects (try: cd ~/projects && gh auth status)"
  fi
else
  warn "~/projects does not exist; skipped gh account context check"
fi

PLIST="$HOME/Library/LaunchAgents/${EXPECTED_HOOK_LABEL}.plist"
if [[ -f "$PLIST" ]]; then
  pass "LaunchAgent plist exists ($PLIST)"
else
  fail "LaunchAgent plist missing ($PLIST)"
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
  if launchctl list | rg -n "$EXPECTED_HOOK_LABEL" >/dev/null 2>&1; then
    pass "launchd service loaded ($EXPECTED_HOOK_LABEL)"
  else
    fail "launchd service not loaded ($EXPECTED_HOOK_LABEL)"
  fi
fi

echo ""
echo "Summary: PASS=$PASS_COUNT WARN=$WARN_COUNT FAIL=$FAIL_COUNT"
if [[ "$FAIL_COUNT" -gt 0 ]]; then
  exit 1
fi
