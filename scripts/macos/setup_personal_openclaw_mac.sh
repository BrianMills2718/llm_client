#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: this script is for macOS only."
  exit 1
fi

PERSONAL_ACCOUNT="BrianMills2718"
PERSONAL_NAME="Brian"
PERSONAL_EMAIL="brianmills2718@gmail.com"
LABEL="com.brian.openclaw.personal"
REPO_ROOT="$HOME/projects"
OPENCLAW_CMD='openclaw gateway run'
NO_LOAD=0
PURGE_WORK=0

usage() {
  cat <<'EOF'
Usage:
  ./scripts/macos/setup_personal_openclaw_mac.sh [options]

Options:
  --openclaw-cmd "<cmd>"      Command to run continuously via launchd
  --repo-root "<path>"        Working directory for OpenClaw (default: ~/projects)
  --label "<launchd.label>"   launchd service label (default: com.brian.openclaw.personal)
  --personal-account "<acct>" GitHub account to force in personal context (default: BrianMills2718)
  --personal-name "<name>"    git user.name (default: Brian)
  --personal-email "<email>"  git user.email (default: brianmills2718@gmail.com)
  --purge-work                Remove ~/.config/gh-work and ~/.gitconfig-work
  --no-load                   Generate files but do not load launchd service
  -h, --help                  Show this help

Examples:
  ./scripts/macos/setup_personal_openclaw_mac.sh \
    --openclaw-cmd 'openclaw gateway run'

  ./scripts/macos/setup_personal_openclaw_mac.sh \
    --openclaw-cmd 'openclaw daemon --config "$HOME/projects/openclaw.yml"' \
    --purge-work
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --openclaw-cmd)
      OPENCLAW_CMD="${2:-}"
      shift 2
      ;;
    --repo-root)
      REPO_ROOT="${2:-}"
      shift 2
      ;;
    --label)
      LABEL="${2:-}"
      shift 2
      ;;
    --personal-account)
      PERSONAL_ACCOUNT="${2:-}"
      shift 2
      ;;
    --personal-name)
      PERSONAL_NAME="${2:-}"
      shift 2
      ;;
    --personal-email)
      PERSONAL_EMAIL="${2:-}"
      shift 2
      ;;
    --purge-work)
      PURGE_WORK=1
      shift
      ;;
    --no-load)
      NO_LOAD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

GH_PERSONAL_DIR="$HOME/.config/gh-personal"
GIT_HOOKS_DIR="$HOME/.config/git/hooks"
LOCAL_BIN="$HOME/.local/bin"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
OPENCLAW_CFG_DIR="$HOME/.config/openclaw-personal"
RUNNER="$LOCAL_BIN/openclaw-personal-runner.sh"
CMD_FILE="$OPENCLAW_CFG_DIR/command.sh"
ROOT_FILE="$OPENCLAW_CFG_DIR/repo_root.txt"
PLIST_PATH="$LAUNCH_AGENTS_DIR/${LABEL}.plist"
GH_WRAPPER="$LOCAL_BIN/gh"
GH_CRED_HELPER="$LOCAL_BIN/gh-auth-git-credential-personal"
PRE_PUSH_HOOK="$GIT_HOOKS_DIR/pre-push"
LOG_OUT="$HOME/Library/Logs/openclaw-personal.out.log"
LOG_ERR="$HOME/Library/Logs/openclaw-personal.err.log"
REPO_ROOT_EXPANDED="${REPO_ROOT/#\~/$HOME}"

if [[ -z "${OPENCLAW_CMD// }" ]]; then
  echo "ERROR: --openclaw-cmd cannot be empty."
  exit 1
fi

mkdir -p "$GH_PERSONAL_DIR" "$GIT_HOOKS_DIR" "$LOCAL_BIN" "$LAUNCH_AGENTS_DIR" "$OPENCLAW_CFG_DIR" "$HOME/Library/Logs"

cat > "$GH_CRED_HELPER" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
unset GITHUB_TOKEN GH_TOKEN GITHUB_ENTERPRISE_TOKEN
export GH_CONFIG_DIR="$HOME/.config/gh-personal"
exec /usr/bin/gh auth git-credential "$@"
EOF
chmod 700 "$GH_CRED_HELPER"

cat > "$GH_WRAPPER" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
unset GITHUB_TOKEN GH_TOKEN GITHUB_ENTERPRISE_TOKEN
case "${PWD:-}" in
  "$HOME/projects"|"$HOME/projects"/*)
    export GH_CONFIG_DIR="$HOME/.config/gh-personal"
    ;;
  *)
    export GH_CONFIG_DIR="${GH_CONFIG_DIR:-$HOME/.config/gh-personal}"
    ;;
esac
exec /usr/bin/gh "$@"
EOF
chmod 700 "$GH_WRAPPER"

printf '%s\n' "$OPENCLAW_CMD" > "$CMD_FILE"
chmod 600 "$CMD_FILE"
printf '%s\n' "$REPO_ROOT_EXPANDED" > "$ROOT_FILE"
chmod 600 "$ROOT_FILE"

cat > "$RUNNER" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
unset GITHUB_TOKEN GH_TOKEN GITHUB_ENTERPRISE_TOKEN
export GH_CONFIG_DIR="$HOME/.config/gh-personal"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
CMD_FILE="$HOME/.config/openclaw-personal/command.sh"
ROOT_FILE="$HOME/.config/openclaw-personal/repo_root.txt"
if [[ ! -f "$CMD_FILE" ]]; then
  echo "ERROR: missing command file: $CMD_FILE" >&2
  exit 1
fi
if [[ ! -f "$ROOT_FILE" ]]; then
  echo "ERROR: missing repo-root file: $ROOT_FILE" >&2
  exit 1
fi
OPENCLAW_CMD="$(cat "$CMD_FILE")"
REPO_ROOT="$(cat "$ROOT_FILE")"
mkdir -p "$REPO_ROOT"
cd "$REPO_ROOT"
exec /bin/bash -lc "$OPENCLAW_CMD"
EOF
chmod 700 "$RUNNER"

cat > "$PRE_PUSH_HOOK" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
remote_name="${1:-origin}"
remote_url="${2:-}"
if [[ -z "$remote_url" ]]; then
  remote_url="$(git remote get-url "$remote_name" 2>/dev/null || true)"
fi
lower_url="$(printf '%s' "$remote_url" | tr '[:upper:]' '[:lower:]')"
if [[ "$lower_url" == *":aisteno/"* || "$lower_url" == *"/aisteno/"* || "$lower_url" == *":brian-steno/"* || "$lower_url" == *"/brian-steno/"* ]]; then
  echo "ERROR: push blocked on personal-only machine."
  echo "Remote appears to be work-scoped: $remote_url"
  echo "Allowed: personal repos only."
  exit 1
fi
exit 0
EOF
chmod 700 "$PRE_PUSH_HOOK"

git config --global user.name "$PERSONAL_NAME"
git config --global user.email "$PERSONAL_EMAIL"
git config --global core.hooksPath "$GIT_HOOKS_DIR"

git config --global --unset-all credential.https://github.com.helper || true
git config --global --add credential.https://github.com.helper ""
git config --global --add credential.https://github.com.helper "!$GH_CRED_HELPER"

git config --global --unset-all credential.https://gist.github.com.helper || true
git config --global --add credential.https://gist.github.com.helper ""
git config --global --add credential.https://gist.github.com.helper "!$GH_CRED_HELPER"

git config --global url."git@github-personal:".insteadOf "git@github.com:"

while IFS= read -r key; do
  value="$(git config --global --get "$key" || true)"
  if [[ "$value" == *"/steno/"* || "$value" == *".gitconfig-work"* ]]; then
    git config --global --unset-all "$key" || true
  fi
done < <(git config --global --name-only --get-regexp '^includeIf\.gitdir:.*\.path$' || true)

if [[ "$PURGE_WORK" == "1" ]]; then
  rm -rf "$HOME/.config/gh-work" "$HOME/.gitconfig-work"
fi

if ! GH_CONFIG_DIR="$GH_PERSONAL_DIR" /usr/bin/gh auth status >/dev/null 2>&1; then
  echo "No GitHub auth found in $GH_PERSONAL_DIR. Starting login flow..."
  GH_CONFIG_DIR="$GH_PERSONAL_DIR" /usr/bin/gh auth login --hostname github.com --git-protocol https --web
fi
GH_CONFIG_DIR="$GH_PERSONAL_DIR" /usr/bin/gh auth switch -u "$PERSONAL_ACCOUNT" >/dev/null 2>&1 || true

cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>${LABEL}</string>

    <key>ProgramArguments</key>
    <array>
      <string>${RUNNER}</string>
    </array>

    <key>WorkingDirectory</key>
    <string>${REPO_ROOT_EXPANDED}</string>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>ProcessType</key>
    <string>Background</string>

    <key>ThrottleInterval</key>
    <integer>10</integer>

    <key>StandardOutPath</key>
    <string>${LOG_OUT}</string>

    <key>StandardErrorPath</key>
    <string>${LOG_ERR}</string>
  </dict>
</plist>
EOF

if [[ "$NO_LOAD" == "0" ]]; then
  uid="$(id -u)"
  launchctl bootout "gui/$uid/$LABEL" "$PLIST_PATH" >/dev/null 2>&1 || true
  launchctl bootstrap "gui/$uid" "$PLIST_PATH"
  launchctl enable "gui/$uid/$LABEL" >/dev/null 2>&1 || true
  launchctl kickstart -k "gui/$uid/$LABEL" >/dev/null 2>&1 || true
fi

echo "Setup complete."
echo "Personal account target: $PERSONAL_ACCOUNT"
echo "Git identity: $PERSONAL_NAME <$PERSONAL_EMAIL>"
echo "GH wrapper: $GH_WRAPPER"
echo "Credential helper: $GH_CRED_HELPER"
echo "Pre-push work-block hook: $PRE_PUSH_HOOK"
echo "OpenClaw runner: $RUNNER"
echo "launchd plist: $PLIST_PATH"
echo "Logs:"
echo "  $LOG_OUT"
echo "  $LOG_ERR"
echo "Verify:"
echo "  cd ~/projects && gh auth status"
echo "  launchctl list | grep ${LABEL}"
echo "  tail -n 50 $LOG_OUT"
