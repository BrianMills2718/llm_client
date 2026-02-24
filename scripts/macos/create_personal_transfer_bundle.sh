#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-$HOME/Desktop}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BUNDLE_NAME="llm_client_personal_migration_bundle_${TIMESTAMP}.tar.gz"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="${TMPDIR:-/tmp}/llm_client_personal_bundle_${TIMESTAMP}"

usage() {
  cat <<'EOF'
Usage:
  ./scripts/macos/create_personal_transfer_bundle.sh [--out-dir PATH] [--name FILE.tar.gz]

Description:
  Builds a personal-only migration bundle for Mac mini setup.
  The bundle includes a tracked-source snapshot of this repo (no .git history,
  no untracked local files), plus migration instructions.

Options:
  --out-dir PATH      Output directory (default: ~/Desktop)
  --name FILE         Bundle filename (default: llm_client_personal_migration_bundle_<timestamp>.tar.gz)
  -h, --help          Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    --name)
      BUNDLE_NAME="${2:-}"
      shift 2
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

if [[ "${BUNDLE_NAME}" != *.tar.gz ]]; then
  echo "ERROR: --name must end in .tar.gz"
  exit 1
fi

mkdir -p "$OUT_DIR"
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR/payload"

echo "Creating tracked-source snapshot from HEAD..."
git -C "$REPO_ROOT" archive --format=tar HEAD | tar -xf - -C "$WORK_DIR/payload"

cat > "$WORK_DIR/README_TRANSFER.txt" <<'EOF'
Personal-only migration bundle for Mac mini

Contents:
- payload/               Tracked source snapshot of llm_client at bundle time
  - scripts/macos/setup_personal_openclaw_mac.sh
  - scripts/macos/verify_personal_only.sh
  - docs/MAC_MINI_MIGRATION_PREP.md
  - BRIAN_READTHIS.md

Notes:
- This bundle excludes git history and untracked local files.
- It is intended to avoid accidental transfer of local credentials/work artifacts.

On Mac mini:
1) Extract bundle.
2) cd payload
3) Follow docs/MAC_MINI_MIGRATION_PREP.md
EOF

(
  cd "$WORK_DIR"
  {
    echo "Bundle created: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Source repo: $REPO_ROOT"
    echo "Source commit: $(git -C "$REPO_ROOT" rev-parse HEAD)"
    echo ""
    echo "File SHA256:"
    find payload -type f | sort | while read -r f; do
      shasum -a 256 "$f"
    done
  } > MANIFEST.txt
)

OUT_PATH="$OUT_DIR/$BUNDLE_NAME"
tar -czf "$OUT_PATH" -C "$WORK_DIR" README_TRANSFER.txt MANIFEST.txt payload

echo "Bundle written:"
echo "  $OUT_PATH"
echo ""
echo "Quick verify:"
echo "  tar -tzf \"$OUT_PATH\" | sed -n '1,40p'"
echo "  shasum -a 256 \"$OUT_PATH\""

rm -rf "$WORK_DIR"
