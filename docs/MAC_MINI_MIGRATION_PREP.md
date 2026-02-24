# Mac Mini Migration Prep (Desktop-Only)

This document is for **prep on desktop** before you move to Mac mini.

Goal:
- Keep migration personal-only (`BrianMills2718`), no work account artifacts.
- Move with a deterministic checklist and command sequence.

## Phase 1: Desktop Prep (no Mac changes yet)

1. Confirm your local repo is current:

```bash
cd /home/brian/projects/llm_client
git pull --ff-only
```

2. Build the personal-only transfer bundle:

```bash
cd /home/brian/projects/llm_client
chmod +x scripts/macos/create_personal_transfer_bundle.sh
./scripts/macos/create_personal_transfer_bundle.sh --out-dir "$HOME/Desktop"
```

3. Record the bundle path and checksum printed by the script.

4. Transfer the bundle to Mac mini (AirDrop, USB, rsync, or scp).

## Phase 2: Mac Mini Setup

1. Extract bundle on Mac mini:

```bash
mkdir -p ~/migrate/llm_client_bundle
tar -xzf ~/Desktop/llm_client_personal_migration_bundle_*.tar.gz -C ~/migrate/llm_client_bundle
```

2. Move payload into `~/projects/llm_client`:

```bash
mkdir -p ~/projects
rm -rf ~/projects/llm_client
mv ~/migrate/llm_client_bundle/payload ~/projects/llm_client
```

3. Run personal-only setup:

```bash
cd ~/projects/llm_client
chmod +x scripts/macos/setup_personal_openclaw_mac.sh
./scripts/macos/setup_personal_openclaw_mac.sh \
  --openclaw-cmd 'openclaw gateway run' \
  --purge-work
```

## Phase 3: Post-Migration Verification (copy/paste sequence)

1. One-command verifier:

```bash
cd ~/projects/llm_client
chmod +x scripts/macos/verify_personal_only.sh
./scripts/macos/verify_personal_only.sh
```

2. Spot-check account and identity:

```bash
cd ~/projects
gh auth status
git config --global user.name
git config --global user.email
```

3. Confirm OpenClaw background service:

```bash
launchctl list | grep com.brian.openclaw.personal
tail -n 50 ~/Library/Logs/openclaw-personal.out.log
tail -n 50 ~/Library/Logs/openclaw-personal.err.log
```

4. Confirm gateway is responsive:

```bash
openclaw gateway status
```

## Expected Results

- `gh` active account in `~/projects` is `BrianMills2718`.
- Global git email is `brianmills2718@gmail.com`.
- `~/.config/gh-work` and `~/.gitconfig-work` are absent.
- Pre-push hook blocks `aisteno/*` and `brian-steno/*` remotes.
- `com.brian.openclaw.personal` is loaded and running.

## If anything fails

1. Re-run setup:

```bash
cd ~/projects/llm_client
./scripts/macos/setup_personal_openclaw_mac.sh \
  --openclaw-cmd 'openclaw gateway run' \
  --purge-work
```

2. Re-run verifier and inspect failing line:

```bash
./scripts/macos/verify_personal_only.sh
```
