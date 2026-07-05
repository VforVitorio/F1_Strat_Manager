#!/usr/bin/env bash
# Idempotent GitHub config for F1 StratLab: branch protection + labels.
# Run once (and after changing required CI contexts or labels):
#   bash scripts/setup-github.sh
# Requires: gh auth login (with admin on the repo).
set -euo pipefail

REPO="${GITHUB_REPOSITORY:-VforVitorio/F1-StratLab}"

# Exact CI job names from .github/workflows/ci.yml — these are the required checks.
REQUIRED_CONTEXTS=(test lint typecheck)

# Release-only branch: strict CI + no force-push + no deletion.
STRICT_BRANCHES=(main)

# Long-lived working branches (three-branch flow: test -> dev -> main). They must
# NEVER be deleted on a PR merge, but must still accept DIRECT work: `test` is the
# daily dev line, `dev` receives plain merges from `test` and is the PR head into
# `main`. So: delete-proof only (allow_deletions:false), NO required checks (they
# would block the test->dev direct merge), force-push allowed (for the periodic
# fast-forward resets of test). Quality is gated at `main`.
DELETE_PROOF_BRANCHES=(dev test)

STRICT_PAYLOAD() {
  cat <<EOF
{
  "required_status_checks": {"strict": true, "contexts": [$(printf '"%s",' "${REQUIRED_CONTEXTS[@]}" | sed 's/,$//')]},
  "enforce_admins": false,
  "required_pull_request_reviews": null,
  "required_conversation_resolution": true,
  "restrictions": null,
  "lock_branch": false,
  "allow_force_pushes": false,
  "allow_deletions": false
}
EOF
}

# Minimal protection: the ONLY guarantee is "cannot be deleted". Everything else
# stays open so direct commits, merges and FF resets keep working.
DELETE_PROOF_PAYLOAD() {
  cat <<EOF
{
  "required_status_checks": null,
  "enforce_admins": false,
  "required_pull_request_reviews": null,
  "required_conversation_resolution": false,
  "restrictions": null,
  "lock_branch": false,
  "allow_force_pushes": true,
  "allow_deletions": false
}
EOF
}

echo "==> Branch protection (strict, release-only)"
for br in "${STRICT_BRANCHES[@]}"; do
  echo "  - $br (strict checks: ${REQUIRED_CONTEXTS[*]})"
  gh api -X PUT "repos/$REPO/branches/$br/protection" --input - <<< "$(STRICT_PAYLOAD)" >/dev/null
done

echo "==> Branch protection (delete-proof, direct work still allowed)"
for br in "${DELETE_PROOF_BRANCHES[@]}"; do
  echo "  - $br (never deleted on merge)"
  gh api -X PUT "repos/$REPO/branches/$br/protection" --input - <<< "$(DELETE_PROOF_PAYLOAD)" >/dev/null
done

# Never auto-delete a head branch on merge (this is what deleted `dev` once). Feature
# branches are cleaned explicitly with `gh pr merge --delete-branch`.
echo "==> Disable auto-delete-on-merge"
gh api -X PATCH "repos/$REPO" -F delete_branch_on_merge=false >/dev/null

echo "==> Labels"
# name|color|description — mirrors .github/labeler.yml + dependabot.yml.
LABELS=(
  "bug|d73a4a|Something isn't working"
  "enhancement|a2eeef|New feature or request"
  "epic|7057ff|Tracks a large block of related work"
  "dependencies|0366d6|Dependency updates"
  "do-not-rebase|fbca04|Skip auto-update-prs rebasing"
  "area: codebase|0e8a16|src/, scripts/, notebooks/"
  "area: deps|1d76db|pyproject.toml, uv.lock"
  "area: ci-cd|5319e7|.github/workflows, dependabot, labeler"
  "area: docs|c5def5|docs/ and root markdown"
  "area: tests|bfd4f2|tests/"
)
for entry in "${LABELS[@]}"; do
  IFS='|' read -r name color desc <<< "$entry"
  gh label create "$name" --color "$color" --description "$desc" --force >/dev/null
  echo "  - $name"
done

echo "Done. Re-run after changing CI job names or labels."
