#!/usr/bin/env bash
# Idempotent GitHub config for F1 StratLab: branch protection + labels.
# Run once (and after changing required CI contexts or labels):
#   bash scripts/setup-github.sh
# Requires: gh auth login (with admin on the repo).
set -euo pipefail

REPO="${GITHUB_REPOSITORY:-VforVitorio/F1-StratLab}"

# Exact CI job names from .github/workflows/ci.yml — these are the required checks.
REQUIRED_CONTEXTS=(test lint typecheck)

# Stable branches that get protected. `test` is the active dev line and stays
# unprotected so daily work can land directly; promotion flows test -> dev -> main.
PROTECTED_BRANCHES=(main dev)

PROTECTION_PAYLOAD() {
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

echo "==> Branch protection"
for br in "${PROTECTED_BRANCHES[@]}"; do
  echo "  - $br (strict checks: ${REQUIRED_CONTEXTS[*]})"
  gh api -X PUT "repos/$REPO/branches/$br/protection" --input - <<< "$(PROTECTION_PAYLOAD)" >/dev/null
done

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
