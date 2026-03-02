#!/usr/bin/env bash
# Synchronise a git submodule with the latest commit from its remote.
# Usage: ./git_sync_submodule.sh <submodule_path>
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

info()  { echo "[INFO]  $*"; }
ok()    { echo "[OK]    $*"; }
warn()  { echo "[WARN]  $*"; }

# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <submodule_path>" >&2
    exit 1
fi

SUBMODULE_PATH="$1"
if [[ ! -d "${SUBMODULE_PATH}" ]]; then
    echo "Error: Submodule path '${SUBMODULE_PATH}' does not exist." >&2
    exit 1
fi

REMOTE_URL=$(git -C "${SUBMODULE_PATH}" config --get remote.origin.url)
info "Submodule : ${SUBMODULE_PATH}"
info "Remote    : ${REMOTE_URL}"

# ---------------------------------------------------------------------------
# Snapshot local state before any changes
# ---------------------------------------------------------------------------

LOCAL_BEFORE=$(git -C "${SUBMODULE_PATH}" rev-parse HEAD)
info "Local HEAD (before) : ${LOCAL_BEFORE}"

# ---------------------------------------------------------------------------
# Fetch latest commits from the submodule's remote
# ---------------------------------------------------------------------------

info "Fetching from remote..."
git -C "${SUBMODULE_PATH}" fetch origin

REMOTE_AFTER=$(git -C "${SUBMODULE_PATH}" rev-parse origin/HEAD 2>/dev/null \
               || git -C "${SUBMODULE_PATH}" rev-parse FETCH_HEAD)
info "Remote HEAD (after fetch) : ${REMOTE_AFTER}"

# ---------------------------------------------------------------------------
# Detect whether the remote has new commits the local doesn't have
# ---------------------------------------------------------------------------

if [[ "${LOCAL_BEFORE}" == "${REMOTE_AFTER}" ]]; then
    ok "No new commits on remote — submodule is already up to date."
    exit 0
fi

NEW_COMMITS=$(git -C "${SUBMODULE_PATH}" log \
    --oneline "${LOCAL_BEFORE}..${REMOTE_AFTER}" 2>/dev/null || true)

if [[ -n "${NEW_COMMITS}" ]]; then
    info "New commits available on remote:"
    while IFS= read -r line; do
        info "  ${line}"
    done <<< "${NEW_COMMITS}"
else
    warn "Commit hashes differ but no log entries found between them."
fi

# ---------------------------------------------------------------------------
# Update parent repo's submodule pointer to the latest remote commit
# (must run from parent repo root, not inside the submodule)
# ---------------------------------------------------------------------------

info "Updating parent repo submodule pointer via 'git submodule update --remote'..."
git submodule update --remote "${SUBMODULE_PATH}"

LOCAL_AFTER=$(git -C "${SUBMODULE_PATH}" rev-parse HEAD)
info "Local HEAD (after)  : ${LOCAL_AFTER}"

# ---------------------------------------------------------------------------
# Confirm the pull succeeded
# ---------------------------------------------------------------------------

if [[ "${LOCAL_AFTER}" == "${REMOTE_AFTER}" ]]; then
    ok "Submodule '${SUBMODULE_PATH}' is now at the latest remote commit."
else
    warn "Local HEAD (${LOCAL_AFTER}) does not match remote HEAD (${REMOTE_AFTER}) — update may be incomplete."
fi

# ---------------------------------------------------------------------------
# Remind the caller to commit the updated pointer in the parent repo
# ---------------------------------------------------------------------------

if git -C "${REPO_ROOT}" diff --quiet "${SUBMODULE_PATH}" 2>/dev/null; then
    ok "Parent repo submodule pointer is unchanged (already recorded this commit)."
else
    info "Parent repo has an unstaged change to the submodule pointer."
    info "Commit it with:"
    info "  git add ${SUBMODULE_PATH} && git commit -m 'Update ${SUBMODULE_PATH} submodule to ${LOCAL_AFTER}'"
fi
