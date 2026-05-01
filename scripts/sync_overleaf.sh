#!/bin/bash
# Sync between draft/ and overleaf/ (Overleaf git repo)
# Usage:
#   ./sync_overleaf.sh push   — draft -> Overleaf
#   ./sync_overleaf.sh pull   — Overleaf -> draft

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DRAFT="$SCRIPT_DIR/draft"
OVERLEAF="$SCRIPT_DIR/overleaf"

if [ ! -d "$OVERLEAF/.git" ]; then
    echo "Error: $OVERLEAF is not a git repo. Clone Overleaf first."
    exit 1
fi

case "${1:-}" in
    push)
        echo "==> Syncing draft -> Overleaf..."
        # Copy tex and bib files
        cp "$DRAFT"/DM.tex "$OVERLEAF"/
        cp "$DRAFT"/*.bib "$OVERLEAF"/ 2>/dev/null || true
        # Copy figures (excluding .gitkeep etc)
        rsync -a --delete "$DRAFT/figures/" "$OVERLEAF/figures/"

        cd "$OVERLEAF"
        git add -A
        if git diff --cached --quiet; then
            echo "No changes to push."
        else
            git status --short
            read -p "Commit message [sync from local]: " msg
            msg="${msg:-sync from local}"
            git commit -m "$msg"
            git push
            echo "==> Pushed to Overleaf."
        fi
        ;;
    pull)
        echo "==> Pulling from Overleaf..."
        cd "$OVERLEAF"
        git pull

        echo "==> Syncing Overleaf -> draft..."
        cp "$OVERLEAF"/DM.tex "$DRAFT"/
        cp "$OVERLEAF"/*.bib "$DRAFT"/ 2>/dev/null || true
        rsync -a --delete "$OVERLEAF/figures/" "$DRAFT/figures/"
        echo "==> Done. Draft updated."
        ;;
    *)
        echo "Usage: $0 {push|pull}"
        echo "  push  — copy draft/ to Overleaf and git push"
        echo "  pull  — git pull Overleaf and copy to draft/"
        exit 1
        ;;
esac
