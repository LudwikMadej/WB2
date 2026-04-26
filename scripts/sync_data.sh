#!/usr/bin/env bash
set -euo pipefail

REMOTE="gdrive:WB2/data"
LOCAL="$(cd "$(dirname "$0")/.." && pwd)/data"

if ! command -v rclone &>/dev/null; then
    echo "rclone not found. Install:"
    echo "  brew install rclone    # macOS"
    echo "  curl https://rclone.org/install.sh | sudo bash  # Linux"
    exit 1
fi

if ! rclone listremotes | grep -q '^gdrive:'; then
    echo "No 'gdrive' remote configured. Run:"
    echo "  rclone config"
    echo "  → n (new) → name: gdrive → Google Drive → follow OAuth flow"
    exit 1
fi

BISYNC_ARGS=(
    "$LOCAL" "$REMOTE"
    --verbose
    --conflict-resolve newer
    --conflict-loser num
    --max-delete 25
)

case "${1:-sync}" in
    init)
        echo "Initializing bisync baseline: $LOCAL <-> $REMOTE"
        rclone bisync "${BISYNC_ARGS[@]}" --resync
        ;;
    sync)
        echo "Syncing: $LOCAL <-> $REMOTE"
        rclone bisync "${BISYNC_ARGS[@]}"
        ;;
    download)
        echo "Downloading: $REMOTE -> $LOCAL"
        rclone copy "$REMOTE" "$LOCAL" --verbose
        ;;
    upload)
        echo "Uploading: $LOCAL -> $REMOTE"
        rclone copy "$LOCAL" "$REMOTE" --verbose
        ;;
    *)
        echo "Usage: $0 {init|sync|download|upload}"
        echo ""
        echo "  init      First-time bisync (establishes baseline)"
        echo "  sync      Bidirectional sync (default)"
        echo "  download  One-way: Drive -> local"
        echo "  upload    One-way: local -> Drive"
        exit 1
        ;;
esac
