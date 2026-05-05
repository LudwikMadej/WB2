#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL="$REPO/data"
BUCKET="wb2"
ENDPOINT="https://f681dbaf79de38cf431e125730894e42.r2.cloudflarestorage.com"
ACCESS_KEY="8e7789ce4343edb621c9da7a0a8740fe"
SECRET_KEY="244bb6fbc0216445e0d8cef64b89d7c520d58e79d137ccf10ac9ae119c363002"

if ! command -v rclone &>/dev/null; then
    echo "rclone not found. Install:"
    echo "  brew install rclone    # macOS"
    echo "  curl https://rclone.org/install.sh | sudo bash  # Linux"
    exit 1
fi

RCLONE_FLAGS=(
    --s3-provider Cloudflare
    --s3-access-key-id "$ACCESS_KEY"
    --s3-secret-access-key "$SECRET_KEY"
    --s3-endpoint "$ENDPOINT"
    --s3-no-check-bucket
    --verbose
    --progress
    --transfers 32
    --checkers 64
)

case "${1:-download}" in
    download)
        echo "Downloading: R2 -> $LOCAL"
        rclone copy ":s3:${BUCKET}" "$LOCAL" "${RCLONE_FLAGS[@]}"
        ;;
    upload)
        echo "Uploading: $LOCAL -> R2"
        rclone copy "$LOCAL" ":s3:${BUCKET}" "${RCLONE_FLAGS[@]}"
        ;;
    sync)
        echo "Syncing (Mirroring): $LOCAL -> R2"
        rclone sync "$LOCAL" ":s3:${BUCKET}" "${RCLONE_FLAGS[@]}"
        ;;
    *)
        echo "Usage: $0 {download|upload|sync}"
        echo ""
        echo "  download  R2 -> local (default)"
        echo "  upload    local -> R2"
        echo "  sync    local -> R2 (overwrite)"
        exit 1
        ;;
esac
