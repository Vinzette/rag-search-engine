#!/usr/bin/env bash
# Downloads the data/ directory from the latest GitHub release.
# Usage: bash download_data.sh

set -e

REPO="Vinzette/rag-search-engine"
ASSET="data.zip"
DEST="data"

if [ -d "$DEST" ] && [ "$(ls -A "$DEST")" ]; then
    echo "data/ directory already exists and is not empty. Skipping download."
    exit 0
fi

echo "Downloading $ASSET from github.com/$REPO ..."

# Try gh CLI first, fall back to curl
if command -v gh &>/dev/null; then
    gh release download --repo "$REPO" --pattern "$ASSET" --clobber
else
    # Fetch the latest release asset URL via the GitHub API
    URL=$(curl -s "https://api.github.com/repos/$REPO/releases/latest" \
        | grep -o "https://github.com/$REPO/releases/download/[^\"]*/$ASSET")
    if [ -z "$URL" ]; then
        echo "Error: Could not find $ASSET in the latest release."
        echo "Please download it manually from https://github.com/$REPO/releases"
        exit 1
    fi
    curl -L -o "$ASSET" "$URL"
fi

echo "Extracting $ASSET ..."
unzip -o "$ASSET"
rm "$ASSET"

echo "Done! data/ directory is ready."
