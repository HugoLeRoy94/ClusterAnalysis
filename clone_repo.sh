#!/bin/bash

# Name of the current repo directory
SRC_REPO="$(basename "$PWD")"
DEST_REPO="${SRC_REPO}_clean"

# GitHub remote — change this!
GITHUB_URL="git@github.com:HugoLeRoy94/ClusterAnalysis.git"

# 1. Create new directory
cd ..
cp -r "$SRC_REPO" "$DEST_REPO"

# 2. Remove Git history
cd "$DEST_REPO"
rm -rf .git

# 3. Reinitialize Git
git init
git add .
git commit -m "Initial commit (clean export, no history)"
git branch -M main

# 4. Set GitHub remote and push
git remote add origin "$GITHUB_URL"
git push -u origin main

echo "✓ Clean full copy pushed to: $GITHUB_URL"
