#!/usr/bin/env bash
# create_learning_structure.sh — lightweight CLI installer for the learning structure
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
FOLDERS=("1_Journey" "2_Real" "3_Environment" "4_UI" "5_Formula" "6_Symbols" "7_Semblance" "8_Test")
ASSETS_DIR="$ROOT_DIR/assets"

echo "Creating learning structure at: $ROOT_DIR"
for f in "${FOLDERS[@]}"; do
  mkdir -p "$ROOT_DIR/$f"
  # Ensure a README exists if not already present
  if [ ! -f "$ROOT_DIR/$f/README.md" ]; then
    echo "# $f" > "$ROOT_DIR/$f/README.md"
  fi
done

# Create an assets folder for image rendering
mkdir -p "$ASSETS_DIR"
for i in {1..8}; do
  mkdir -p "$ASSETS_DIR/$(printf "%02d" $i)_folder"
done

# Make script executable (best-effort; user may need to chmod after cloning)
chmod +x "$ROOT_DIR/create_learning_structure.sh" || true

echo "Structure created. Add images under $ROOT_DIR/assets/ and customize the READMEs."