#!/bin/bash

# Absolute path to the script directory (for fix_orientation.py)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Optional: target directory (defaults to current directory)
TARGET_DIR="${1:-.}"

# Use find to traverse recursively and find .dae and .stl files
find "$TARGET_DIR" -type f \( -iname '*.dae' -o -iname '*.stl' \) | while read -r file; do
  # Get extension and base name
  ext="${file##*.}"
  base_name="${file%.*}"
  obj_file="${base_name}.obj"

  echo "Converting $file to $obj_file"
  assimp export "$file" "$obj_file"

  # If this was a .dae file, fix orientation
  if [[ "$ext" == "dae" || "$ext" == "DAE" ]]; then
    echo "  Fixing orientation for $obj_file"
    python3 "$SCRIPT_DIR/fix_orientation.py" "$obj_file"
  fi
done