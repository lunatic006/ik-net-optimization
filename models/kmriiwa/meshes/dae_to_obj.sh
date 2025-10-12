#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Loop through all .dae files in the current directory
for file in *.dae; do
  # Check if the file exists and is a .dae file
  if [[ -f "$file" && "$file" == *.dae ]]; then
    # Get the base filename (without extension)
    base_name="${file%.dae}"
    
    # Convert .dae to .obj using assimp
    assimp export "$file" "${base_name}.obj"
    
    # Fix orientation with Python
    python3 "$SCRIPT_DIR/fix_orientation.py" "${base_name}.obj"
    
    echo "Converted $file to ${base_name}.obj"
  fi
done
