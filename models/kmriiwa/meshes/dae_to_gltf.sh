#!/bin/bash

# Loop through all .dae files in the current directory
for file in *.dae; do
  # Check if the file exists and is a .dae file
  if [[ -f "$file" && "$file" == *.dae ]]; then
    # Get the base filename (without extension)
    base_name="${file%.dae}"
    
    # Convert .dae to .gltf using assimp
    assimp export "$file" "${base_name}.gltf"
    
    echo "Converted $file to ${base_name}.gltf"
  fi
done
