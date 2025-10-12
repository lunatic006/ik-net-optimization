#!/bin/bash
for stl_file in *.stl; do
  # Get the base name without extension
  base_name=$(basename "$stl_file" .stl)
  # Convert STL to OBJ
  meshlabserver -i "$stl_file" -o "${base_name}.obj"
done