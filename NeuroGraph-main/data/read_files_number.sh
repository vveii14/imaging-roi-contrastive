#!/bin/bash

traverse() {
    local dir="$1"
    local indent="$2"
    local root_dir="$3"
    # Determine display name
    local display_name
    if [[ "$dir" == "$root_dir" ]]; then
        display_name="$dir"
    else
        display_name=$(basename "$dir")
    fi
    # Count all files under $dir using efficient method
    local count=$(find "$dir" -type f -printf '.' | wc -c)
    # Print the line
    echo "${indent}${display_name} [${count} files]"
    # Process subdirectories, sorted alphabetically
    find "$dir" -mindepth 1 -maxdepth 1 -type d | sort | while read -r subdir; do
        traverse "$subdir" "${indent}    " "$root_dir"
    done
}

start_dir="${1:-.}"
start_dir=$(realpath "$start_dir")
echo "File counts in tree format for: $start_dir"
traverse "$start_dir" "" "$start_dir"