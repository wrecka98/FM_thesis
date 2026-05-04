#!/usr/bin/env bash

# Threshold (change if needed)
THRESHOLD="+100M"

echo "🔍 Scanning for large files ($THRESHOLD)..."
echo ""

# Temp files
TMP_FILES=$(mktemp)
TMP_DIRS=$(mktemp)
TMP_EXT=$(mktemp)

# Find large files
find . -type f -size $THRESHOLD > "$TMP_FILES"

if [ ! -s "$TMP_FILES" ]; then
  echo "✅ No large files found."
  exit 0
fi

echo "📦 Large files found:"
cat "$TMP_FILES"
echo ""

# Extract directories
echo "📁 Extracting directories..."
awk -F/ '{print $2}' "$TMP_FILES" | sort | uniq > "$TMP_DIRS"

# Extract extensions
echo "📄 Extracting file types..."
awk -F. 'NF>1 {print "*."$NF}' "$TMP_FILES" | sort | uniq > "$TMP_EXT"

# Append to .gitignore
echo "✍️ Updating .gitignore..."

{
  echo ""
  echo "# === Auto-generated: Large files ($(date)) ==="
  echo ""

  echo "# Folders:"
  while read dir; do
    [ -n "$dir" ] && echo "$dir/"
  done < "$TMP_DIRS"

  echo ""
  echo "# File types:"
  while read ext; do
    echo "$ext"
  done < "$TMP_EXT"

} >> .gitignore

echo ""
echo "✅ .gitignore updated!"
echo ""

# Cleanup
rm "$TMP_FILES" "$TMP_DIRS" "$TMP_EXT"