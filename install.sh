#!/usr/bin/env bash
# Central Brain Installation & Setup Script

set -e

echo "🧠 Installing Central Brain..."

BRAIN_DIR="$HOME/.central_brain"
BIN_DIR="$HOME/.local/bin"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create directories
mkdir -p "$BRAIN_DIR/knowledge" "$BRAIN_DIR/projects" "$BRAIN_DIR/episodes" "$BRAIN_DIR/db" "$BIN_DIR"

# Copy brain engine
cp "$SCRIPT_DIR/brain.py" "$BRAIN_DIR/brain.py"
chmod +x "$BRAIN_DIR/brain.py"

# Symlink to PATH
ln -sf "$BRAIN_DIR/brain.py" "$BIN_DIR/brain"

# Initialize default sources.json if missing
if [ ! -f "$BRAIN_DIR/sources.json" ]; then
    cat << 'EOF' > "$BRAIN_DIR/sources.json"
[
  "/home/meoclavezz/.central_brain/knowledge",
  "/home/meoclavezz/.central_brain/projects",
  "/home/meoclavezz/.central_brain/episodes",
  "/home/meoclavezz/Documents/Configs",
  "/home/meoclavezz/.agents/project_map.md",
  "/home/meoclavezz/AGENTS.md"
]
EOF
fi

# Setup working memory helper if missing
if [ ! -f "$HOME/.working_memory" ]; then
    cp "$SCRIPT_DIR/helpers/working_memory_template.sh" "$HOME/.working_memory"
fi

echo "✅ Central Brain installed successfully!"
echo "Run 'brain status' or 'brain query <text>' to get started."
