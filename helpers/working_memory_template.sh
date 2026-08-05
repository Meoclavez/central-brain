# My Persistent Working Memory (session notes — persists across shell sessions)
# Created for local multi-agent context tracking

## User Profile
- Prefers concise, practical responses. No fluff.
- Uses direct tool calls and Central Brain CLI at ~/.central_brain/brain.py.

## Session Log
# Log entry helper functions
log_session_entry() {
    local note="$1"
    local timestamp=$(date +"%Y-%m-%d %H:%M")
    echo "  - $timestamp — $note" >> "$HOME/.working_memory"
    if command -v brain >/dev/null 2>&1; then
        brain remember "$note" --entity "Session" --category "WorkingMemory"
    fi
}

flush_working_memory() {
    if command -v brain >/dev/null 2>&1; then
        brain sync >/dev/null 2>&1 || true
    fi
}

trap flush_working_memory EXIT
