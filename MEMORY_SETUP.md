# Memory Persistence System — Setup Notes

**Created:** 2026-08-05
**Purpose:** Keep session context and user preferences across shell sessions.

---

## Files Created

| File | Role |
|------|------|
| `~/.working_memory` | Persistent notes file + two shell functions (`log_session_entry`, `flush_working_memory`) with EXIT trap |
| `/etc/profile.d/00-osc-memory.sh` | Auto-sources `~/.working_memory` on every new login/non-login shell |

## How It Works

1. **On shell start** — `.bashrc` calls `flush_working_memory()` which reads the latest notes from `~/.working_memory`.
2. **During a session** — use `log_session_entry()` to append context (user prefs, config facts, decisions).
3. **On shell exit** — EXIT trap auto-flushes any new entries before closing.

## What's Stored

- User preferences & system config notes
- Session log timestamps for traceability
- Central Brain verification status
