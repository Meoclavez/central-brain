# Central Brain Agent Directives

You are connected to a local **Central Brain (`brain`)** CLI tool on this system.
The `brain` tool is NOT an LLM function/tool call. It is an executable CLI binary available in your system path. Execute it directly using shell/bash commands (e.g. `run_command` or bash terminal).

1. **SEARCH BEFORE ANALYZING (BASH)**:
   - Before troubleshooting, writing code, or answering technical questions, execute the bash command:
     `brain query "<topic/error/feature>"`
   - Use the retrieved output to inform your technical response.

2. **RECORD SOLUTIONS & LESSONS (BASH)**:
   - Immediately after solving a problem, fixing a bug, or updating a config, execute the bash command:
     `brain remember "<fact/solution>" --entity "<Topic>" --category "<Fix|Rule|Knowledge|Project>"`

3. **INGEST, SYNC & PRUNE KNOWLEDGE (BASH)**:
   - When creating or updating project documentation, execute:
     `brain ingest <path>`
     `brain sync`
   - After removing, moving, or deleting files, execute:
     `brain prune`
     (Purges orphan files deleted from disk, deduplicates facts, and vacuums SQLite storage).

4. **EMPIRICAL ANSWERS**:
   - Base technical decisions on actual outputs returned from `brain query` and terminal commands.
