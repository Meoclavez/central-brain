# CENTRAL BRAIN AGENT DIRECTIVE (ARCH LINUX / ASUS TUF)

You are an AI coding and system assistant connected to a local Central Brain (`brain`) CLI binary.
The `brain` tool is NOT an LLM tool/function call. Execute it directly via bash/shell terminal commands:

1. **SPEC-DRIVEN EXECUTION LOOP (BASH)**:
   - **Discuss**: Clarify requirements and constraints before writing code.
   - **Plan**: Inspect project state with `brain state` and decompose work into testable steps.
   - **Execute**: Use isolated subagents for heavy codebase exploration to prevent context rot.
   - **Verify**: Empirically verify code through tests, compilation, and shell commands.
   - **Ship & Remember**: Commit changes and persist verified facts to the brain.

2. **BEFORE ANSWERING OR CODING (BASH)**:
   - Always query the Central Brain to retrieve verified system facts, kernel configurations, and past fixes:
     `brain query "<keywords_or_problem_description>"`
   - To inspect project status, milestones, and blockers:
     `brain state [project_path]`
   - Read `.agents/project_map.md` to locate active scripts and system rules.

3. **AFTER RESOLVING ISSUES OR DISCOVERING FIXES (BASH)**:
   - Persist findings immediately so they survive context resets and kernel changes:
     `brain remember "<verified_fix_or_rule>" --entity "<Topic>" --category "<Fix|Rule|Knowledge|Project>"`

4. **CORRECTING SUPERSEDED / WRONG MEMORIES (BASH)**:
   - When a past solution is replaced or no longer works:
     `brain correct "<Topic>" "<new_verified_solution>" --category "Fix"`
   - To erase an invalid memory:
     `brain forget "<search_term>"`

5. **SYNC, BACKUP & PRECISION FILTERING (BASH)**:
   - For structured JSON output, pass `--json` to any command (e.g. `brain query "<query>" --json`).
   - To filter search by category or entity:
     `brain query "<query>" -c "<Fix|Rule|Knowledge>" -e "<Entity>"`
   - To scaffold a spec-driven planning structure for a new project:
     `brain init-project <name> [path] -d "description"`
   - To export a compiled memory digest:
     `brain export [output_file.md]`
   - To create a full point-in-time snapshot:
     `brain backup [path.tar.gz]`
   - To sync or clean:
     `brain sync` / `brain prune`

6. **EMPIRICAL ANSWERS**:
   - Base technical decisions on actual outputs returned from `brain query`, `brain state`, and terminal commands.
