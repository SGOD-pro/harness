# AI Harness Limitations

The **AI Harness** is a powerful tool for building, testing, and maintaining codebases, but it has its boundaries. Below is a concise list of important capabilities it *cannot* provide:

1. **Persistent state across runs** – The harness runs in a stateless sandbox. Anything not explicitly written to the repository remains in memory only.
2. **Direct modification of external resources** – It cannot edit files outside the project directory or interact with external services (e.g., databases, APIs) without explicit tooling.
3. **Manual code review** – Human judgement and contextual understanding beyond the harness’s AGENTS.md are required for thorough code reviews.
4. **Execution of arbitrary shell commands** – Only the `bash` tool is available, and execution is confined to the sandbox; privileged or destructive commands are blocked.
5. **Changing repository history** – The harness does not permit rewriting git history (e.g., `git rebase`, `git filter-branch`). All changes are committed in a linear fashion.
6. **Committing without a repository** – It requires a committed state to act; it cannot create a new repository from scratch or push changes to a remote.
7. **Real‑time error handling** – While it can run tests, it doesn’t capture live logs or continue execution on unexpected runtime errors; such errors terminate the sandbox session.
8. **High‑impact architectural decisions** – Unilaterally deciding on a major tech stack or architectural pattern without human input will trigger the `/architect` workflow.
9. **Personal data processing** – The harness is designed to operate on public code. It has no built‑in GDPR‑compliant data handling or anonymisation capabilities.
10. **Vision, audio, or sensor integration** – It cannot process non‑textual data such as images, sound, or real‑world sensors.

Refer to the **/architect** skill for any decision that requires a human review of design trade‑offs.
