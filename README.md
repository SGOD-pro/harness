# LLM Harness

This repository implements a **LLM harness** built **from scratch**.  The purpose of the harness is to allow fast and lightweight testing of language‑model behaviour without relying on external frameworks or heavy dependencies.

## Goals

- **Zero‑dependency** core – only the standard library and the minimal runtime packages that the repository already pulls in.
- **Modular design** – split a harness into reusable components: input generators, evaluation functions, and report writers.
- **Ease of extension** – add new tasks or modify scoring logic without touching the core.

## Getting Started

```bash
# run the default test harness
python -m main
```

The hub script will locate LLM configurations in *tools.py* and *llm.py*, create test prompts based on the definitions in *src/harness*, and aggregate the results.

## Project Structure

```
├── src/harness          # harness modules (input, scorer, reporter)
├── llm.py               # abstraction over LLM clients
├── tools.py             # utilities for prompt generation
├── main.py              # entry‑point for running the harness
├── pyproject.toml
└── README.md
```

## Why From Scratch?

A dedicated harness allows us to:

1. Embed it in lightweight Docker containers.
2. Control how prompts are extended or decorated.
3. Push introspection metrics directly to a custom UI.
4. Avoid version drift caused by third‑party testing libraries.

Feel free to explore or extend the harness – the core logic lives in `src/harness` and is intentionally simple.

---

Happy hacking!
