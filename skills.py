import os
import yaml
from pathlib import Path

# Skills live at <cwd>/.agents/skills/ or <cwd>/.agent/agents/skills/
BASE_DIR = Path(os.getcwd())
SKILL_DIRS = [
    BASE_DIR / ".agents" / "skills",
    BASE_DIR / ".agent" / "agents" / "skills",
    BASE_DIR / ".agent" / "skills",
]


def find_skills() -> dict:
    """Map each skill name to its description and SKILL.md path."""
    skills = {}
    for directory in SKILL_DIRS:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*/SKILL.md")):
            try:
                _, frontmatter, _ = path.read_text().split("---", 2)
                meta = yaml.safe_load(frontmatter)
                description = " ".join(meta["description"].split())
                skills[meta["name"]] = {"description": description, "path": path}
            except Exception:
                # Skip malformed skill files
                continue
    return skills


def read_skill(name: str) -> str:
    """Return the full contents of a skill's SKILL.md by name."""
    skills = find_skills()
    if name not in skills:
        available = ", ".join(skills.keys()) or "none"
        return f"Error: skill '{name}' not found. Available skills: {available}"
    return skills[name]["path"].read_text()


def skills_prompt() -> str:
    """Build the skills section for the system prompt."""
    skills = find_skills()
    if not skills:
        return "No skills are currently available."
    lines = ["Available skills:"]
    for name, info in skills.items():
        lines.append(f"  - {name}: {info['description']}")
    return "\n".join(lines)


# Eagerly loaded at import time so the system prompt is static per run
SKILLS = find_skills()
