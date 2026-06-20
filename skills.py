"""Skills: curated best-practice playbooks the model can load on demand.

A skill is a markdown file in skills/ with simple frontmatter:

    ---
    name: kubectl
    description: one line shown in the catalog
    ---
    # ...full playbook body...

Progressive disclosure (same idea as Claude Code skills): only the catalog
(name + description) goes into the system prompt, cheaply. The model calls the
`load_skill` tool to read a skill's full body when it judges one relevant, so the
body costs tokens only when actually used. This scales to many skills.

To add a skill, drop a new .md file in skills/. No code change needed.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

SKILLS_DIR = os.getenv("SKILLS_DIR", os.path.join(os.path.dirname(__file__), "skills"))

_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def _parse(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    name, description, body = path.stem, "", text
    m = _FRONTMATTER.match(text)
    if m:
        front, body = m.group(1), m.group(2)
        for line in front.splitlines():
            key, _, value = line.partition(":")
            key, value = key.strip().lower(), value.strip()
            if key == "name" and value:
                name = value
            elif key == "description":
                description = value
    return {"name": name, "description": description, "body": body.strip()}


def list_skills(skills_dir: str = SKILLS_DIR) -> list[dict]:
    directory = Path(skills_dir)
    if not directory.is_dir():
        return []
    return [_parse(p) for p in sorted(directory.glob("*.md"))]


def skills_catalog(skills_dir: str = SKILLS_DIR) -> str:
    """The short catalog appended to the system prompt: names + descriptions.
    Empty string if there are no skills."""
    skills = list_skills(skills_dir)
    if not skills:
        return ""
    lines = "\n".join(f"- {s['name']}: {s['description']}" for s in skills)
    return (
        "\n\nYou have these best-practice playbooks (skills). When one is relevant "
        "to the problem, call load_skill(name) to read it before diagnosing:\n" + lines
    )


def get_skill(name: str, skills_dir: str = SKILLS_DIR) -> str:
    """Full body of one skill, or an error listing the available names."""
    skills = list_skills(skills_dir)
    for s in skills:
        if s["name"] == name:
            return s["body"]
    available = ", ".join(s["name"] for s in skills) or "(none)"
    return f"ERROR: no skill named '{name}'. Available: {available}"
