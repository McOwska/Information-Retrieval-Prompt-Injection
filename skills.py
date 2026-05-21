from __future__ import annotations

import os
import pathlib
import re
from typing import Any, Dict, List, Optional, Tuple

WORKSPACE = pathlib.Path.cwd().resolve()


def get_skills_dir() -> pathlib.Path:
    env = os.getenv("SKILLS_DIR")
    if env:
        p = pathlib.Path(env)
        return p if p.is_absolute() else (WORKSPACE / p).resolve()
    return (WORKSPACE / ".skills").resolve()


_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _parse_frontmatter(raw: str) -> Tuple[Dict[str, Any], str]:
    # Minimal YAML-frontmatter parser for:
    # ---
    # key: value
    # ---
    if not raw.startswith("---"):
        return {}, raw
    parts = raw.split("\n")
    if len(parts) < 3 or parts[0].strip() != "---":
        return {}, raw
    fm_lines: List[str] = []
    body_start = None
    for i in range(1, len(parts)):
        if parts[i].strip() == "---":
            body_start = i + 1
            break
        fm_lines.append(parts[i])
    if body_start is None:
        return {}, raw
    data: Dict[str, Any] = {}
    for line in fm_lines:
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        key = k.strip()
        value = v.strip().strip('"').strip("'")
        if key:
            data[key] = value
    body = "\n".join(parts[body_start:])
    return data, body


def _skill_metadata(skill_dir: pathlib.Path) -> Optional[Dict[str, str]]:
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        return None
    raw = skill_md.read_text(encoding="utf-8", errors="replace")
    fm, _ = _parse_frontmatter(raw)
    name = str(fm.get("name", "")).strip()
    description = str(fm.get("description", "")).strip()
    if not name or not description:
        return None
    if len(name) > 64 or not _NAME_RE.match(name):
        return None
    if name != skill_dir.name:
        return None
    return {
        "name": name,
        "description": description,
        "path": str(skill_md),
        "dir": str(skill_dir),
    }


def discover_skills() -> List[Dict[str, str]]:
    skills_dir = get_skills_dir()
    if not skills_dir.is_dir():
        return []
    out: List[Dict[str, str]] = []
    for entry in skills_dir.iterdir():
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        meta = _skill_metadata(entry)
        if meta:
            out.append(meta)
    return sorted(out, key=lambda s: s["name"])


def get_skill_metadata(name: str) -> Optional[Dict[str, str]]:
    skills_dir = get_skills_dir()
    skill_dir = skills_dir / name
    if not skill_dir.is_dir():
        return None
    return _skill_metadata(skill_dir)


def get_skill_content(name: str) -> Optional[str]:
    meta = get_skill_metadata(name)
    if not meta:
        return None
    return pathlib.Path(meta["path"]).read_text(encoding="utf-8", errors="replace")


def resolve_skill_path(skill_name: str, relative_path: str) -> Optional[pathlib.Path]:
    meta = get_skill_metadata(skill_name)
    if not meta:
        return None
    base = pathlib.Path(meta["dir"]).resolve()
    # normalize and prevent path traversal
    rel = pathlib.Path(relative_path)
    normalized = pathlib.Path(os.path.normpath(str(rel))).as_posix().lstrip("/")
    if normalized.startswith(".."):
        return None
    resolved = (base / normalized).resolve()
    if resolved == base or str(resolved).startswith(str(base) + os.sep):
        return resolved
    return None


def read_skill_file(skill_name: str, relative_path: str) -> str:
    resolved = resolve_skill_path(skill_name, relative_path)
    if not resolved:
        return "error: skill not found or path outside skill directory"
    if not resolved.is_file():
        return "error: not a file" if resolved.exists() else "error: file not found"
    return resolved.read_text(encoding="utf-8", errors="replace")


def list_skill_files(skill_name: str) -> str:
    meta = get_skill_metadata(skill_name)
    if not meta:
        return "error: skill not found"
    base = pathlib.Path(meta["dir"]).resolve()
    lines: List[str] = []
    for d in ["scripts", "references", "assets"]:
        full = base / d
        if not full.is_dir():
            continue
        for entry in sorted(full.iterdir(), key=lambda p: p.name.lower()):
            rel = f"{d}/{entry.name}"
            lines.append(rel + "/" if entry.is_dir() else rel)
    return "\n".join(lines) if lines else "no optional directories (scripts/, references/, assets/)"


def build_available_skills_xml() -> str:
    skills = discover_skills()
    if not skills:
        return ""
    parts: List[str] = []
    for s in skills:
        parts.append(
            "\n".join(
                [
                    "<available_skill>",
                    f"  <name>{_escape_xml(s['name'])}</name>",
                    f"  <description>{_escape_xml(s['description'])}</description>",
                    f"  <path>{_escape_xml(s['path'])}</path>",
                    "</available_skill>",
                ]
            )
        )
    return "<available_skills>\n" + "\n".join(parts) + "\n</available_skills>"


def _escape_xml(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )

