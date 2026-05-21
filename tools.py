from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import urllib.request
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, TypedDict, Union, cast

import skills

WORKSPACE = pathlib.Path.cwd().resolve()


def _resolve_safe(rel_path: str) -> pathlib.Path:
    resolved = (WORKSPACE / rel_path).resolve()
    if resolved == WORKSPACE or str(resolved).startswith(str(WORKSPACE) + os.sep):
        return resolved
    raise ValueError("path must be inside workspace")


class ToolDef(TypedDict):
    type: Literal["function"]
    name: str
    description: str
    strict: bool
    parameters: Dict[str, Any]


TOOL_DEFINITIONS: List[ToolDef] = [
    {
        "type": "function",
        "name": "ping",
        "description": "Ping a host on the internet. Use to check reachability or latency.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Hostname or IP address to ping (e.g. 8.8.8.8 or google.com)",
                }
            },
            "required": ["host"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "bash",
        "description": "Run a bash command on the system. Use for shell scripting or to execute arbitrary shell commands. Dangerous! Use with care.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The shell command to run"}},
            "required": ["command"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "read_file",
        "description": "Read the full contents of a file in the workspace. Use to inspect source code, configs, or any text file. Path is relative to the project root.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "File path relative to project root"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "list_dir",
        "description": "List contents of a directory (files and subdirectories). Use to explore project structure. Path is relative to project root; use '.' for root.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path relative to project root (e.g. . or src)"},
                "recursive": {
                    "type": "boolean",
                    "description": "If true, list recursively with depth 2 (one level of subdirs). Default false.",
                },
            },
            "required": ["path", "recursive"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "fetch_url",
        "description": "Fetch content from a URL (GET). Use to read docs, APIs, or any public web page. Returns response body as text.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "Full URL to fetch (e.g. https://example.com/docs)"}},
            "required": ["url"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "search_files",
        "description": "Search for a string or regex in file contents under the workspace. Use to find where something is defined or used. Optionally limit by directory or file glob.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "String or regex pattern to search for"},
                "dir": {"type": "string", "description": "Directory to search in, relative to project root. Default: ."},
                "glob": {"type": "string", "description": "Optional file glob to limit matches (e.g. *.ts or **/*.md). Default: all files"},
            },
            "required": ["pattern", "dir", "glob"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "load_skill",
        "description": "Load an Agent Skill by name. Use when a task matches an available skill's description. Returns the full SKILL.md content (instructions). Call this to activate a skill before following its instructions.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Skill name (e.g. pdf-processing). Must match a skill listed in available_skills.",
                }
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "read_skill_file",
        "description": "Read a file inside a skill's scripts/, references/, or assets/ directory. Use after loading a skill when the skill instructions reference a bundled file.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "skill_name": {"type": "string", "description": "Skill name (e.g. pdf-processing)"},
                "path": {"type": "string", "description": "Path relative to skill root (e.g. references/REFERENCE.md or scripts/extract.py)"},
            },
            "required": ["skill_name", "path"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "list_skills",
        "description": "List available Agent Skills (name and description). Use to see what skills are installed when the user asks.",
        "strict": True,
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
]

def get_chat_tools() -> List[Dict[str, Any]]:
    """
    Return OpenAI/Berget Chat Completions `tools` schema:
      [{"type":"function","function":{"name":...,"description":...,"parameters":...}}]
    """
    out: List[Dict[str, Any]] = []
    for t in TOOL_DEFINITIONS:
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
        )
    return out

ToolName = cast(
    Any,
    Literal[
        "ping",
        "bash",
        "read_file",
        "list_dir",
        "fetch_url",
        "search_files",
        "load_skill",
        "read_skill_file",
        "list_skills",
    ],
)


def run_tool(name: str, args: Dict[str, Any]) -> str:
    if name == "ping":
        host = str(args.get("host", "")).strip()
        if not host:
            return "error: host must be a non-empty string"
        return _ping(host)
    if name == "bash":
        command = str(args.get("command", "")).strip()
        if not command:
            return "error: command must be a non-empty string"
        return _bash(command)
    if name == "read_file":
        path = str(args.get("path", "")).strip()
        if not path:
            return "error: path must be a non-empty string"
        return _read_file(path)
    if name == "list_dir":
        dir_path = str(args.get("path", ".")).strip() or "."
        recursive = args.get("recursive") is True
        return _list_dir(dir_path, recursive)
    if name == "fetch_url":
        url = str(args.get("url", "")).strip()
        if not url:
            return "error: url must be a non-empty string"
        return _fetch_url(url)
    if name == "search_files":
        pattern = str(args.get("pattern", "")).strip()
        if not pattern:
            return "error: pattern must be a non-empty string"
        dir_ = str(args.get("dir", ".")).strip() or "."
        glob = str(args.get("glob", "")).strip() or None
        return _search_files(pattern, dir_, glob)
    if name == "load_skill":
        skill_name = str(args.get("name", "")).strip()
        if not skill_name:
            return "error: name must be a non-empty string"
        content = skills.get_skill_content(skill_name)
        return content if content is not None else f'error: skill "{skill_name}" not found'
    if name == "read_skill_file":
        skill_name = str(args.get("skill_name", "")).strip()
        rel_path = str(args.get("path", "")).strip()
        if not skill_name:
            return "error: skill_name must be a non-empty string"
        if not rel_path:
            return "error: path must be a non-empty string"
        return skills.read_skill_file(skill_name, rel_path)
    if name == "list_skills":
        discovered = skills.discover_skills()
        if not discovered:
            return "No skills installed.\nAdd skill folders with SKILL.md to the .skills directory."
        return "\n\n".join([f"{s['name']}: {s['description']}" for s in discovered])

    return f'error: unknown tool "{name}"'


def _ping(host: str) -> str:
    try:
        proc = subprocess.run(
            ["ping", "-c", "5", host],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if proc.returncode == 0:
            return proc.stdout
        return f"error: {proc.stderr or proc.stdout or 'ping failed'}"
    except Exception as e:
        return f"error: {e}"


def _bash(command: str) -> str:
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            shell=True,
            executable="/bin/bash",
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode == 0:
            return out or "(ok)"
        return f"error: {out or 'command failed'}"
    except Exception as e:
        return f"error: {e}"


def _read_file(rel_path: str) -> str:
    try:
        resolved = _resolve_safe(rel_path)
        if not resolved.is_file():
            return "error: not a file"
        return resolved.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "error: file not found"
    except ValueError:
        return "error: path must be inside workspace"
    except Exception as e:
        return f"error: {e}"


def _list_dir(rel_dir: str, recursive: bool) -> str:
    try:
        resolved = _resolve_safe(rel_dir)
        if not resolved.is_dir():
            return "error: not a directory"

        base = pathlib.Path(rel_dir).as_posix() or "."

        def list_dir(dir_path: pathlib.Path, prefix: str, depth: int) -> List[str]:
            entries = sorted(dir_path.iterdir(), key=lambda p: p.name.lower())
            lines: List[str] = []
            for entry in entries:
                rel = f"{prefix}/{entry.name}" if prefix not in (".", "") else f"{base}/{entry.name}"
                rel = rel.replace("//", "/").lstrip("./")
                if entry.is_dir():
                    lines.append(rel + "/")
                    if recursive and depth < 2:
                        lines.extend(list_dir(entry, rel, depth + 1))
                else:
                    lines.append(rel)
            return lines

        items = list_dir(resolved, ".", 0)
        return "\n".join(items) if items else "(empty)"
    except FileNotFoundError:
        return "error: directory not found"
    except ValueError:
        return "error: path must be inside workspace"
    except Exception as e:
        return f"error: {e}"


def _fetch_url(url: str) -> str:
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "PolymathAgent/1.0"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read(300_000)
            try:
                return raw.decode("utf-8", errors="replace")
            except Exception:
                return raw.decode(errors="replace")
    except Exception as e:
        return f"error: {e}"


def _search_files(pattern: str, rel_dir: str, glob: Optional[str]) -> str:
    try:
        base = _resolve_safe(rel_dir)
        if not base.is_dir():
            return "error: not a directory"

        try:
            regex = re.compile(pattern)
        except re.error:
            regex = re.compile(re.escape(pattern))

        max_results = 200
        max_bytes = 500_000
        results: List[str] = []

        ext = None
        if glob:
            m = re.search(r"\*\.(\w+)$", glob)
            if m:
                ext = "." + m.group(1)

        def walk(dir_path: pathlib.Path) -> None:
            nonlocal results
            if len(results) >= max_results:
                return
            for entry in sorted(dir_path.iterdir(), key=lambda p: p.name.lower()):
                if entry.is_dir():
                    if entry.name.startswith(".") or entry.name == "node_modules":
                        continue
                    walk(entry)
                    continue
                if ext and entry.suffix != ext:
                    continue
                try:
                    data = entry.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                if len(data) > max_bytes:
                    continue
                for match in regex.finditer(data):
                    if len(results) >= max_results:
                        return
                    line_no = data.count("\n", 0, match.start()) + 1
                    rel = entry.relative_to(WORKSPACE).as_posix()
                    results.append(f"{rel}:{line_no}: {match.group(0)}")

        walk(base)
        return "\n".join(results) if results else "no matches"
    except FileNotFoundError:
        return "error: directory not found"
    except ValueError:
        return "error: path must be inside workspace"
    except Exception as e:
        return f"error: {e}"
