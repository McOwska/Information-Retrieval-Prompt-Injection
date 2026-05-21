# Agents (Python)

This is a small Python re-implementation of the TypeScript agent template in [BirgerMoell/agents](https://github.com/BirgerMoell/agents), adapted to use Berget's OpenAI-compatible Chat Completions API with tool calling in a loop.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `BERGET_API_KEY` (or `OPENAI_API_KEY`) in `.env`, then:

```bash
python agent.py
```

## Skills

The agent supports a local `.skills/` directory containing Agent Skills-style folders:

```
.skills/
  example-skill/
    SKILL.md
    references/
      ...
    scripts/
      ...
```

At startup it injects an `<available_skills>` block into the system prompt, and exposes:

- `list_skills`
- `load_skill`
- `read_skill_file`
