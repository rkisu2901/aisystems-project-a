# Project Instructions

You are working on **Novus Bank RAG assistant** (Python + pgvector + OpenAI), with a persistent **LLM wiki** in `../wiki/` (shared across Project A and Project B) that records everything we build, decide, and change.

Two systems, two jobs:
- **Workshop** — `.claude/skills/code-review/` and `.claude/skills/implement-change/` for effective coding.
- **Library** — `.claude/skills/wiki-*` for institutional memory. Read before re-explaining; write to preserve.

## Hard rules (also enforced by hooks)

- `raw/**` is **immutable** — never modify any file under it.
- `../wiki/log.md` is **append-only** — never edit past entries.
- Every entity mention with its own wiki page uses `[[wikilink]]` syntax.
- Page schemas live in `../wiki/SCHEMAS.md` — consult before creating any wiki page.

## Skills (load on demand by name)

- `wiki-write` · `wiki-read` · `wiki-trace` · `wiki-maintain` · `wiki-map`
- `code-review` · `implement-change` (auto-chains code-review + wiki-write)

When a task fits a skill's trigger, invoke it. Skills override default behaviour for that procedure.

## Session start

1. Read `../wiki/log.md` (last 10 entries).
2. Read `../wiki/index.md`.
3. Greet with: page count, last activity date, open work from the most recent log entries.

## Codebase

Production code lives in `novus-rag-assistant/` (separate `CLAUDE.md` in there for codebase-specific notes). Source inputs (PRs, docs, code snapshots) live under `raw/`.

## Wiki location

The wiki is shared with Project B. It lives at `../wiki/` (one level up, at `Week-1/wiki/`).
- Project A pages: modules, apis, data-models, flows, decisions, debt, scaling, concepts, analyses
- Project B pages: will be added under the same wiki with `project-b/` prefixes where needed
