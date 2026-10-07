# Agentic development setup

This repository is designed to be worked on with AI coding agents (pi, Codex,
Claude Code, and others). The setup is vendor-neutral:

| Path | Committed? | Purpose |
|---|---|---|
| `AGENTS.md` | yes | Canonical instructions for all agents (agents.md standard). Contains hand-written project conventions plus the auto-managed GitNexus block. |
| `.agents/skills/` | yes | Portable skills (Agent Skills specification). Discovered natively by pi and readable by other tools. |
| `docs/agentic-setup.md` | yes | This file. |
| `.github/workflows/gitnexus.yml` | yes | CI job that maps PR diffs to indexed symbols. |
| `.gitnexus/` | no | GitNexus knowledge-graph index — generated, ~119 MB binary, contains machine paths. Regenerate locally. |
| `.claude/` | no | Claude Code local settings and generated skill adapters. |
| `.codex/` | no | Codex local config. |
| `.pi/` | no | pi local runtime state. |
| `CLAUDE.md` | yes | One-line pointer to `AGENTS.md` (Supabase pattern) so Claude Code reads the same canonical file. |
| `CLAUDE.local.md`, `AGENTS.override.md` | no | Personal overrides. |

## One-time setup

Requires Node.js 18+. From the repository root:

```bash
# Build the knowledge-graph index (~20s, ~119 MB in .gitnexus/)
npx -y gitnexus@1.6.12 analyze --index-only
```

`--index-only` builds the index without touching `AGENTS.md` or skill files.

Configure the MCP server for your client (interactive, picks up installed
editors/agents):

```bash
npx -y gitnexus@1.6.12 setup
```

## Keeping the index fresh

The index is local and regenerable — never commit it. Check freshness with:

```bash
node .gitnexus/run.cjs status
```

Refresh after pulling, merging, rebasing, or before graph-backed work:

```bash
node .gitnexus/run.cjs analyze --index-only
```

Reindexing is incremental (parse cache); a full cold analysis takes ~20s on
this repository.

## Updating AGENTS.md

Hand-written content lives above the GitNexus block. To refresh the
auto-generated GitNexus section after major structural changes:

```bash
node .gitnexus/run.cjs analyze
```

This updates the block between the GitNexus markers in `AGENTS.md` and leaves
all hand-written content intact. It may also refresh skill files under
`.claude/skills/` and `.agents/skills/` — commit any intentional updates there.
`analyze` may regenerate `CLAUDE.md` as a full copy; if that happens, restore
the one-line pointer (`@AGENTS.md`) and commit only the canonical file.

## Using the graph from an agent

With the MCP server running (or via the CLI), the main operations:

- `query <concept>` — find symbols and execution flows by concept.
- `context <symbol>` — callers, callees, tests for a symbol.
- `impact <symbol>` — blast radius before changing a symbol.
- `detect-changes --scope compare --base-ref dev` — map a branch diff to
  affected symbols and flows.
- `check --cycles` — structural checks (circular imports).

CI runs the same `detect-changes` step on every PR; see
`.github/workflows/gitnexus.yml`.
