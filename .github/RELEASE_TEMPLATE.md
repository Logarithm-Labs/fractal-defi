<!--
============================================================================
Release-PR description template (also reusable as GitHub Release notes).

Workflow:
  1. Open the dev → main release PR (per gitflow). Title: ``vX.Y.Z``.
  2. Copy this whole file into the PR description.
  3. Replace every ``<...>`` placeholder. Drop sections that don't apply
     (HTML comments tell you which are conditional).
  4. Run the ``Pre-merge checklist`` commands locally; tick each box.
  5. After squash-merge, create a GitHub Release from the ``vX.Y.Z`` tag —
     paste the same body (or use ``--generate-notes`` and trim).

Keep it human-readable. The ``CHANGELOG.md`` already holds the exhaustive
list — this template is the elevator pitch.
============================================================================
-->

## Release v<X.Y.Z>

<!-- One paragraph. What is this release about? Who should care? Mention
     default-behaviour changes (or lack thereof) explicitly so users skim
     the right thing. Two-three sentences max. -->

<one-paragraph summary>

### Highlights

<!-- 4-8 bullets, each one sentence. User-visible only — no PR-internal
     jargon (Codex M1/L2/etc), no "moved this constant" cleanups. Think
     "what would I tell someone deciding whether to upgrade?" -->

- <highlight 1>
- <highlight 2>
- <highlight 3>

### Migration

<!-- DROP THIS SECTION IF NOTHING BREAKS. Otherwise list every backward-
     incompatible change with a one-line "what to do". Floor-pin bumps
     that exclude older versions count; behavioural changes in default
     code paths count; deprecated-now-removed APIs count. -->

- <breaking change 1 — what users should do>
- <breaking change 2 — what users should do>

### Full changelog

See [`CHANGELOG.md`](../CHANGELOG.md) `[v<X.Y.Z>]` section for the
complete list grouped by area.

### Pre-merge checklist

<!-- Run these locally before requesting review. The numbers come from
     ``pytest -m <marker> --collect-only --no-header`` — paste the
     ``X/Y collected`` line so reviewer can sanity-check. Tick boxes
     only after the command actually finished green; an unchecked box
     is a real to-do, not a placeholder. -->

- [ ] `make smoke` green locally (build wheel + install + tests against it)
- [ ] `make pre-commit` green (lint at 10/10, isort clean)
- [ ] `make docs-strict` builds with 0 warnings
- [ ] `pytest -m core` <N>/<N> ✓
- [ ] `pytest -m slow` <N>/<N> ✓
- [ ] `pytest -m integration` <N>/<N> ✓ (with `THE_GRAPH_API_KEY`)
- [ ] `CHANGELOG.md` `[v<X.Y.Z>]` section dated (no longer `Unreleased`)
- [ ] `setup.py::version` matches the release tag
- [ ] `fractal.__version__` matches `setup.py::version`
- [ ] Release-context CI green on this PR (lint, docs, slow, integration, smoke, e2e)

### Post-merge actions

<!-- Reminder for what happens AFTER squash-merge. Not a checklist for
     the PR reviewer — but having it inline keeps you from forgetting. -->

```bash
# 1. Pull main locally
git checkout main && git pull

# 2. Build + smoke
make clean && make smoke

# 3. (optional) TestPyPI dry-run
make release-test

# 4. Production PyPI
make release

# 5. Tag the release commit + push tag
git tag -a v<X.Y.Z> -m "v<X.Y.Z>"
git push origin v<X.Y.Z>

# 6. GitHub Release from the tag, attach wheel + sdist
gh release create v<X.Y.Z> --generate-notes \
    dist/fractal_defi-<X.Y.Z>-py3-none-any.whl \
    dist/fractal_defi-<X.Y.Z>.tar.gz

# 7. Re-align dev with main (post squash-merge sync)
git checkout dev && make post-release
```
