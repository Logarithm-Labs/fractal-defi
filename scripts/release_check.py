"""Mechanical pre-release checks for fractal-defi (stdlib only, no network).

    make release-check                       # all checks except ``wheel``, base origin/main
    python scripts/release_check.py [CHECK ...] [--base origin/main]

Runs in CI on every release PR (``dev`` -> ``main``). On ``dev`` between
releases ``versions`` is expected to fail while ``[Unreleased]`` has entries.

Checks (default: all except ``wheel``):
    versions   setup.py / CITATION.cff / README citation / CHANGELOG agree; [Unreleased] is empty
    deps       dependency floors identical in setup.py, requirements.txt, .pre-commit-config.yaml
    docs       every fractal module has a docs/source page that is listed in a toctree
    exports    every name in an ``__all__`` exists in its module
    data       no csv/ipynb/pickle/parquet/.env added since --base
    examples   every ``fractal.*`` import in examples/ (py + ipynb) resolves against the installed package
    wheel      the built wheel in dist/ ships every tracked fractal/*.py (run after ``make build``/``make smoke``)

Exit code 1 if any check reports a problem. Output is one line per problem.
"""
import argparse
import ast
import importlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_PATHS = ("fractal", "setup.py", "requirements.txt")
PROBLEMS: list[str] = []


def problem(check: str, msg: str) -> None:
    PROBLEMS.append(f"[{check}] {msg}")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


# ---------------------------------------------------------------- versions
def check_versions(_base: str) -> None:
    setup_v = re.search(r"version='([^']+)'", (ROOT / "setup.py").read_text()).group(1)
    cff = (ROOT / "CITATION.cff").read_text()
    cff_v = re.search(r"^version: (\S+)", cff, re.M).group(1)
    cff_date = re.search(r"^date-released: (\S+)", cff, re.M).group(1)
    readme_v = re.search(r"version\s*=\s*\{([^}]+)\}", (ROOT / "README.md").read_text()).group(1)
    changelog = (ROOT / "CHANGELOG.md").read_text()
    head = re.search(r"^## \[v([^\]]+)\] — (\S+)", changelog, re.M)
    sources = (("CITATION.cff", cff_v), ("README citation", readme_v), ("CHANGELOG top section", head.group(1)))
    for name, value in sources:
        if value != setup_v:
            problem("versions", f"{name} says {value}, setup.py says {setup_v}")
    if head.group(2) != cff_date:
        problem("versions", f"CHANGELOG date {head.group(2)} != CITATION.cff date-released {cff_date}")
    unreleased = re.search(r"^## \[Unreleased\]\n(.*?)^## \[", changelog, re.M | re.S)
    if unreleased and unreleased.group(1).strip():
        problem("versions", "CHANGELOG [Unreleased] is not empty: fold it into the release section")
    tags = git("tag", "--list", f"v{setup_v}").strip()
    # A docs-only follow-up to an already tagged release keeps its version.
    if tags and git("diff", "--name-only", f"v{setup_v}", "HEAD", "--", *_PACKAGE_PATHS).strip():
        problem("versions", f"tag v{setup_v} already exists and the package changed since: bump the version")


# ---------------------------------------------------------------- deps
_REQ = re.compile(r"^\s*-?\s*[\"']?([A-Za-z0-9_.\-\[\]]+[<>=!~][^\"'\s,#]+)")


def _reqs(lines) -> dict[str, str]:
    out = {}
    for line in lines:
        m = _REQ.match(line)
        if m:
            spec = m.group(1)
            name = re.split(r"[<>=!~\[]", spec, maxsplit=1)[0].lower()
            out[name] = spec
    return out


def check_deps(_base: str) -> None:
    setup_text = (ROOT / "setup.py").read_text()
    setup = _reqs(re.findall(r"\"([^\"]+[<>=][^\"]+)\"", setup_text))
    reqs = _reqs((ROOT / "requirements.txt").read_text().splitlines())
    if setup != reqs:
        for name in sorted(set(setup) | set(reqs)):
            if setup.get(name) != reqs.get(name):
                problem("deps", f"{name}: setup.py={setup.get(name)} requirements.txt={reqs.get(name)}")
    precommit = (ROOT / ".pre-commit-config.yaml").read_text()
    for block in re.findall(r"additional_dependencies:\n((?:\s+- .+\n)+)", precommit):
        for name, spec in _reqs(block.splitlines()).items():
            if setup.get(name) != spec:
                problem("deps", f"pre-commit additional_dependencies {spec} != setup.py {setup.get(name)}")
    ruff_rev = re.search(r"ruff-pre-commit\s*\n\s*rev:\s*v?(\S+)", precommit)
    if ruff_rev and setup.get("ruff") and not setup["ruff"].endswith(ruff_rev.group(1)):
        problem("deps", f"pre-commit ruff rev {ruff_rev.group(1)} != setup.py {setup['ruff']}")


# ---------------------------------------------------------------- docs
def check_docs(_base: str) -> None:
    src = ROOT / "docs" / "source"
    pages = {p.stem for p in src.glob("fractal*.rst")}
    toctree_entries = set()
    for page in src.glob("*.rst"):
        toctree_entries |= set(re.findall(r"^\s{3}(fractal[\w.]*)\s*$", page.read_text(), re.M))
    for path in sorted((ROOT / "fractal").rglob("*.py")):
        rel = path.relative_to(ROOT).with_suffix("")
        parts = rel.parts
        if any(p.startswith("_") and p != "__init__" for p in parts) or "__pycache__" in parts:
            continue  # private modules are documented through their package
        module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        if module not in pages:
            problem("docs", f"{module} has no docs/source/{module}.rst page")
        elif module != "fractal" and module not in toctree_entries:
            problem("docs", f"{module}.rst exists but is in no toctree")


# ---------------------------------------------------------------- exports
def check_exports(_base: str) -> None:
    sys.path.insert(0, str(ROOT))
    for init in sorted((ROOT / "fractal").rglob("*.py")):
        tree = ast.parse(init.read_text())
        names = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "__all__" for t in node.targets):
                names = [elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)]
        if not names:
            continue
        rel = init.relative_to(ROOT).with_suffix("")
        module = ".".join(rel.parts[:-1] if rel.parts[-1] == "__init__" else rel.parts)
        mod = importlib.import_module(module)
        for name in names:
            if not hasattr(mod, name):
                problem("exports", f"{module}.__all__ lists {name!r}, which does not exist")
        if len(names) != len(set(names)):
            problem("exports", f"{module}.__all__ has duplicates")


# ---------------------------------------------------------------- data files
def check_data(base: str) -> None:
    added = git("diff", "--name-only", "--diff-filter=A", f"{base}..HEAD").split()
    for path in added:
        if not re.search(r"\.(csv|ipynb|pickle|pkl|parquet)$|(^|/)\.env($|\.)", path) or path.endswith(".env.example"):
            continue
        rule = subprocess.run(["git", "check-ignore", "-v", "--no-index", path], cwd=ROOT, check=False,
                              capture_output=True, text=True).stdout
        pattern = rule.split("\t")[0].split(":", 2)[-1] if rule else ""
        if not pattern.startswith("!"):  # tracked without an explicit whitelist entry (force-added)
            problem("data", f"{path} added since {base} without a .gitignore whitelist entry")


# ---------------------------------------------------------------- examples
def _fractal_imports(source: str, origin: str):
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        problem("examples", f"{origin}: syntax error {exc.msg} at line {exc.lineno}")
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("fractal"):
            for alias in node.names:
                yield node.module, alias.name, node.lineno
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("fractal"):
                    yield alias.name, None, node.lineno


def check_examples(_base: str) -> None:
    sys.path.insert(0, str(ROOT))
    sources = []
    for rel in git("ls-files", "examples").split():
        path = ROOT / rel
        if rel.endswith(".py"):
            sources.append((rel, path.read_text()))
        elif rel.endswith(".ipynb"):
            for i, cell in enumerate(json.loads(path.read_text()).get("cells", [])):
                if cell.get("cell_type") == "code":
                    lines = "".join(cell.get("source", [])).splitlines()
                    code = "\n".join("" if re.match(r"\s*[%!]", line) else line for line in lines)
                    sources.append((f"{rel}[cell {i}]", code))
    for origin, source in sources:
        for module, name, lineno in _fractal_imports(source, origin):
            try:
                mod = importlib.import_module(module)
            except ImportError as exc:
                problem("examples", f"{origin}:{lineno}: cannot import {module} ({exc})")
                continue
            if name and name != "*" and not hasattr(mod, name):
                try:
                    importlib.import_module(f"{module}.{name}")
                except ImportError:
                    problem("examples", f"{origin}:{lineno}: {module} has no {name}")


# ---------------------------------------------------------------- wheel
def check_wheel(_base: str) -> None:
    wheels = sorted((ROOT / "dist").glob("*.whl"))
    if not wheels:
        problem("wheel", "no wheel in dist/: run `make build` (or `make smoke`) first")
        return
    shipped = set(zipfile.ZipFile(wheels[-1]).namelist())
    tracked = [p for p in git("ls-files", "fractal").split() if p.endswith(".py")]
    for path in tracked:
        if path not in shipped:
            problem("wheel", f"{wheels[-1].name} is missing {path}")


CHECKS = {
    "versions": check_versions, "deps": check_deps, "docs": check_docs, "exports": check_exports,
    "data": check_data, "examples": check_examples, "wheel": check_wheel,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checks", nargs="*", help=f"any of: {', '.join(CHECKS)}")
    parser.add_argument("--base", default="origin/main", help="last released ref (default origin/main)")
    args = parser.parse_args()
    unknown = set(args.checks) - set(CHECKS)
    if unknown:
        parser.error(f"unknown check(s): {', '.join(sorted(unknown))}")
    selected = args.checks or [c for c in CHECKS if c != "wheel"]
    for name in selected:
        before = len(PROBLEMS)
        CHECKS[name](args.base)
        print(f"{name:9} {'OK' if len(PROBLEMS) == before else f'{len(PROBLEMS) - before} problem(s)'}")
    for line in PROBLEMS:
        print("  " + line)
    return 1 if PROBLEMS else 0


if __name__ == "__main__":
    sys.exit(main())
