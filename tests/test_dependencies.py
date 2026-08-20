"""The two-dependency constraint, enforced rather than documented.

The package exists so that it can be opened and run without a build step, an LLVM
toolchain, or a GPU. That property is only real if it is checked: a single
``import numba`` added in passing would quietly destroy the reason the package
exists. CI installs the declared dependencies with ``--no-deps``, so a stray import
fails there too. This test localises the failure to the offending module and name.
"""

import ast
import pathlib
import subprocess
import sys

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "vex_rddme"

# Third-party imports the package is permitted to make. Anything else is a new
# dependency and must be a deliberate change to pyproject.toml, not a drift.
ALLOWED_THIRD_PARTY = {"numpy", "matplotlib", "vex_rddme"}


def _stdlib_names():
    names = getattr(sys, "stdlib_module_names", None)
    if names is not None:
        return set(names)
    # Python 3.9 has no stdlib_module_names; fall back to the modules the package
    # actually uses from the standard library.
    return {
        "abc", "argparse", "collections", "contextlib", "dataclasses", "enum",
        "functools", "itertools", "json", "math", "pathlib", "sys", "typing",
        "warnings", "logging", "textwrap", "ast", "os", "time",
    }


STDLIB = _stdlib_names()


def _source_files():
    return sorted(SRC.rglob("*.py"))


def _top_level_imports(path):
    """Yield (module_root, lineno) for every import in a source file."""
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import within the package
                continue
            if node.module:
                yield node.module.split(".")[0], node.lineno


def test_source_tree_is_not_empty():
    """Guard the guard: an empty glob would make every check below vacuous."""
    assert _source_files(), f"no source files found under {SRC}"


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: p.name)
def test_no_undeclared_dependencies(path):
    offenders = [
        (name, lineno)
        for name, lineno in _top_level_imports(path)
        if name not in STDLIB and name not in ALLOWED_THIRD_PARTY
    ]
    assert not offenders, (
        f"{path.name} imports undeclared dependencies: "
        + ", ".join(f"{name!r} (line {lineno})" for name, lineno in offenders)
        + f". Allowed third-party imports are {sorted(ALLOWED_THIRD_PARTY)}. "
        "Adding a dependency requires editing pyproject.toml and this whitelist."
    )


def test_matplotlib_is_not_imported_at_package_import_time():
    """Importing vex_rddme must not drag in a plotting backend.

    The solver is usable in a headless or minimal environment; only ``viz`` needs
    matplotlib, and it imports it lazily so that ``import vex_rddme`` stays cheap.

    Run in a subprocess deliberately. Clearing ``sys.modules`` in-process would give
    the rest of the suite a *second* copy of the package, with a second
    ``GuardViolation`` class that ``pytest.raises`` in other test modules would fail
    to catch. It is a genuinely confusing failure that this test caused once already.
    """
    code = (
        "import sys; import vex_rddme; "
        "sys.exit(1 if 'matplotlib' in sys.modules else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(SRC.parents[2]),
    )
    assert result.returncode == 0, (
        "importing vex_rddme pulled in matplotlib; keep the import inside viz.py so "
        f"the solver works without a plotting backend.\nstderr: {result.stderr}"
    )


def _readme_blocks():
    import re

    readme = pathlib.Path(__file__).resolve().parents[1] / "README.md"
    return re.findall(r"```python\n(.*?)```", readme.read_text(), re.S)


def test_readme_has_the_expected_number_of_blocks():
    """Guard the guard: an empty match would make the execution test vacuous."""
    assert len(_readme_blocks()) >= 9


@pytest.mark.parametrize("index", range(len(_readme_blocks())))
def test_readme_block_runs_as_printed(index):
    """Every python block in the README must execute.

    The quickstart is the first thing a reader meets and the tutorials are the second,
    so a block that has drifted from the API is worse than no block at all. Each runs
    in a fresh namespace, because each is written to stand alone.
    """
    import contextlib
    import io
    import logging

    block = _readme_blocks()[index]
    logging.disable(logging.CRITICAL)      # guard INFO lines are not the subject here
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(block, f"README block {index}", "exec"),
                 {"__name__": "__readme__"})
    finally:
        logging.disable(logging.NOTSET)
