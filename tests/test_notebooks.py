"""Execute every notebook's code cells and assert their agreement claims hold.

A notebook that has drifted out of agreement with the package is worse than no
notebook: it is a demonstration that quietly demonstrates the wrong thing. These
tests run the cells in order with a reduced step count and check the discrepancy the
notebook advertises.

Step counts are overridden so the suite stays fast. The full counts in the notebooks
give tighter agreement, so a pass here at looser tolerance implies a pass there.
"""

import json
import pathlib

import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg")

NOTEBOOKS = sorted((pathlib.Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))

# Shrink the runs. Each entry maps a source substring to its replacement.
SPEEDUPS = {
    # Demonstration 1 has three independent parts, each with its own step count.
    # Part 3 needs the most: shrink its box as well, because relaxation scales as
    # L^2 and a shortened run on a 24-wide box is not equilibrated.
    "EOS_STEPS     = 60_000": "EOS_STEPS     = 12_000",
    "FLUCT_STEPS    = 40_000": "FLUCT_STEPS    = 9_000",
    "FLUCT_SAMPLE_EVERY = 100": "FLUCT_SAMPLE_EVERY = 30",
    "FLUCT_OCCUPANCIES = (3, 6)": "FLUCT_OCCUPANCIES = (6,)",
    "MIX_SHAPE     = (32, 24)": "MIX_SHAPE     = (16, 12)",
    "MIX_STEPS     = 80_000": "MIX_STEPS     = 20_000",
    # Demonstration 2
    "N_STEPS      = 40_000": "N_STEPS      = 8_000",
    "CROWDER_COUNTS = (0, 2, 4, 6)": "CROWDER_COUNTS = (0, 4)",
    # Demonstration 3
    "N_STEPS    = 30_000": "N_STEPS    = 6_000",
    # never open a window
    "plt.show()": "plt.close('all')",
}

# Looser than the notebooks' own claims, because the runs are shortened.
TOLERANCE = {
    "01_volume_excluded_lattice_gas": 0.30,
    "02_macromolecular_crowding": 0.60,
    "03_reversible_reaction_ramp_potential": 0.25,
}


def test_notebooks_exist():
    assert NOTEBOOKS, "no notebooks found"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_is_valid_json_with_code_cells(path):
    doc = json.loads(path.read_text())
    assert doc["nbformat"] == 4
    kinds = [c["cell_type"] for c in doc["cells"]]
    assert "code" in kinds and "markdown" in kinds
    assert kinds[0] == "markdown", "every notebook should open with an explanation"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_states_its_verdict(path):
    """Each notebook must print measured-vs-predicted, not just draw a plot."""
    src = "\n".join(
        "".join(c["source"]) for c in json.loads(path.read_text())["cells"]
        if c["cell_type"] == "code"
    )
    # The call form, not the import: `import report_comparison` alone would satisfy a
    # substring check while the notebook printed nothing comparative.
    assert "report_comparison(" in src, (
        "notebook must *call* report_comparison so it prints measured vs predicted, "
        "not merely import it"
    )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_runs_and_agrees(path):
    doc = json.loads(path.read_text())
    code = []
    for cell in doc["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        for old, new in SPEEDUPS.items():
            src = src.replace(old, new)
        code.append(src)

    ns = {"__name__": "__notebook__"}
    for i, src in enumerate(code):
        try:
            exec(compile(src, f"{path.name}[cell {i}]", "exec"), ns)
        except Exception as exc:  # pragma: no cover - diagnostic path
            pytest.fail(f"{path.name} cell {i} raised {type(exc).__name__}: {exc}")

    tol = TOLERANCE[path.stem]
    verdicts = ns.get("VERDICTS")
    assert verdicts, (
        f"{path.name} must set VERDICTS = {{label: discrepancy}} so its claim is "
        "checkable outside the notebook"
    )
    for label, value in verdicts.items():
        assert value < tol, (
            f"{path.name}: '{label}' discrepancy {value:.3f} exceeds {tol} "
            f"(shortened run; the notebook's own run is longer)"
        )
