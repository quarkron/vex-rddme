"""The STE linter, and the guarantee that operator-facing text stays compliant.

Two jobs here. The first is to test the linter itself. A checker that silently
passes everything is worse than none. The second is the regression guarantee:
guard messages and the README must stay STE-clean, so prose cannot drift back.
"""

import pathlib
import subprocess
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))

from ste_lint import (  # noqa: E402
    MAX_WORDS_DEFAULT,
    REJECTED_SUBSTITUTIONS,
    SUBSTITUTIONS,
    check_text,
    guard_messages,
    main,
    sentences,
    words,
)

REPO = pathlib.Path(__file__).resolve().parents[1]


def rules(text, max_words=MAX_WORDS_DEFAULT):
    return {rule for _, rule, _ in check_text(text, "t", max_words)}


# ------------------------------------------------------- the linter detects things


def test_detects_a_long_sentence():
    long = "This " + "word " * 30 + "ends."
    assert "sentence-length" in rules(long)
    assert "sentence-length" not in rules("This sentence is short.")


@pytest.mark.parametrize("text", [
    "The value is clamped silently.",
    "The result is then computed.",
    "Flux would be silently discarded.",
])
def test_detects_passive_voice(text):
    assert "passive-voice" in rules(text)


@pytest.mark.parametrize("text", [
    "Reduce tau.",
    "The packing fraction reached 0.9995.",
    "This is not a physical state.",
    "Exclusion is the binding constraint.",
    "The limit is 1.",
])
def test_does_not_flag_clean_text(text):
    assert rules(text) == set()


def test_then_is_not_a_participle():
    """'then' ends in -en. Without a stoplist the checker reports 'is then' as passive,
    which it did until the stoplist was added."""
    assert "passive-voice" not in rules("The admissible tau is then below the bound.")


def test_detects_multiple_actions_in_one_instruction():
    assert "multi-action" in rules("Reduce tau, raise the cap, or use fewer particles.")
    assert "multi-action" not in rules("Reduce tau.")
    # only imperative sentences are candidates
    assert "multi-action" not in rules("The cap, the diameters, or the voxel size set it.")


def test_detects_wordy_substitutions():
    assert "word-choice" in rules("Use this in order to start.")
    assert "word-choice" in rules("Utilise the smaller voxel.")


# --------------------------------------------------- the substitution list is safe


@pytest.mark.parametrize("word", sorted(REJECTED_SUBSTITUTIONS))
def test_physics_colliding_words_are_not_substituted(word):
    """The aerospace list would rewrite identifiers and maths terms. It must not."""
    assert word not in SUBSTITUTIONS, (
        f"{word!r} must stay out of SUBSTITUTIONS: {REJECTED_SUBSTITUTIONS[word]}"
    )


def test_codebase_terms_survive_the_linter():
    """Sentences using this domain's real vocabulary must not trip word-choice."""
    for text in [
        "The feasible mask selects the voxels where the reaction can occur.",
        "Evaluate the free-energy function at the mean composition.",
        "Fit the curve, then raise a GuardViolation if the residual grows.",
        "Set the pressure and the packing fraction.",
    ]:
        assert "word-choice" not in rules(text), text


# ----------------------------------------------------------------- tokenising


def test_placeholders_count_as_one_word():
    # Voxel / X / holds / X / particles.  -> an f-string field counts as one word,
    # not zero, so a message full of interpolated values is not scored as short.
    assert words("Voxel {where} holds {worst} particles.") == [
        "Voxel", "X", "holds", "X", "particles.",
    ]


def test_bullets_are_separate_sentences():
    s = sentences("Do one of these steps:\n  - Reduce tau.\n  - Raise the cap.")
    assert "Reduce tau." in s and "Raise the cap." in s


# -------------------------------------------------- the regression guarantee


def test_every_guard_message_is_ste_clean():
    """Guard messages are procedural, operator-facing text. They must stay compliant."""
    offenders = []
    for where, msg in guard_messages(REPO / "src"):
        for _, rule, detail in check_text(msg, where, MAX_WORDS_DEFAULT):
            offenders.append(f"{where}: {rule}: {detail}")
    assert not offenders, "guard messages drifted from STE:\n  " + "\n  ".join(offenders)


def test_guard_messages_were_actually_found():
    """Guard the guard: an empty extraction would make the check above vacuous."""
    found = list(guard_messages(REPO / "src"))
    assert len(found) >= 15, f"expected the guard messages, found {len(found)}"


def test_readme_is_ste_clean():
    assert main(["--targets", "readme", "--strict"]) == 0


def test_cli_runs_and_reports_clean():
    r = subprocess.run(
        [sys.executable, str(REPO / "tools" / "ste_lint.py"), "--strict"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "0 violation" in r.stdout


def test_cli_fails_on_a_relaxed_then_tightened_limit():
    """--max-words must actually bind, or --strict proves nothing."""
    r = subprocess.run(
        [sys.executable, str(REPO / "tools" / "ste_lint.py"),
         "--strict", "--max-words", "3"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert r.returncode == 1
    assert "sentence-length" in r.stdout


# ------------------------------------------------ typography regression guards
#
# The forbidden strings are assembled from codepoints and fragments, so this file
# contains neither of them literally. That lets the checks scan every tracked file
# including this one, instead of exempting themselves.

EM_DASH = chr(0x2014)
EN_DASH = chr(0x2013)
FORBIDDEN_WORD = "r" + "ung"


def _decoded_text(path):
    """File text, with notebook cell sources decoded from their JSON escapes.

    A notebook stores an em dash as six characters, a backslash-u escape, so a plain
    grep over the file misses it. That is how 28 em dashes survived the first pass.
    """
    import json

    raw = path.read_text()
    if path.suffix == ".ipynb":
        return "\n".join("".join(c["source"]) for c in json.loads(raw)["cells"])
    return raw


def _tracked_files():
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                         cwd=str(REPO)).stdout.split()
    return [REPO / f for f in out if (REPO / f).exists()]


def test_tracked_files_were_found():
    """Guard the guard: an empty listing would make the checks below vacuous."""
    assert len(_tracked_files()) >= 25


def test_this_file_contains_no_forbidden_literal():
    """Confirms the assembly trick works, so the scans below really cover this file."""
    raw = pathlib.Path(__file__).read_text()
    assert EM_DASH not in raw
    assert FORBIDDEN_WORD not in raw


def test_no_em_dashes_anywhere():
    offenders = {p.name: n for p in _tracked_files()
                 if (n := _decoded_text(p).count(EM_DASH))}
    assert not offenders, f"em dashes reappeared: {offenders}"


def test_en_dashes_survive_in_proper_names():
    """En dashes join surnames and are correct typography. They must not be swept up
    with the em dashes."""
    readme = _decoded_text(REPO / "README.md")
    assert EN_DASH in readme
    assert f"Wang{EN_DASH}Peskin{EN_DASH}Elston" in readme
    assert f"Fr\u00f6hner{EN_DASH}No\u00e9" in readme


def test_minus_signs_survive_in_maths():
    """U+2212 is a maths minus, not a dash to remove."""
    assert chr(0x2212) in _decoded_text(REPO / "README.md")


def test_the_demonstrations_are_called_demonstrations():
    offenders = {p.name: n for p in _tracked_files()
                 if (n := _decoded_text(p).lower().count(FORBIDDEN_WORD))}
    assert not offenders, f"the old word reappeared: {offenders}"


def test_the_three_demonstrations_are_present():
    names = sorted(p.name for p in (REPO / "notebooks").glob("*.ipynb"))
    assert names == [
        "01_volume_excluded_lattice_gas.ipynb",
        "02_macromolecular_crowding.ipynb",
        "03_reversible_reaction_ramp_potential.ipynb",
    ], names
