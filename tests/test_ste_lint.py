"""The STE linter, and the guarantee that operator-facing text stays compliant.

Two jobs here. The first is to test the linter itself — a checker that silently
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
