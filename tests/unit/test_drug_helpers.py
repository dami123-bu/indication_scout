"""Unit tests for helpers/drug_helpers."""

from indication_scout.helpers.drug_helpers import normalize_drug_name


def test_strips_hydrobromide():
    assert normalize_drug_name("Dextromethorphan Hydrobromide") == "dextromethorphan"


def test_strips_hydrochloride():
    assert normalize_drug_name("Sertraline Hydrochloride") == "sertraline"


def test_strips_sulfate():
    assert normalize_drug_name("Morphine Sulfate") == "morphine"


def test_strips_mesylate():
    assert normalize_drug_name("Imatinib Mesylate") == "imatinib"


def test_strips_tartrate():
    assert normalize_drug_name("Metoprolol Tartrate") == "metoprolol"


def test_strips_maleate():
    assert normalize_drug_name("Enalapril Maleate") == "enalapril"


def test_strips_phosphate():
    assert normalize_drug_name("Codeine Phosphate") == "codeine"


def test_strips_succinate():
    assert normalize_drug_name("Metoprolol Succinate") == "metoprolol"


def test_no_suffix_returned_lowercased():
    assert normalize_drug_name("Semaglutide") == "semaglutide"


def test_already_normalized():
    assert normalize_drug_name("ibuprofen") == "ibuprofen"


def test_does_not_strip_partial_suffix_match():
    # "sulfate" only stripped at end — "sulfated" should not match
    assert normalize_drug_name("Heparin Sulfated") == "heparin sulfated"


def test_preserves_valid_second_word():
    # "acetate" is not in SALT_SUFFIXES — second word should be kept
    assert (
        normalize_drug_name("Medroxyprogesterone Acetate")
        == "medroxyprogesterone acetate"
    )
