"""Unit tests for ClinicalTrialsClient helper functions."""

import pytest

from indication_scout.data_sources.clinical_trials import (
    ClinicalTrialsClient,
    _classify_stop_reason,
)
from indication_scout.models.model_clinical_trials import MeshTerm, Trial

# --- _classify_stop_reason keyword-based classification ---


@pytest.mark.parametrize(
    "why_stopped,expected_category",
    [
        # Efficacy keywords
        ("Lack of efficacy in interim analysis", "efficacy"),
        ("Study stopped due to futility", "efficacy"),
        ("No benefit observed in treatment arm", "efficacy"),
        ("Primary endpoint not met - lack of efficacy", "efficacy"),
        # Safety keywords
        ("Safety signal observed in treatment arm", "safety"),
        ("Serious adverse events reported", "safety"),
        ("Stopped due to toxicity concerns", "safety"),
        ("Unexpected side effect in patients", "safety"),
        # Enrollment keywords
        ("Difficulty with patient enrollment", "enrollment"),
        ("Slow accrual rate", "enrollment"),
        ("Recruitment challenges", "enrollment"),
        # Business keywords
        ("The trial was terminated for strategic reasons.", "business"),
        ("Sponsor business decision", "business"),
        ("Funding withdrawn", "business"),
        ("Commercial considerations", "business"),
        # Negation - safety keyword negated
        ("no safety concerns observed", "other"),
        ("unrelated to adverse events", "other"),
        # Negation - efficacy keyword negated
        ("Stopped, not due to efficacy concerns", "other"),
        ("No efficacy issues were observed", "other"),
        ("no lack of efficacy", "other"),
        # New safety keywords
        ("safety signal detected in interim review", "safety"),
        ("safety concern raised by DSMB", "safety"),
        ("FDA issued a clinical hold", "safety"),
        # New efficacy keyword
        ("no significant difference between arms", "efficacy"),
        # Other - no matching keywords
        ("COVID-19 pandemic impact", "other"),
        ("Protocol amendment required", "other"),
        ("Regulatory hold", "other"),
        # Unknown - no text
        (None, "unknown"),
        ("", "unknown"),
    ],
)
def test_classify_stop_reason(why_stopped, expected_category):
    """_classify_stop_reason should classify based on keywords."""
    assert _classify_stop_reason(why_stopped) == expected_category


def test_classify_stop_reason_case_insensitive():
    """_classify_stop_reason should match keywords case-insensitively."""
    assert _classify_stop_reason("LACK OF EFFICACY") == "efficacy"
    assert _classify_stop_reason("Safety Concerns") == "safety"
    assert _classify_stop_reason("BUSINESS Decision") == "business"


# --- ClinicalTrialsClient static helper methods ---


@pytest.mark.parametrize(
    "phases,expected",
    [
        # Single phases
        (["PHASE1"], "Phase 1"),
        (["PHASE2"], "Phase 2"),
        (["PHASE3"], "Phase 3"),
        (["PHASE4"], "Phase 4"),
        (["EARLY_PHASE1"], "Early Phase 1"),
        (["NA"], "Not Applicable"),
        # Combined phases
        (["PHASE1", "PHASE2"], "Phase 1/Phase 2"),
        (["PHASE2", "PHASE3"], "Phase 2/Phase 3"),
        (["PHASE3", "PHASE4"], "Phase 3/Phase 4"),
        # Empty list
        ([], "Not Applicable"),
    ],
)
def test_normalize_phase(phases, expected):
    """_normalize_phase should convert API phase list to display string."""
    assert ClinicalTrialsClient._normalize_phase(phases) == expected


@pytest.mark.parametrize(
    "phase,expected_rank",
    [
        ("Not Applicable", 0),
        ("Early Phase 1", 1),
        ("Phase 1", 2),
        ("Phase 1/Phase 2", 3),
        ("Phase 2", 4),
        ("Phase 2/Phase 3", 5),
        ("Phase 3", 6),
        ("Phase 3/Phase 4", 7),
        ("Phase 4", 8),
        # Unknown phase defaults to 0
        ("Unknown", 0),
        ("", 0),
    ],
)
def test_phase_rank(phase, expected_rank):
    """_phase_rank should return numeric ranking for phase comparison."""
    assert ClinicalTrialsClient._phase_rank(phase) == expected_rank


def test_phase_rank_ordering():
    """_phase_rank should order phases correctly: Phase 4 > Phase 3 > Phase 2 > Phase 1."""
    assert ClinicalTrialsClient._phase_rank(
        "Phase 4"
    ) > ClinicalTrialsClient._phase_rank("Phase 3")
    assert ClinicalTrialsClient._phase_rank(
        "Phase 3"
    ) > ClinicalTrialsClient._phase_rank("Phase 2")
    assert ClinicalTrialsClient._phase_rank(
        "Phase 2"
    ) > ClinicalTrialsClient._phase_rank("Phase 1")
    assert ClinicalTrialsClient._phase_rank(
        "Phase 1"
    ) > ClinicalTrialsClient._phase_rank("Early Phase 1")


@pytest.mark.parametrize(
    "date_struct,expected",
    [
        ({"date": "2021-03-15"}, "2021-03-15"),
        ({"date": "2024-01"}, "2024-01"),
        ({}, None),
        (None, None),
    ],
)
def test_extract_date(date_struct, expected):
    """_extract_date should extract date string from v2 date struct."""
    assert ClinicalTrialsClient._extract_date(date_struct) == expected


# --- _filter_by_mesh ---


def _trial_with_mesh(
    nct_id: str,
    conditions: list[tuple[str, str]],
    ancestors: list[tuple[str, str]],
) -> Trial:
    """Build a Trial with the given MeSH conditions and ancestors."""
    return Trial(
        nct_id=nct_id,
        title=f"Trial {nct_id}",
        phase="Phase 2",
        overall_status="COMPLETED",
        mesh_conditions=[MeshTerm(id=mid, term=term) for mid, term in conditions],
        mesh_ancestors=[MeshTerm(id=mid, term=term) for mid, term in ancestors],
    )


def test_filter_by_mesh_target_in_conditions_kept():
    """Trial is kept when target_mesh_id matches a mesh_conditions id."""
    trial = _trial_with_mesh(
        "NCT00000001",
        conditions=[("D006973", "Hypertension")],
        ancestors=[],
    )
    result = ClinicalTrialsClient._filter_by_mesh([trial], "D006973")
    assert len(result) == 1
    assert result[0].nct_id == "NCT00000001"


def test_filter_by_mesh_target_in_ancestors_kept():
    """Trial is kept when target_mesh_id matches a mesh_ancestors id."""
    trial = _trial_with_mesh(
        "NCT00000002",
        conditions=[("D003924", "Diabetes Mellitus, Type 2")],
        ancestors=[("D008659", "Metabolic Diseases")],
    )
    result = ClinicalTrialsClient._filter_by_mesh([trial], "D008659")
    assert len(result) == 1
    assert result[0].nct_id == "NCT00000002"


def test_filter_by_mesh_unrelated_dropped():
    """Trial is dropped when target_mesh_id matches neither conditions nor ancestors."""
    trial = _trial_with_mesh(
        "NCT00000003",
        conditions=[("D005901", "Glaucoma")],
        ancestors=[("D005128", "Eye Diseases")],
    )
    result = ClinicalTrialsClient._filter_by_mesh([trial], "D006973")
    assert result == []


def test_filter_by_mesh_empty_mesh_dropped():
    """Trial is dropped when both mesh_conditions and mesh_ancestors are empty."""
    trial = _trial_with_mesh("NCT00000004", conditions=[], ancestors=[])
    result = ClinicalTrialsClient._filter_by_mesh([trial], "D006973")
    assert result == []


def test_filter_by_mesh_mixed_list():
    """Across a mixed list, only trials matching target are kept."""
    kept_cond = _trial_with_mesh(
        "NCT_KEEP_COND",
        conditions=[("D006973", "Hypertension")],
        ancestors=[],
    )
    kept_anc = _trial_with_mesh(
        "NCT_KEEP_ANC",
        conditions=[("D003924", "Diabetes Mellitus, Type 2")],
        ancestors=[("D006973", "Hypertension")],
    )
    dropped_unrelated = _trial_with_mesh(
        "NCT_DROP_UNREL",
        conditions=[("D005901", "Glaucoma")],
        ancestors=[("D005128", "Eye Diseases")],
    )
    dropped_empty = _trial_with_mesh("NCT_DROP_EMPTY", conditions=[], ancestors=[])

    result = ClinicalTrialsClient._filter_by_mesh(
        [kept_cond, kept_anc, dropped_unrelated, dropped_empty], "D006973"
    )
    nct_ids = [t.nct_id for t in result]
    assert nct_ids == ["NCT_KEEP_COND", "NCT_KEEP_ANC"]


# --- _parse_trial populates mesh_ancestors ---


def test_parse_trial_populates_mesh_ancestors():
    """_parse_trial extracts ancestors from derivedSection.conditionBrowseModule."""
    study = {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT77777777",
                "briefTitle": "Ancestor fixture trial",
            },
            "statusModule": {"overallStatus": "COMPLETED"},
            "designModule": {"phases": ["PHASE2"]},
            "conditionsModule": {"conditions": ["Type 2 Diabetes Mellitus"]},
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Sponsor"}},
            "armsInterventionsModule": {
                "interventions": [{"type": "DRUG", "name": "DrugY"}]
            },
            "outcomesModule": {},
            "referencesModule": {},
            "descriptionModule": {},
        },
        "derivedSection": {
            "conditionBrowseModule": {
                "meshes": [{"id": "D003924", "term": "Diabetes Mellitus, Type 2"}],
                "ancestors": [
                    {"id": "D003920", "term": "Diabetes Mellitus"},
                    {"id": "D044882", "term": "Glucose Metabolism Disorders"},
                ],
            }
        },
    }

    client = ClinicalTrialsClient()
    trial = client._parse_trial(study)

    assert len(trial.mesh_conditions) == 1
    assert trial.mesh_conditions[0].id == "D003924"
    assert trial.mesh_conditions[0].term == "Diabetes Mellitus, Type 2"

    assert len(trial.mesh_ancestors) == 2
    assert trial.mesh_ancestors[0].id == "D003920"
    assert trial.mesh_ancestors[0].term == "Diabetes Mellitus"
    assert trial.mesh_ancestors[1].id == "D044882"
    assert trial.mesh_ancestors[1].term == "Glucose Metabolism Disorders"


def test_parse_trial_missing_ancestors_defaults_empty():
    """_parse_trial returns empty mesh_ancestors when derivedSection lacks ancestors."""
    study = {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT88888888",
                "briefTitle": "No ancestors fixture",
            },
            "statusModule": {"overallStatus": "COMPLETED"},
            "designModule": {"phases": ["PHASE1"]},
            "conditionsModule": {"conditions": ["Condition X"]},
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Sponsor"}},
            "armsInterventionsModule": {
                "interventions": [{"type": "DRUG", "name": "DrugZ"}]
            },
            "outcomesModule": {},
            "referencesModule": {},
            "descriptionModule": {},
        },
    }

    client = ClinicalTrialsClient()
    trial = client._parse_trial(study)

    assert trial.mesh_ancestors == []
