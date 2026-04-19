"""Unit tests for ClinicalTrialsClient methods."""

from unittest.mock import AsyncMock, patch

from indication_scout.data_sources.clinical_trials import ClinicalTrialsClient
from indication_scout.models.model_clinical_trials import (
    Intervention,
    Trial,
)


def _make_study(
    nct_id: str, why_stopped: str | None, intervention_name: str = "Drug A"
) -> dict:
    """Minimal study dict that _parse_trial can handle."""
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct_id, "briefTitle": f"Trial {nct_id}"},
            "statusModule": {
                "overallStatus": "TERMINATED",
                "whyStopped": why_stopped,
            },
            "designModule": {"phases": ["PHASE2"], "enrollmentInfo": {"count": 100}},
            "conditionsModule": {"conditions": ["Test Indication"]},
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Test Sponsor"}},
            "armsInterventionsModule": {
                "interventions": [{"type": "DRUG", "name": intervention_name}]
            },
            "outcomesModule": {},
            "referencesModule": {},
            "descriptionModule": {},
        }
    }


# ------------------------------------------------------------------
# get_terminated — dual-query shape
# ------------------------------------------------------------------


async def test_get_terminated_drug_query_filters_to_safety_efficacy_only(tmp_path):
    """Drug query results are filtered to stop_category in {safety, efficacy}.

    Business and enrollment terminations from the drug query are dropped as noise.
    """
    drug_studies = [
        _make_study("NCT00000001", "Serious adverse events observed"),  # safety
        _make_study("NCT00000002", "Lack of efficacy in interim analysis"),  # efficacy
        _make_study(
            "NCT00000003", "Strategic decision by sponsor"
        ),  # business — dropped
        _make_study("NCT00000004", "Insufficient enrollment"),  # enrollment — dropped
    ]
    indication_studies = []

    client = ClinicalTrialsClient()
    with patch.object(
        client,
        "_rest_get",
        new=AsyncMock(
            side_effect=[
                {"studies": drug_studies},
                {"studies": indication_studies},
            ]
        ),
    ):
        results = await client.get_terminated("testdrug", "testindication")

    nct_ids = {t.nct_id for t in results}
    assert "NCT00000001" in nct_ids  # safety — kept
    assert "NCT00000002" in nct_ids  # efficacy — kept
    assert "NCT00000003" not in nct_ids  # business — dropped
    assert "NCT00000004" not in nct_ids  # enrollment — dropped


async def test_get_terminated_indication_query_not_filtered(tmp_path):
    """Indication query results are not filtered — all stop categories pass through."""
    drug_studies = []
    indication_studies = [
        _make_study("NCT00000010", "Serious adverse events observed"),  # safety
        _make_study("NCT00000011", "Strategic decision by sponsor"),  # business
        _make_study("NCT00000012", "Insufficient enrollment"),  # enrollment
        _make_study("NCT00000013", None),  # unknown
    ]

    client = ClinicalTrialsClient()
    with patch.object(
        client,
        "_rest_get",
        new=AsyncMock(
            side_effect=[
                {"studies": drug_studies},
                {"studies": indication_studies},
            ]
        ),
    ):
        results = await client.get_terminated("testdrug", "testindication")

    nct_ids = {t.nct_id for t in results}
    assert "NCT00000010" in nct_ids
    assert "NCT00000011" in nct_ids
    assert "NCT00000012" in nct_ids
    assert "NCT00000013" in nct_ids


async def test_get_terminated_deduplicates_by_nct_id(tmp_path):
    """A trial appearing in both drug and indication queries appears only once."""
    shared_study = _make_study("NCT00000020", "Lack of efficacy")

    client = ClinicalTrialsClient()
    with patch.object(
        client,
        "_rest_get",
        new=AsyncMock(
            side_effect=[
                {"studies": [shared_study]},  # drug query
                {"studies": [shared_study]},  # indication query
            ]
        ),
    ):
        results = await client.get_terminated("testdrug", "testindication")

    assert len([t for t in results if t.nct_id == "NCT00000020"]) == 1


async def test_get_terminated_max_results_caps_indication_not_drug(tmp_path):
    """Settings cap limits indication results; drug safety/efficacy results are not capped."""
    drug_studies = [
        _make_study(f"NCT9000{i:04d}", "Serious adverse events observed")
        for i in range(5)
    ]
    indication_studies = [
        _make_study(f"NCT8000{i:04d}", "Strategic decision by sponsor")
        for i in range(10)
    ]

    client = ClinicalTrialsClient()
    with (
        patch.object(
            client,
            "_rest_get",
            new=AsyncMock(
                side_effect=[
                    {"studies": drug_studies},
                    {"studies": indication_studies},
                ]
            ),
        ),
        patch(
            "indication_scout.data_sources.clinical_trials._settings"
        ) as mock_settings,
    ):
        mock_settings.clinical_trials_terminated_drug_page_size = 50
        mock_settings.clinical_trials_terminated_indication_max = 3
        results = await client.get_terminated("testdrug", "testindication")

    drug_ids = {t.nct_id for t in results if t.nct_id.startswith("NCT9")}
    indication_ids = {t.nct_id for t in results if t.nct_id.startswith("NCT8")}

    # All 5 drug safety results included
    assert len(drug_ids) == 5
    # Indication results capped at settings value (3)
    assert len(indication_ids) == 3


# ------------------------------------------------------------------
# _aggregate_landscape — true total count
# ------------------------------------------------------------------


def _make_drug_trial(
    nct_id: str, phase: str = "Phase 2", enrollment: int = 100
) -> Trial:
    """Minimal Trial with a Drug intervention for landscape aggregation."""
    return Trial(
        nct_id=nct_id,
        title=f"Trial {nct_id}",
        phase=phase,
        overall_status="COMPLETED",
        interventions=[
            Intervention(intervention_type="Drug", intervention_name="DrugA")
        ],
        sponsor="Sponsor A",
        enrollment=enrollment,
    )


def test_aggregate_landscape_uses_passed_total_count():
    """_aggregate_landscape stores passed total_count, not len(trials).

    50 trials fetched but the API reported 330 total → total_trial_count == 330.
    """
    trials = [_make_drug_trial(f"NCT{i:08d}") for i in range(50)]
    client = ClinicalTrialsClient()

    result = client._aggregate_landscape(trials, total_count=330, top_n=10)

    assert result.total_trial_count == 330


async def test_get_landscape_calls_fetch_and_count():
    """get_landscape calls both _fetch_all_indication_trials and _count_trials."""
    fake_trials = [_make_drug_trial("NCT00000001")]

    client = ClinicalTrialsClient()
    with (
        patch.object(
            client,
            "_fetch_all_indication_trials",
            new=AsyncMock(return_value=fake_trials),
        ) as mock_fetch,
        patch.object(
            client,
            "_count_trials",
            new=AsyncMock(return_value=330),
        ) as mock_count,
    ):
        result = await client.get_landscape("gastroparesis")

    mock_fetch.assert_awaited_once()
    mock_count.assert_awaited_once()
    assert result.total_trial_count == 330


# ------------------------------------------------------------------
# _parse_terminated_trial — field population
# ------------------------------------------------------------------


def test_parse_terminated_trial_field_population():
    """_parse_terminated_trial populates title, enrollment, sponsor, start_date, termination_date from Trial."""
    study = {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT99887766",
                "briefTitle": "A Phase 2 Study of DrugX in Lupus",
            },
            "statusModule": {
                "overallStatus": "TERMINATED",
                "whyStopped": "Lack of efficacy in interim analysis",
                "startDateStruct": {"date": "2019-06"},
                "primaryCompletionDateStruct": {"date": "2021-11"},
            },
            "designModule": {
                "phases": ["PHASE2"],
                "enrollmentInfo": {"count": 245},
            },
            "conditionsModule": {"conditions": ["Systemic Lupus Erythematosus"]},
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "BioPharm Inc"}},
            "armsInterventionsModule": {
                "interventions": [{"type": "DRUG", "name": "DrugX"}]
            },
            "outcomesModule": {},
            "referencesModule": {},
            "descriptionModule": {},
        }
    }

    client = ClinicalTrialsClient()
    result = client._parse_terminated_trial(study)

    assert result.nct_id == "NCT99887766"
    assert result.title == "A Phase 2 Study of DrugX in Lupus"
    assert result.enrollment == 245
    assert result.sponsor == "BioPharm Inc"
    assert result.start_date == "2019-06"
    assert result.termination_date == "2021-11"
    assert result.drug_name == "DrugX"
    assert result.indication == "Systemic Lupus Erythematosus"
    assert result.phase == "Phase 2"
    assert result.why_stopped == "Lack of efficacy in interim analysis"
    assert result.stop_category == "efficacy"


# ------------------------------------------------------------------
# _count_trials — post-filter counting (Phase 3)
# ------------------------------------------------------------------


def _make_study_with_mesh(
    nct_id: str, mesh_ids: list[str], ancestor_ids: list[str] | None = None
) -> dict:
    """Minimal study dict with MeSH tags for filter-count testing."""
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct_id, "briefTitle": f"Trial {nct_id}"},
            "statusModule": {"overallStatus": "COMPLETED"},
            "designModule": {"phases": ["PHASE2"]},
            "conditionsModule": {"conditions": ["X"]},
            "sponsorCollaboratorsModule": {"leadSponsor": {"name": "S"}},
            "armsInterventionsModule": {
                "interventions": [{"type": "DRUG", "name": "Drug A"}]
            },
            "outcomesModule": {},
            "referencesModule": {},
            "descriptionModule": {},
        },
        "derivedSection": {
            "conditionBrowseModule": {
                "meshes": [{"id": mid, "term": mid} for mid in mesh_ids],
                "ancestors": [
                    {"id": aid, "term": aid} for aid in (ancestor_ids or [])
                ],
            }
        },
    }


async def test_count_trials_without_mesh_returns_total_count():
    """Without target_mesh_id, _count_trials returns the API totalCount verbatim."""
    client = ClinicalTrialsClient()
    with patch.object(
        client,
        "_rest_get",
        new=AsyncMock(return_value={"studies": [], "totalCount": 4200}),
    ) as mock_get:
        count = await client._count_trials(drug=None, indication="hypertension")

    assert count == 4200
    # Single request with pageSize=1 — not paginated
    assert mock_get.await_count == 1
    args, _ = mock_get.await_args
    assert args[1]["pageSize"] == 1


async def test_count_trials_with_mesh_returns_post_filter_count():
    """With target_mesh_id, _count_trials paginates, filters, and returns filtered count."""
    client = ClinicalTrialsClient()
    studies = [
        _make_study_with_mesh("NCT00000001", ["D006973"]),  # keep — exact
        _make_study_with_mesh("NCT00000002", ["D059468"], ["D006973"]),  # keep — anc
        _make_study_with_mesh("NCT00000003", ["D005901"], ["D005128"]),  # drop — unrelated
        _make_study_with_mesh("NCT00000004", [], []),  # drop — empty mesh
    ]
    with patch.object(
        client,
        "_rest_get",
        new=AsyncMock(return_value={"studies": studies, "totalCount": 9999}),
    ):
        count = await client._count_trials(
            drug=None, indication="hypertension", target_mesh_id="D006973"
        )

    # totalCount (9999) is ignored; only the 2 trials matching D006973 are counted
    assert count == 2


async def test_count_trials_with_mesh_all_dropped_returns_zero():
    """With target_mesh_id, when no trial's mesh matches, count is 0 even if totalCount is nonzero."""
    client = ClinicalTrialsClient()
    studies = [
        _make_study_with_mesh("NCT00000010", ["D005901"]),
        _make_study_with_mesh("NCT00000011", ["D059468"], ["D005128"]),
    ]
    with patch.object(
        client,
        "_rest_get",
        new=AsyncMock(return_value={"studies": studies, "totalCount": 50}),
    ):
        count = await client._count_trials(
            drug=None, indication="hypertension", target_mesh_id="D006973"
        )

    assert count == 0
