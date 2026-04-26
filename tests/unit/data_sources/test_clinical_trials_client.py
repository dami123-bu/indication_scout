"""Unit tests for ClinicalTrialsClient methods."""

from unittest.mock import AsyncMock, patch

from indication_scout.data_sources.clinical_trials import ClinicalTrialsClient
from indication_scout.models.model_clinical_trials import (
    Intervention,
    Trial,
)


def _make_study(
    nct_id: str,
    why_stopped: str | None = None,
    overall_status: str = "TERMINATED",
    phases: list[str] | None = None,
    intervention_name: str = "Drug A",
) -> dict:
    """Minimal study dict that _parse_trial can handle."""
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct_id, "briefTitle": f"Trial {nct_id}"},
            "statusModule": {
                "overallStatus": overall_status,
                "whyStopped": why_stopped,
            },
            "designModule": {
                "phases": phases if phases is not None else ["PHASE2"],
                "enrollmentInfo": {"count": 100},
            },
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
# _count_trials_total — single-call countTotal
# ------------------------------------------------------------------


async def test_count_trials_total_returns_total_count_with_pagesize_one():
    """_count_trials_total issues one HTTP call with pageSize=1 and returns totalCount."""
    client = ClinicalTrialsClient()
    with patch.object(
        client,
        "_rest_get",
        new=AsyncMock(return_value={"studies": [], "totalCount": 4200}),
    ) as mock_get:
        count = await client._count_trials_total(
            drug="metformin", indication='AREA[ConditionMeshTerm]"Hypertension"'
        )

    assert count == 4200
    assert mock_get.await_count == 1
    args, _ = mock_get.await_args
    params = args[1]
    assert params["pageSize"] == 1
    assert params["countTotal"] == "true"
    assert params["query.intr"] == "metformin"
    assert params["query.cond"] == 'AREA[ConditionMeshTerm]"Hypertension"'


async def test_count_trials_total_zero_when_missing_total():
    """When the API response lacks totalCount, _count_trials_total returns 0."""
    client = ClinicalTrialsClient()
    with patch.object(
        client, "_rest_get", new=AsyncMock(return_value={"studies": []})
    ):
        count = await client._count_trials_total(
            drug="x", indication='AREA[ConditionMeshTerm]"Y"'
        )
    assert count == 0


async def test_count_trials_total_passes_status_and_phase_filters():
    """status_filter goes to filter.overallStatus; phase_filter is appended via AREA[Phase]."""
    client = ClinicalTrialsClient()
    with patch.object(
        client,
        "_rest_get",
        new=AsyncMock(return_value={"totalCount": 7}),
    ) as mock_get:
        count = await client._count_trials_total(
            drug="metformin",
            indication='AREA[ConditionMeshTerm]"Hypertension"',
            status_filter="COMPLETED",
            phase_filter="PHASE3",
        )

    assert count == 7
    args, _ = mock_get.await_args
    params = args[1]
    assert params["filter.overallStatus"] == "COMPLETED"
    assert params["query.term"] == "AREA[Phase]PHASE3"


# ------------------------------------------------------------------
# search_trials — 4 counts + 1 fetch, MeSH server-side
# ------------------------------------------------------------------


async def test_search_trials_assembles_counts_and_fetch():
    """search_trials fans out four count calls plus one fetch and returns a SearchTrialsResult."""
    client = ClinicalTrialsClient()
    fake_trial = Trial(
        nct_id="NCT11111111",
        title="T",
        phase="Phase 2",
        overall_status="RECRUITING",
        sponsor="S",
    )

    with (
        patch.object(
            client,
            "_count_trials_total",
            new=AsyncMock(side_effect=[131, 12, 4, 1]),
        ) as mock_count,
        patch.object(
            client,
            "_paginated_search",
            new=AsyncMock(return_value=([fake_trial], False)),
        ) as mock_fetch,
    ):
        result = await client.search_trials("bupropion", "Depressive Disorder")

    assert result.total_count == 131
    assert result.by_status == {
        "RECRUITING": 12,
        "ACTIVE_NOT_RECRUITING": 4,
        "WITHDRAWN": 1,
    }
    assert len(result.trials) == 1
    assert result.trials[0].nct_id == "NCT11111111"

    # 4 counts: total + recruiting + active + withdrawn
    assert mock_count.await_count == 4
    # 1 fetch with EnrollmentCount:desc sort and the FETCH_MAX cap
    assert mock_fetch.await_count == 1
    fetch_kwargs = mock_fetch.await_args.kwargs
    assert fetch_kwargs["drug"] == "bupropion"
    assert fetch_kwargs["indication"] == 'AREA[ConditionMeshTerm]"Depressive Disorder"'
    assert fetch_kwargs["sort"] == "EnrollmentCount:desc"
    assert fetch_kwargs["max_results"] == 50


async def test_search_trials_uses_same_mesh_cond_for_count_and_fetch():
    """The same AREA[ConditionMeshTerm] filter is applied to every count call and the fetch."""
    client = ClinicalTrialsClient()
    expected_cond = 'AREA[ConditionMeshTerm]"Hypertension"'

    with (
        patch.object(
            client,
            "_count_trials_total",
            new=AsyncMock(return_value=0),
        ) as mock_count,
        patch.object(
            client,
            "_paginated_search",
            new=AsyncMock(return_value=([], False)),
        ) as mock_fetch,
    ):
        await client.search_trials("metformin", "Hypertension")

    for call in mock_count.await_args_list:
        assert call.kwargs["indication"] == expected_cond
    assert mock_fetch.await_args.kwargs["indication"] == expected_cond


# ------------------------------------------------------------------
# get_completed_trials — 2 counts + 1 fetch
# ------------------------------------------------------------------


async def test_get_completed_trials_returns_total_phase3_and_trials():
    """get_completed_trials issues total + Phase 3 count calls and a COMPLETED fetch."""
    client = ClinicalTrialsClient()
    fake_trial = Trial(
        nct_id="NCT22222222",
        title="T",
        phase="Phase 3",
        overall_status="COMPLETED",
        sponsor="S",
    )

    with (
        patch.object(
            client,
            "_count_trials_total",
            new=AsyncMock(side_effect=[20, 5]),
        ) as mock_count,
        patch.object(
            client,
            "_paginated_search",
            new=AsyncMock(return_value=([fake_trial], False)),
        ) as mock_fetch,
    ):
        result = await client.get_completed_trials("semaglutide", "Diabetes Mellitus, Type 2")

    assert result.total_count == 20
    assert result.phase3_count == 5
    assert len(result.trials) == 1
    assert result.trials[0].nct_id == "NCT22222222"

    assert mock_count.await_count == 2
    # First count: completed total. Second count: completed Phase 3.
    first_call_kwargs = mock_count.await_args_list[0].kwargs
    second_call_kwargs = mock_count.await_args_list[1].kwargs
    assert first_call_kwargs["status_filter"] == "COMPLETED"
    assert first_call_kwargs.get("phase_filter") is None
    assert second_call_kwargs["status_filter"] == "COMPLETED"
    assert second_call_kwargs["phase_filter"] == "PHASE3"

    assert mock_fetch.await_count == 1
    fetch_kwargs = mock_fetch.await_args.kwargs
    assert fetch_kwargs["status_filter"] == "COMPLETED"
    assert fetch_kwargs["max_results"] == 50
    assert fetch_kwargs["sort"] == "EnrollmentCount:desc"


# ------------------------------------------------------------------
# get_terminated_trials — 1 count + 1 fetch
# ------------------------------------------------------------------


async def test_get_terminated_trials_returns_total_and_trials():
    """get_terminated_trials issues a TERMINATED count and a TERMINATED fetch."""
    client = ClinicalTrialsClient()
    fake_trial = Trial(
        nct_id="NCT33333333",
        title="T",
        phase="Phase 2",
        overall_status="TERMINATED",
        why_stopped="Lack of efficacy",
        sponsor="S",
    )

    with (
        patch.object(
            client,
            "_count_trials_total",
            new=AsyncMock(return_value=3),
        ) as mock_count,
        patch.object(
            client,
            "_paginated_search",
            new=AsyncMock(return_value=([fake_trial], False)),
        ) as mock_fetch,
    ):
        result = await client.get_terminated_trials("metformin", "Hypertension")

    assert result.total_count == 3
    assert len(result.trials) == 1
    assert result.trials[0].nct_id == "NCT33333333"
    assert result.trials[0].why_stopped == "Lack of efficacy"

    assert mock_count.await_count == 1
    assert mock_count.await_args.kwargs["status_filter"] == "TERMINATED"

    assert mock_fetch.await_count == 1
    fetch_kwargs = mock_fetch.await_args.kwargs
    assert fetch_kwargs["status_filter"] == "TERMINATED"
    assert fetch_kwargs["max_results"] == 50
    assert fetch_kwargs["sort"] == "EnrollmentCount:desc"


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


async def test_get_landscape_calls_fetch_and_count_total():
    """get_landscape calls both _fetch_all_indication_trials and _count_trials_total."""
    fake_trials = [_make_drug_trial("NCT00000001")]

    client = ClinicalTrialsClient()
    with (
        patch.object(
            client,
            "_fetch_all_indication_trials",
            new=AsyncMock(return_value=(fake_trials, False)),
        ) as mock_fetch,
        patch.object(
            client,
            "_count_trials_total",
            new=AsyncMock(return_value=330),
        ) as mock_count,
    ):
        result = await client.get_landscape("Gastroparesis")

    mock_fetch.assert_awaited_once()
    mock_count.assert_awaited_once()
    assert result.total_trial_count == 330
    # Server-side MeSH filter is applied to both fetch and count
    expected_cond = 'AREA[ConditionMeshTerm]"Gastroparesis"'
    assert mock_fetch.await_args.args[0] == expected_cond
    assert mock_count.await_args.kwargs["indication"] == expected_cond
