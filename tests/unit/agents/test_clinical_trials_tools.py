"""Unit tests for clinical_trials_tools.build_clinical_trials_tools."""

import logging
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import ToolCall as LCToolCall

from indication_scout.agents.clinical_trials.clinical_trials_tools import (
    build_clinical_trials_tools,
)
from indication_scout.data_sources.base_client import DataSourceError
from indication_scout.models.model_clinical_trials import (
    ApprovalCheck,
    CompetitorEntry,
    CompletedTrialsResult,
    IndicationLandscape,
    Intervention,
    PrimaryOutcome,
    RecentStart,
    SearchTrialsResult,
    TerminatedTrialsResult,
    Trial,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _mock_client(**method_returns) -> AsyncMock:
    """Build an AsyncMock that works as an async context manager."""
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    for name, return_value in method_returns.items():
        setattr(client, name, AsyncMock(return_value=return_value))
    return client


def _get_tool(tools: list, name: str):
    for t in tools:
        if t.name == name:
            return t
    raise ValueError(f"Tool '{name}' not found")


# ------------------------------------------------------------------
# search_trials
# ------------------------------------------------------------------


async def test_search_trials_returns_search_trials_result_artifact():
    """search_trials returns a SearchTrialsResult artifact with all fields intact."""
    trial = Trial(
        nct_id="NCT00127933",
        title="XeNA Study",
        phase="Phase 4",
        overall_status="COMPLETED",
        why_stopped=None,
        indications=["Breast Cancer"],
        interventions=[
            Intervention(intervention_type="Drug", intervention_name="Trastuzumab"),
            Intervention(intervention_type="Drug", intervention_name="Capecitabine"),
        ],
        sponsor="Hoffmann-La Roche",
        enrollment=157,
        start_date="2005-08",
        completion_date="2009-04",
        primary_outcomes=[PrimaryOutcome(measure="pCR rate", time_frame="4 cycles")],
        references=[],
    )
    mock_result = SearchTrialsResult(
        total_count=1,
        by_status={"RECRUITING": 0, "ACTIVE_NOT_RECRUITING": 0, "WITHDRAWN": 0},
        trials=[trial],
    )

    mock_client = _mock_client(search_trials=mock_result)
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=("D001943", "Breast Neoplasms")),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools._resolve_drug_aliases",
            new=AsyncMock(return_value=["trastuzumab", "herceptin"]),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            return_value=mock_client,
        ),
    ):
        msg = await _get_tool(tools, "search_trials").ainvoke(
            LCToolCall(
                name="search_trials",
                args={"drug": "trastuzumab", "indication": "breast cancer"},
                id="tc4",
                type="tool_call",
            )
        )

    mock_client.search_trials.assert_awaited_once_with(
        "trastuzumab",
        "Breast Neoplasms",
        date_before=None,
    )
    assert isinstance(msg.artifact, SearchTrialsResult)
    assert msg.artifact.total_count == 1
    assert msg.artifact.by_status == {
        "RECRUITING": 0,
        "ACTIVE_NOT_RECRUITING": 0,
        "WITHDRAWN": 0,
    }
    assert len(msg.artifact.trials) == 1
    t = msg.artifact.trials[0]
    assert t.nct_id == "NCT00127933"
    assert t.title == "XeNA Study"
    assert t.phase == "Phase 4"
    assert t.overall_status == "COMPLETED"
    assert t.why_stopped is None
    assert t.indications == ["Breast Cancer"]
    assert t.interventions[0].intervention_type == "Drug"
    assert t.interventions[0].intervention_name == "Trastuzumab"
    assert t.interventions[1].intervention_type == "Drug"
    assert t.interventions[1].intervention_name == "Capecitabine"
    assert t.sponsor == "Hoffmann-La Roche"
    assert t.enrollment == 157
    assert t.start_date == "2005-08"
    assert t.completion_date == "2009-04"
    assert t.primary_outcomes[0].measure == "pCR rate"
    assert t.primary_outcomes[0].time_frame == "4 cycles"
    assert t.references == []
    assert "1 trials" in msg.content


async def test_search_trials_passes_date_before():
    """search_trials forwards date_before from closure into the client call."""
    cutoff = date(2018, 1, 1)
    mock_client = _mock_client(search_trials=SearchTrialsResult())
    tools = build_clinical_trials_tools(date_before=cutoff)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=("D000505", "Alopecia Areata")),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools._resolve_drug_aliases",
            new=AsyncMock(return_value=["tofacitinib"]),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            return_value=mock_client,
        ),
    ):
        await _get_tool(tools, "search_trials").ainvoke(
            LCToolCall(
                name="search_trials",
                args={"drug": "tofacitinib", "indication": "alopecia areata"},
                id="tc5",
                type="tool_call",
            )
        )

    mock_client.search_trials.assert_awaited_once_with(
        "tofacitinib",
        "Alopecia Areata",
        date_before=cutoff,
    )


async def test_search_trials_returns_empty_when_resolver_returns_none():
    """search_trials returns a default SearchTrialsResult and skips the client when resolver is None."""
    client_factory = MagicMock()
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            new=client_factory,
        ),
    ):
        msg = await _get_tool(tools, "search_trials").ainvoke(
            LCToolCall(
                name="search_trials",
                args={"drug": "metformin", "indication": "xyzzy_bogus"},
                id="tc_ms2",
                type="tool_call",
            )
        )

    assert isinstance(msg.artifact, SearchTrialsResult)
    assert msg.artifact.total_count == 0
    assert msg.artifact.by_status == {}
    assert msg.artifact.trials == []
    assert "MeSH unresolved" in msg.content
    client_factory.assert_not_called()


async def test_search_trials_content_notes_top_50_when_total_exceeds_shown():
    """When total_count exceeds len(trials), the content string flags the 50-cap."""
    trial = Trial(
        nct_id="NCT00000001",
        title="T",
        phase="Phase 2",
        overall_status="RECRUITING",
        sponsor="S",
        interventions=[
            Intervention(intervention_type="Drug", intervention_name="Bupropion"),
        ],
    )
    mock_result = SearchTrialsResult(
        total_count=131,
        by_status={"RECRUITING": 12, "ACTIVE_NOT_RECRUITING": 4, "WITHDRAWN": 1},
        trials=[trial],
    )
    mock_client = _mock_client(search_trials=mock_result)
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=("D003866", "Depressive Disorder")),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools._resolve_drug_aliases",
            new=AsyncMock(return_value=["bupropion", "wellbutrin"]),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            return_value=mock_client,
        ),
    ):
        msg = await _get_tool(tools, "search_trials").ainvoke(
            LCToolCall(
                name="search_trials",
                args={"drug": "bupropion", "indication": "depression"},
                id="tc_search_cap",
                type="tool_call",
            )
        )

    assert "131 trials" in msg.content
    assert "recruiting=12" in msg.content
    assert "active=4" in msg.content
    assert "withdrawn=1" in msg.content
    assert "top 50 shown" in msg.content


# ------------------------------------------------------------------
# get_completed
# ------------------------------------------------------------------


async def test_get_completed_returns_completed_trials_result_artifact():
    """get_completed returns a CompletedTrialsResult artifact and reports phase3 count in content."""
    trial = Trial(
        nct_id="NCT04111111",
        title="Phase 3 Study",
        phase="Phase 3",
        overall_status="COMPLETED",
        sponsor="Sponsor",
        enrollment=500,
        interventions=[
            Intervention(intervention_type="Drug", intervention_name="Semaglutide"),
        ],
    )
    mock_result = CompletedTrialsResult(total_count=12, phase3_count=3, trials=[trial])
    mock_client = _mock_client(get_completed_trials=mock_result)
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=("D003924", "Diabetes Mellitus, Type 2")),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools._resolve_drug_aliases",
            new=AsyncMock(return_value=["semaglutide", "ozempic"]),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            return_value=mock_client,
        ),
    ):
        msg = await _get_tool(tools, "get_completed").ainvoke(
            LCToolCall(
                name="get_completed",
                args={"drug": "semaglutide", "indication": "type 2 diabetes"},
                id="tc_completed",
                type="tool_call",
            )
        )

    mock_client.get_completed_trials.assert_awaited_once_with(
        "semaglutide",
        "Diabetes Mellitus, Type 2",
        date_before=None,
    )
    assert isinstance(msg.artifact, CompletedTrialsResult)
    assert msg.artifact.total_count == 12
    assert msg.artifact.phase3_count == 3
    assert len(msg.artifact.trials) == 1
    assert msg.artifact.trials[0].nct_id == "NCT04111111"
    assert "12 total" in msg.content
    assert "3 Phase 3" in msg.content
    # only 1 shown of 12 → cap note present
    assert "top 50 shown" in msg.content


async def test_get_completed_passes_date_before():
    """get_completed forwards date_before from closure into the client call."""
    cutoff = date(2020, 1, 1)
    mock_client = _mock_client(get_completed_trials=CompletedTrialsResult())
    tools = build_clinical_trials_tools(date_before=cutoff)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=("D000001", "Test Term")),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools._resolve_drug_aliases",
            new=AsyncMock(return_value=["drug_x"]),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            return_value=mock_client,
        ),
    ):
        await _get_tool(tools, "get_completed").ainvoke(
            LCToolCall(
                name="get_completed",
                args={"drug": "drug_x", "indication": "indication_y"},
                id="tc_completed_date",
                type="tool_call",
            )
        )

    mock_client.get_completed_trials.assert_awaited_once_with(
        "drug_x",
        "Test Term",
        date_before=cutoff,
    )


async def test_get_completed_returns_empty_when_resolver_returns_none():
    """get_completed returns a default CompletedTrialsResult and skips client when resolver is None."""
    client_factory = MagicMock()
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            new=client_factory,
        ),
    ):
        msg = await _get_tool(tools, "get_completed").ainvoke(
            LCToolCall(
                name="get_completed",
                args={"drug": "metformin", "indication": "xyzzy_bogus"},
                id="tc_completed_none",
                type="tool_call",
            )
        )

    assert isinstance(msg.artifact, CompletedTrialsResult)
    assert msg.artifact.total_count == 0
    assert msg.artifact.phase3_count == 0
    assert msg.artifact.trials == []
    assert "MeSH unresolved" in msg.content
    client_factory.assert_not_called()


# ------------------------------------------------------------------
# get_terminated
# ------------------------------------------------------------------


async def test_get_terminated_returns_terminated_trials_result_artifact():
    """get_terminated returns TerminatedTrialsResult and counts safety/efficacy stops in content."""
    safety_trial = Trial(
        nct_id="NCT04012255",
        title="Safety Stop Trial",
        phase="Phase 2",
        overall_status="TERMINATED",
        why_stopped="Serious adverse events observed",
        sponsor="Novo Nordisk",
        enrollment=40,
        interventions=[
            Intervention(intervention_type="Drug", intervention_name="Semaglutide"),
        ],
    )
    business_trial = Trial(
        nct_id="NCT04012256",
        title="Business Stop Trial",
        phase="Phase 1",
        overall_status="TERMINATED",
        why_stopped="The trial was terminated for strategic reasons.",
        sponsor="Novo Nordisk",
        enrollment=20,
        interventions=[
            Intervention(intervention_type="Drug", intervention_name="Semaglutide"),
        ],
    )
    mock_result = TerminatedTrialsResult(
        total_count=2, trials=[safety_trial, business_trial]
    )
    mock_client = _mock_client(get_terminated_trials=mock_result)
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=("D050177", "Overweight")),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools._resolve_drug_aliases",
            new=AsyncMock(return_value=["semaglutide", "ozempic"]),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            return_value=mock_client,
        ),
    ):
        msg = await _get_tool(tools, "get_terminated").ainvoke(
            LCToolCall(
                name="get_terminated",
                args={"drug": "semaglutide", "indication": "overweight"},
                id="tc_term",
                type="tool_call",
            )
        )

    mock_client.get_terminated_trials.assert_awaited_once_with(
        "semaglutide",
        "Overweight",
        date_before=None,
    )
    assert isinstance(msg.artifact, TerminatedTrialsResult)
    assert msg.artifact.total_count == 2
    assert len(msg.artifact.trials) == 2
    assert msg.artifact.trials[0].nct_id == "NCT04012255"
    assert msg.artifact.trials[0].why_stopped == "Serious adverse events observed"
    assert msg.artifact.trials[1].nct_id == "NCT04012256"
    # 1 safety stop, 1 business → "1 safety/efficacy"
    assert "2 total" in msg.content
    assert "1 safety/efficacy" in msg.content


async def test_get_terminated_content_notes_cap_when_total_exceeds_shown():
    """When total_count exceeds shown trials, the content string flags the 50-cap."""
    trial = Trial(
        nct_id="NCT04012255",
        title="Trial",
        phase="Phase 2",
        overall_status="TERMINATED",
        why_stopped="Lack of efficacy",
        sponsor="S",
        interventions=[
            Intervention(intervention_type="Drug", intervention_name="Drug_x"),
        ],
    )
    mock_result = TerminatedTrialsResult(total_count=80, trials=[trial])
    mock_client = _mock_client(get_terminated_trials=mock_result)
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=("D050177", "Overweight")),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools._resolve_drug_aliases",
            new=AsyncMock(return_value=["drug_x"]),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            return_value=mock_client,
        ),
    ):
        msg = await _get_tool(tools, "get_terminated").ainvoke(
            LCToolCall(
                name="get_terminated",
                args={"drug": "drug_x", "indication": "indication_y"},
                id="tc_term_cap",
                type="tool_call",
            )
        )

    assert "80 total" in msg.content
    assert "top 50 shown" in msg.content
    assert "stop-category counts cover the 1 shown only" in msg.content


async def test_get_terminated_passes_date_before():
    """get_terminated forwards date_before from closure into the client call."""
    cutoff = date(2018, 1, 1)
    mock_client = _mock_client(get_terminated_trials=TerminatedTrialsResult())
    tools = build_clinical_trials_tools(date_before=cutoff)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=("D000001", "Test Term")),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools._resolve_drug_aliases",
            new=AsyncMock(return_value=["drug_x"]),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            return_value=mock_client,
        ),
    ):
        await _get_tool(tools, "get_terminated").ainvoke(
            LCToolCall(
                name="get_terminated",
                args={"drug": "drug_x", "indication": "indication_y"},
                id="tc9",
                type="tool_call",
            )
        )

    mock_client.get_terminated_trials.assert_awaited_once_with(
        "drug_x",
        "Test Term",
        date_before=cutoff,
    )


async def test_get_terminated_returns_empty_when_resolver_returns_none():
    """get_terminated returns a default TerminatedTrialsResult and skips client when resolver is None."""
    client_factory = MagicMock()
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            new=client_factory,
        ),
    ):
        msg = await _get_tool(tools, "get_terminated").ainvoke(
            LCToolCall(
                name="get_terminated",
                args={"drug": "metformin", "indication": "xyzzy_bogus"},
                id="tc_ms8",
                type="tool_call",
            )
        )

    assert isinstance(msg.artifact, TerminatedTrialsResult)
    assert msg.artifact.total_count == 0
    assert msg.artifact.trials == []
    assert "MeSH unresolved" in msg.content
    client_factory.assert_not_called()


# ------------------------------------------------------------------
# Intervention-name filter (Essie noise mitigation)
# ------------------------------------------------------------------


async def test_search_trials_drops_trial_where_drug_only_in_eligibility():
    """A trial returning from CT.gov whose drug appears only in eligibility/exclusion
    text (not in the parsed interventions list as a Drug/Biological) is dropped from
    the artifact, the total_count is decremented, and the content string flags the drop.
    Reproduces the dasatinib × GIST NCT00688766 (IPI-504 / retaspimycin) false positive.
    """
    real_trial = Trial(
        nct_id="NCT00000111",
        title="Real dasatinib GIST trial",
        phase="Phase 2",
        overall_status="COMPLETED",
        sponsor="S",
        interventions=[
            Intervention(intervention_type="Drug", intervention_name="Dasatinib"),
        ],
    )
    noise_trial = Trial(
        nct_id="NCT00688766",
        title="IPI-504 in GIST after imatinib/sunitinib failure",
        phase="Phase 3",
        overall_status="TERMINATED",
        sponsor="Infinity",
        interventions=[
            Intervention(
                intervention_type="Drug",
                intervention_name="retaspimycin hydrochloride (IPI-504)",
            ),
            Intervention(intervention_type="Drug", intervention_name="placebo"),
        ],
    )
    mock_result = SearchTrialsResult(
        total_count=2,
        by_status={"RECRUITING": 0, "ACTIVE_NOT_RECRUITING": 0, "WITHDRAWN": 0},
        trials=[real_trial, noise_trial],
    )
    mock_client = _mock_client(search_trials=mock_result)
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=("D046152", "Gastrointestinal Stromal Tumors")),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools._resolve_drug_aliases",
            new=AsyncMock(return_value=["dasatinib", "sprycel", "bms-354825"]),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            return_value=mock_client,
        ),
    ):
        msg = await _get_tool(tools, "search_trials").ainvoke(
            LCToolCall(
                name="search_trials",
                args={"drug": "dasatinib", "indication": "gist"},
                id="tc_filter_search",
                type="tool_call",
            )
        )

    assert isinstance(msg.artifact, SearchTrialsResult)
    assert msg.artifact.total_count == 1
    assert len(msg.artifact.trials) == 1
    assert msg.artifact.trials[0].nct_id == "NCT00000111"
    assert "1 trials" in msg.content
    assert "dropped 1 non-intervention trials" in msg.content


async def test_search_trials_drops_observational_trial_with_no_interventions():
    """An observational adherence study (no Drug/Biological interventions) returned
    by CT.gov because the drug name appears in the description is dropped.
    Reproduces dasatinib × GIST NCT03880617 (CML/GIST adherence study).
    """
    noise_trial = Trial(
        nct_id="NCT03880617",
        title="Distress, Medication Adherence and Care Needs in CML and GIST",
        phase="Not Applicable",
        overall_status="COMPLETED",
        sponsor="Taiwan University",
        interventions=[],
    )
    mock_result = SearchTrialsResult(
        total_count=1,
        by_status={"RECRUITING": 0, "ACTIVE_NOT_RECRUITING": 0, "WITHDRAWN": 0},
        trials=[noise_trial],
    )
    mock_client = _mock_client(search_trials=mock_result)
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=("D046152", "Gastrointestinal Stromal Tumors")),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools._resolve_drug_aliases",
            new=AsyncMock(return_value=["dasatinib", "sprycel"]),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            return_value=mock_client,
        ),
    ):
        msg = await _get_tool(tools, "search_trials").ainvoke(
            LCToolCall(
                name="search_trials",
                args={"drug": "dasatinib", "indication": "gist"},
                id="tc_filter_obs",
                type="tool_call",
            )
        )

    assert msg.artifact.total_count == 0
    assert msg.artifact.trials == []
    assert "0 trials" in msg.content
    assert "dropped 1 non-intervention trials" in msg.content


async def test_search_trials_keeps_trial_matched_by_trade_name_alias():
    """A trial whose intervention is recorded under a trade name (e.g. Sprycel
    for dasatinib) is kept when the alias list includes that trade name."""
    trial = Trial(
        nct_id="NCT00000222",
        title="Sprycel study",
        phase="Phase 2",
        overall_status="RECRUITING",
        sponsor="BMS",
        interventions=[
            Intervention(intervention_type="Drug", intervention_name="Sprycel 100 mg"),
        ],
    )
    mock_result = SearchTrialsResult(
        total_count=1,
        by_status={"RECRUITING": 1, "ACTIVE_NOT_RECRUITING": 0, "WITHDRAWN": 0},
        trials=[trial],
    )
    mock_client = _mock_client(search_trials=mock_result)
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=("D046152", "Gastrointestinal Stromal Tumors")),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools._resolve_drug_aliases",
            new=AsyncMock(return_value=["dasatinib", "sprycel", "bms-354825"]),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            return_value=mock_client,
        ),
    ):
        msg = await _get_tool(tools, "search_trials").ainvoke(
            LCToolCall(
                name="search_trials",
                args={"drug": "dasatinib", "indication": "gist"},
                id="tc_filter_trade",
                type="tool_call",
            )
        )

    assert msg.artifact.total_count == 1
    assert len(msg.artifact.trials) == 1
    assert msg.artifact.trials[0].nct_id == "NCT00000222"
    assert "dropped" not in msg.content


async def test_search_trials_skips_filter_when_alias_resolution_fails():
    """When _resolve_drug_aliases returns None (drug not in ChEMBL), the filter
    is skipped and trials are returned unfiltered. Better unfiltered than empty."""
    trial = Trial(
        nct_id="NCT00000333",
        title="Some trial",
        phase="Phase 2",
        overall_status="RECRUITING",
        sponsor="S",
        interventions=[
            Intervention(intervention_type="Drug", intervention_name="Some Other Drug"),
        ],
    )
    mock_result = SearchTrialsResult(
        total_count=1,
        by_status={"RECRUITING": 1, "ACTIVE_NOT_RECRUITING": 0, "WITHDRAWN": 0},
        trials=[trial],
    )
    mock_client = _mock_client(search_trials=mock_result)
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=("D000001", "Test Term")),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools._resolve_drug_aliases",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            return_value=mock_client,
        ),
    ):
        msg = await _get_tool(tools, "search_trials").ainvoke(
            LCToolCall(
                name="search_trials",
                args={"drug": "obscuredrug", "indication": "test"},
                id="tc_filter_skip",
                type="tool_call",
            )
        )

    assert msg.artifact.total_count == 1
    assert len(msg.artifact.trials) == 1
    assert "dropped" not in msg.content


async def test_get_terminated_drops_eligibility_only_trial_and_recomputes_safety():
    """Filter is applied to get_terminated: dropped trials are removed from the
    safety/efficacy stop count and from total_count."""
    real_trial = Trial(
        nct_id="NCT00000444",
        title="Real dasatinib termination",
        phase="Phase 2",
        overall_status="TERMINATED",
        why_stopped="Lack of efficacy",
        sponsor="S",
        interventions=[
            Intervention(intervention_type="Drug", intervention_name="Dasatinib"),
        ],
    )
    noise_trial = Trial(
        nct_id="NCT00688766",
        title="IPI-504 in GIST",
        phase="Phase 3",
        overall_status="TERMINATED",
        why_stopped="Sponsor decision",
        sponsor="Infinity",
        interventions=[
            Intervention(
                intervention_type="Drug",
                intervention_name="retaspimycin hydrochloride (IPI-504)",
            ),
        ],
    )
    mock_result = TerminatedTrialsResult(
        total_count=2, trials=[real_trial, noise_trial]
    )
    mock_client = _mock_client(get_terminated_trials=mock_result)
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=("D046152", "Gastrointestinal Stromal Tumors")),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools._resolve_drug_aliases",
            new=AsyncMock(return_value=["dasatinib", "sprycel"]),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            return_value=mock_client,
        ),
    ):
        msg = await _get_tool(tools, "get_terminated").ainvoke(
            LCToolCall(
                name="get_terminated",
                args={"drug": "dasatinib", "indication": "gist"},
                id="tc_filter_term",
                type="tool_call",
            )
        )

    assert msg.artifact.total_count == 1
    assert len(msg.artifact.trials) == 1
    assert msg.artifact.trials[0].nct_id == "NCT00000444"
    assert "1 total" in msg.content
    assert "1 safety/efficacy" in msg.content
    assert "dropped 1 non-intervention trials" in msg.content


async def test_get_completed_drops_eligibility_only_trial():
    """Filter is applied to get_completed: dropped trial is removed and total_count decremented.
    phase3_count is unfiltered (separate count call) and may overstate after filtering.
    """
    real_trial = Trial(
        nct_id="NCT00000555",
        title="Real dasatinib completion",
        phase="Phase 3",
        overall_status="COMPLETED",
        sponsor="BMS",
        interventions=[
            Intervention(intervention_type="Drug", intervention_name="Dasatinib"),
        ],
    )
    noise_trial = Trial(
        nct_id="NCT03880617",
        title="Adherence study mentioning dasatinib",
        phase="Not Applicable",
        overall_status="COMPLETED",
        sponsor="Taiwan University",
        interventions=[],
    )
    mock_result = CompletedTrialsResult(
        total_count=2, phase3_count=1, trials=[real_trial, noise_trial]
    )
    mock_client = _mock_client(get_completed_trials=mock_result)
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=("D046152", "Gastrointestinal Stromal Tumors")),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools._resolve_drug_aliases",
            new=AsyncMock(return_value=["dasatinib", "sprycel"]),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            return_value=mock_client,
        ),
    ):
        msg = await _get_tool(tools, "get_completed").ainvoke(
            LCToolCall(
                name="get_completed",
                args={"drug": "dasatinib", "indication": "gist"},
                id="tc_filter_completed",
                type="tool_call",
            )
        )

    assert msg.artifact.total_count == 1
    assert msg.artifact.phase3_count == 1
    assert len(msg.artifact.trials) == 1
    assert msg.artifact.trials[0].nct_id == "NCT00000555"
    assert "1 total" in msg.content
    assert "1 Phase 3" in msg.content
    assert "dropped 1 non-intervention trials" in msg.content


# ------------------------------------------------------------------
# _trial_intervenes_with_drug (alias matching unit)
# ------------------------------------------------------------------


def test_trial_intervenes_with_drug_matches_whole_word_substring():
    """Whole-word substring match: aliases bounded by non-alphanumeric chars or string edges."""
    from indication_scout.agents.clinical_trials.clinical_trials_tools import (
        _trial_intervenes_with_drug,
    )

    trial = Trial(
        nct_id="NCT1",
        interventions=[
            Intervention(
                intervention_type="Drug", intervention_name="Dasatinib 100 mg"
            ),
        ],
    )
    assert _trial_intervenes_with_drug(trial, ["dasatinib"]) is True


def test_trial_intervenes_with_drug_rejects_eligibility_only_drug():
    """A trial with no intervention whose name contains the alias is rejected."""
    from indication_scout.agents.clinical_trials.clinical_trials_tools import (
        _trial_intervenes_with_drug,
    )

    trial = Trial(
        nct_id="NCT1",
        interventions=[
            Intervention(
                intervention_type="Drug",
                intervention_name="retaspimycin hydrochloride (IPI-504)",
            ),
        ],
    )
    assert _trial_intervenes_with_drug(trial, ["dasatinib", "sprycel"]) is False


def test_trial_intervenes_with_drug_rejects_partial_substring():
    """Short aliases must match as whole words — 'asa' must not match 'asacol'."""
    from indication_scout.agents.clinical_trials.clinical_trials_tools import (
        _trial_intervenes_with_drug,
    )

    trial = Trial(
        nct_id="NCT1",
        interventions=[
            Intervention(intervention_type="Drug", intervention_name="Asacol"),
        ],
    )
    assert _trial_intervenes_with_drug(trial, ["asa"]) is False


def test_trial_intervenes_with_drug_rejects_non_drug_intervention_type():
    """A behavioral/device intervention whose name happens to mention the drug is not a match."""
    from indication_scout.agents.clinical_trials.clinical_trials_tools import (
        _trial_intervenes_with_drug,
    )

    trial = Trial(
        nct_id="NCT1",
        interventions=[
            Intervention(
                intervention_type="Behavioral",
                intervention_name="Counseling on dasatinib adherence",
            ),
        ],
    )
    assert _trial_intervenes_with_drug(trial, ["dasatinib"]) is False


def test_trial_intervenes_with_drug_matches_biological_type():
    """Biological intervention type (e.g. monoclonal antibody) is treated equivalently to Drug."""
    from indication_scout.agents.clinical_trials.clinical_trials_tools import (
        _trial_intervenes_with_drug,
    )

    trial = Trial(
        nct_id="NCT1",
        interventions=[
            Intervention(
                intervention_type="Biological",
                intervention_name="Trastuzumab",
            ),
        ],
    )
    assert _trial_intervenes_with_drug(trial, ["trastuzumab", "herceptin"]) is True


# ------------------------------------------------------------------
# get_landscape
# ------------------------------------------------------------------


async def test_get_landscape_returns_landscape_artifact():
    """get_landscape returns IndicationLandscape artifact with all fields intact."""
    mock_result = IndicationLandscape(
        total_trial_count=95,
        competitors=[
            CompetitorEntry(
                sponsor="Chinese University of Hong Kong",
                drug_name="Esomeprazole",
                drug_type="Drug",
                max_phase="Phase 4",
                trial_count=1,
                statuses={"COMPLETED"},
                total_enrollment=155,
            ),
        ],
        phase_distribution={"Phase 2": 40, "Phase 3": 15, "Phase 4": 10},
        recent_starts=[
            RecentStart(
                nct_id="NCT06836557",
                sponsor="Vanda Pharmaceuticals",
                drug="Tradipitant",
                phase="Phase 3",
            ),
        ],
    )

    mock_client = _mock_client(get_landscape=mock_result)
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=("D018589", "Gastroparesis")),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            return_value=mock_client,
        ),
    ):
        msg = await _get_tool(tools, "get_landscape").ainvoke(
            LCToolCall(
                name="get_landscape",
                args={"indication": "gastroparesis"},
                id="tc6",
                type="tool_call",
            )
        )

    mock_client.get_landscape.assert_awaited_once_with(
        "Gastroparesis",
        date_before=None,
        top_n=10,
    )
    assert isinstance(msg.artifact, IndicationLandscape)
    assert msg.artifact.total_trial_count == 95
    assert msg.artifact.phase_distribution == {
        "Phase 2": 40,
        "Phase 3": 15,
        "Phase 4": 10,
    }
    assert len(msg.artifact.competitors) == 1
    assert msg.artifact.competitors[0].sponsor == "Chinese University of Hong Kong"
    assert msg.artifact.competitors[0].drug_name == "Esomeprazole"
    assert msg.artifact.competitors[0].drug_type == "Drug"
    assert msg.artifact.competitors[0].max_phase == "Phase 4"
    assert msg.artifact.competitors[0].trial_count == 1
    assert msg.artifact.competitors[0].total_enrollment == 155
    assert len(msg.artifact.recent_starts) == 1
    assert msg.artifact.recent_starts[0].nct_id == "NCT06836557"
    assert msg.artifact.recent_starts[0].sponsor == "Vanda Pharmaceuticals"
    assert msg.artifact.recent_starts[0].drug == "Tradipitant"
    assert msg.artifact.recent_starts[0].phase == "Phase 3"
    assert "1 competitors" in msg.content


async def test_get_landscape_passes_date_before():
    """get_landscape forwards date_before from closure to the client."""
    cutoff = date(2020, 1, 1)
    mock_result = IndicationLandscape(
        total_trial_count=10, competitors=[], phase_distribution={}, recent_starts=[]
    )
    mock_client = _mock_client(get_landscape=mock_result)
    tools = build_clinical_trials_tools(date_before=cutoff)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=("D018589", "Gastroparesis")),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            return_value=mock_client,
        ),
    ):
        await _get_tool(tools, "get_landscape").ainvoke(
            LCToolCall(
                name="get_landscape",
                args={"indication": "gastroparesis"},
                id="tc7",
                type="tool_call",
            )
        )

    mock_client.get_landscape.assert_awaited_once_with(
        "Gastroparesis",
        date_before=cutoff,
        top_n=10,
    )


async def test_get_landscape_returns_empty_when_resolver_returns_none():
    """get_landscape returns default IndicationLandscape and skips client when resolver is None."""
    client_factory = MagicMock()
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_mesh_id",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.ClinicalTrialsClient",
            new=client_factory,
        ),
    ):
        msg = await _get_tool(tools, "get_landscape").ainvoke(
            LCToolCall(
                name="get_landscape",
                args={"indication": "xyzzy_bogus"},
                id="tc_ms6",
                type="tool_call",
            )
        )

    assert isinstance(msg.artifact, IndicationLandscape)
    assert msg.artifact.total_trial_count is None
    assert msg.artifact.competitors == []
    assert msg.artifact.phase_distribution == {}
    assert msg.artifact.recent_starts == []
    assert "MeSH unresolved" in msg.content
    client_factory.assert_not_called()


# ------------------------------------------------------------------
# check_fda_approval
# ------------------------------------------------------------------


def _fda_client_mock(label_texts: list[str]) -> AsyncMock:
    """AsyncMock FDAClient that works as an async context manager."""
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.get_all_label_indications = AsyncMock(return_value=label_texts)
    return client


async def test_check_fda_approval_approved_match():
    """Drug resolved, labels found, indication approved → is_approved=True with all fields populated."""
    drug_names = ["semaglutide", "ozempic", "wegovy", "rybelsus"]
    label_texts = ["INDICATIONS AND USAGE: treatment of type 2 diabetes mellitus"]
    fda_client = _fda_client_mock(label_texts)
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_drug_name",
            new=AsyncMock(return_value="CHEMBL2108724"),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.get_all_drug_names",
            new=AsyncMock(return_value=drug_names),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.FDAClient",
            return_value=fda_client,
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.extract_approved_from_labels",
            new=AsyncMock(return_value={"type 2 diabetes mellitus"}),
        ),
    ):
        msg = await _get_tool(tools, "check_fda_approval").ainvoke(
            LCToolCall(
                name="check_fda_approval",
                args={"drug": "semaglutide", "indication": "type 2 diabetes mellitus"},
                id="tc_fda1",
                type="tool_call",
            )
        )

    fda_client.get_all_label_indications.assert_awaited_once_with(drug_names)
    assert isinstance(msg.artifact, ApprovalCheck)
    assert msg.artifact.is_approved is True
    assert msg.artifact.label_found is True
    assert msg.artifact.matched_indication == "type 2 diabetes mellitus"
    assert msg.artifact.drug_names_checked == drug_names
    assert "APPROVED" in msg.content


async def test_check_fda_approval_label_found_but_not_approved():
    """Labels returned but the indication is not in the approved set → is_approved=False, label_found=True."""
    drug_names = ["semaglutide", "ozempic", "wegovy", "rybelsus"]
    label_texts = ["INDICATIONS AND USAGE: treatment of type 2 diabetes mellitus"]
    fda_client = _fda_client_mock(label_texts)
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_drug_name",
            new=AsyncMock(return_value="CHEMBL2108724"),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.get_all_drug_names",
            new=AsyncMock(return_value=drug_names),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.FDAClient",
            return_value=fda_client,
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.extract_approved_from_labels",
            new=AsyncMock(return_value=set()),
        ),
    ):
        msg = await _get_tool(tools, "check_fda_approval").ainvoke(
            LCToolCall(
                name="check_fda_approval",
                args={"drug": "semaglutide", "indication": "huntington disease"},
                id="tc_fda2",
                type="tool_call",
            )
        )

    assert isinstance(msg.artifact, ApprovalCheck)
    assert msg.artifact.is_approved is False
    assert msg.artifact.label_found is True
    assert msg.artifact.matched_indication is None
    assert msg.artifact.drug_names_checked == drug_names
    assert "not on FDA label" in msg.content


async def test_check_fda_approval_indication_match_is_case_insensitive():
    """Indication match strips and lowercases both sides before comparing."""
    drug_names = ["semaglutide"]
    fda_client = _fda_client_mock(["some label text"])
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_drug_name",
            new=AsyncMock(return_value="CHEMBL2108724"),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.get_all_drug_names",
            new=AsyncMock(return_value=drug_names),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.FDAClient",
            return_value=fda_client,
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.extract_approved_from_labels",
            new=AsyncMock(return_value={"TYPE 2 Diabetes Mellitus"}),
        ),
    ):
        msg = await _get_tool(tools, "check_fda_approval").ainvoke(
            LCToolCall(
                name="check_fda_approval",
                args={
                    "drug": "semaglutide",
                    "indication": "  type 2 diabetes mellitus  ",
                },
                id="tc_fda3",
                type="tool_call",
            )
        )

    assert msg.artifact.is_approved is True
    assert msg.artifact.label_found is True
    assert msg.artifact.matched_indication == "  type 2 diabetes mellitus  "
    assert msg.artifact.drug_names_checked == drug_names


async def test_check_fda_approval_no_labels_found():
    """Drug names resolved but FDA has no labels → label_found=False, is_approved=False."""
    drug_names = ["aducanumab", "aduhelm"]
    fda_client = _fda_client_mock([])
    tools = build_clinical_trials_tools(date_before=None)
    extract_mock = AsyncMock()

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_drug_name",
            new=AsyncMock(return_value="CHEMBL4650346"),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.get_all_drug_names",
            new=AsyncMock(return_value=drug_names),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.FDAClient",
            return_value=fda_client,
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.extract_approved_from_labels",
            new=extract_mock,
        ),
    ):
        msg = await _get_tool(tools, "check_fda_approval").ainvoke(
            LCToolCall(
                name="check_fda_approval",
                args={"drug": "aducanumab", "indication": "alzheimer disease"},
                id="tc_fda4",
                type="tool_call",
            )
        )

    assert isinstance(msg.artifact, ApprovalCheck)
    assert msg.artifact.is_approved is False
    assert msg.artifact.label_found is False
    assert msg.artifact.matched_indication is None
    assert msg.artifact.drug_names_checked == drug_names
    assert "no FDA label" in msg.content
    extract_mock.assert_not_awaited()


async def test_check_fda_approval_unresolved_drug_returns_default():
    """resolve_drug_name raises DataSourceError → default ApprovalCheck, no FDA call, no extract call."""
    fda_factory = MagicMock()
    extract_mock = AsyncMock()
    get_names_mock = AsyncMock()
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_drug_name",
            new=AsyncMock(
                side_effect=DataSourceError("chembl", "No drug found for 'xyzzybogus'")
            ),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.get_all_drug_names",
            new=get_names_mock,
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.FDAClient",
            new=fda_factory,
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.extract_approved_from_labels",
            new=extract_mock,
        ),
    ):
        msg = await _get_tool(tools, "check_fda_approval").ainvoke(
            LCToolCall(
                name="check_fda_approval",
                args={"drug": "xyzzybogus", "indication": "alzheimer disease"},
                id="tc_fda5",
                type="tool_call",
            )
        )

    assert isinstance(msg.artifact, ApprovalCheck)
    assert msg.artifact.is_approved is False
    assert msg.artifact.label_found is False
    assert msg.artifact.matched_indication is None
    assert msg.artifact.drug_names_checked == []
    assert "drug not resolved" in msg.content
    get_names_mock.assert_not_awaited()
    fda_factory.assert_not_called()
    extract_mock.assert_not_awaited()


async def test_check_fda_approval_no_drug_names_returns_default():
    """get_all_drug_names returns empty → default ApprovalCheck, no FDA call, no extract call."""
    fda_factory = MagicMock()
    extract_mock = AsyncMock()
    tools = build_clinical_trials_tools(date_before=None)

    with (
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.resolve_drug_name",
            new=AsyncMock(return_value="CHEMBL2108724"),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.get_all_drug_names",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.FDAClient",
            new=fda_factory,
        ),
        patch(
            "indication_scout.agents.clinical_trials.clinical_trials_tools.extract_approved_from_labels",
            new=extract_mock,
        ),
    ):
        msg = await _get_tool(tools, "check_fda_approval").ainvoke(
            LCToolCall(
                name="check_fda_approval",
                args={"drug": "semaglutide", "indication": "type 2 diabetes mellitus"},
                id="tc_fda6",
                type="tool_call",
            )
        )

    assert isinstance(msg.artifact, ApprovalCheck)
    assert msg.artifact.is_approved is False
    assert msg.artifact.label_found is False
    assert msg.artifact.matched_indication is None
    assert msg.artifact.drug_names_checked == []
    assert "no drug names" in msg.content
    fda_factory.assert_not_called()
    extract_mock.assert_not_awaited()


# ------------------------------------------------------------------
# Tool registration
# ------------------------------------------------------------------


async def test_check_fda_approval_is_registered_in_tool_list():
    """build_clinical_trials_tools exposes check_fda_approval."""
    tools = build_clinical_trials_tools(date_before=None)
    names = {t.name for t in tools}
    assert "check_fda_approval" in names


async def test_tool_list_contains_expected_tools():
    """build_clinical_trials_tools returns the documented set of tools and excludes detect_whitespace."""
    tools = build_clinical_trials_tools(date_before=None)
    names = {t.name for t in tools}
    assert names == {
        "search_trials",
        "get_completed",
        "get_terminated",
        "get_landscape",
        "check_fda_approval",
        "finalize_analysis",
    }


# ------------------------------------------------------------------
# finalize_analysis
# ------------------------------------------------------------------


async def test_finalize_analysis_returns_summary_as_artifact():
    """finalize_analysis returns the summary string as artifact and confirms completion in content."""
    tools = build_clinical_trials_tools(date_before=None)

    text = "No trials found for tirzepatide in Huntington disease. The landscape shows 5 competitors."
    msg = await _get_tool(tools, "finalize_analysis").ainvoke(
        LCToolCall(
            name="finalize_analysis",
            args={"summary": text},
            id="tc_fin",
            type="tool_call",
        )
    )

    assert msg.artifact == text
    assert "Analysis complete" in msg.content
