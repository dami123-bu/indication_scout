"""Unit tests for ClinicalTrialsClient helper functions."""

import pytest

from indication_scout.data_sources.clinical_trials import (
    ClinicalTrialsClient,
    _classify_stop_reason,
)
from indication_scout.models.model_clinical_trials import (
    CompetitorEntry,
    ConditionLandscape,
    Intervention,
    PrimaryOutcome,
    RecentStart,
    Trial,
)


class TestClassifyStopReason:
    """Tests for _classify_stop_reason keyword-based classification."""

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
            # Other - no matching keywords
            ("COVID-19 pandemic impact", "other"),
            ("Protocol amendment required", "other"),
            ("Regulatory hold", "other"),
            # Unknown - no text
            (None, "unknown"),
            ("", "unknown"),
        ],
    )
    def test_classify_stop_reason(self, why_stopped, expected_category):
        """_classify_stop_reason should classify based on keywords."""
        assert _classify_stop_reason(why_stopped) == expected_category

    def test_classify_stop_reason_case_insensitive(self):
        """_classify_stop_reason should match keywords case-insensitively."""
        assert _classify_stop_reason("LACK OF EFFICACY") == "efficacy"
        assert _classify_stop_reason("Safety Concerns") == "safety"
        assert _classify_stop_reason("BUSINESS Decision") == "business"


class TestClinicalTrialsClientHelpers:
    """Tests for ClinicalTrialsClient static helper methods."""

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
    def test_normalize_phase(self, phases, expected):
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
    def test_phase_rank(self, phase, expected_rank):
        """_phase_rank should return numeric ranking for phase comparison."""
        assert ClinicalTrialsClient._phase_rank(phase) == expected_rank

    def test_phase_rank_ordering(self):
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
    def test_extract_date(self, date_struct, expected):
        """_extract_date should extract date string from v2 date struct."""
        assert ClinicalTrialsClient._extract_date(date_struct) == expected


class TestParseTrial:
    """Tests for ClinicalTrialsClient._parse_trial."""

    def test_parse_trial_full_record(self):
        """_parse_trial should map all v2 API fields to a Trial model."""
        study = {
            "hasResults": True,
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT04971785",
                    "briefTitle": "Semaglutide vs Placebo in NASH",
                },
                "statusModule": {
                    "overallStatus": "COMPLETED",
                    "whyStopped": "Efficacy endpoint met",
                    "startDateStruct": {"date": "2021-03-15"},
                    "primaryCompletionDateStruct": {"date": "2024-06-30"},
                },
                "designModule": {
                    "phases": ["PHASE2", "PHASE3"],
                    "studyType": "INTERVENTIONAL",
                    "enrollmentInfo": {"count": 320},
                },
                "descriptionModule": {
                    "briefSummary": "A study evaluating semaglutide in NASH patients.",
                },
                "conditionsModule": {
                    "conditions": [
                        "Non-alcoholic Steatohepatitis",
                        "Liver Fibrosis",
                    ],
                },
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {"name": "Novo Nordisk"},
                    "collaborators": [
                        {"name": "NIDDK"},
                        {"name": "Mayo Clinic"},
                    ],
                },
                "armsInterventionsModule": {
                    "interventions": [
                        {
                            "type": "DRUG",
                            "name": "Semaglutide",
                            "description": "2.4 mg SC weekly",
                        },
                        {
                            "type": "OTHER",
                            "name": "Placebo",
                            "description": None,
                        },
                    ],
                },
                "outcomesModule": {
                    "primaryOutcomes": [
                        {
                            "measure": "NASH resolution without worsening fibrosis",
                            "timeFrame": "72 weeks",
                        },
                        {
                            "measure": "Fibrosis improvement by >=1 stage",
                            "timeFrame": "72 weeks",
                        },
                    ],
                },
                "referencesModule": {
                    "references": [
                        {"pmid": "34567890", "type": "RESULT"},
                        {"pmid": "34567891", "type": "BACKGROUND"},
                        {"type": "BACKGROUND"},
                    ],
                },
            },
        }

        client = ClinicalTrialsClient.__new__(ClinicalTrialsClient)
        result = client._parse_trial(study)

        assert isinstance(result, Trial)
        assert result.nct_id == "NCT04971785"
        assert result.title == "Semaglutide vs Placebo in NASH"
        assert (
            result.brief_summary == "A study evaluating semaglutide in NASH patients."
        )
        assert result.phase == "Phase 2/Phase 3"
        assert result.overall_status == "COMPLETED"
        assert result.why_stopped == "Efficacy endpoint met"
        assert result.conditions == ["Non-alcoholic Steatohepatitis", "Liver Fibrosis"]
        assert result.sponsor == "Novo Nordisk"
        assert result.collaborators == ["NIDDK", "Mayo Clinic"]
        assert result.enrollment == 320
        assert result.start_date == "2021-03-15"
        assert result.completion_date == "2024-06-30"
        assert result.study_type == "INTERVENTIONAL"
        assert result.results_posted is True
        assert result.references == ["34567890", "34567891"]

        assert len(result.interventions) == 2
        drug_interv = result.interventions[0]
        assert drug_interv.intervention_type == "Drug"
        assert drug_interv.intervention_name == "Semaglutide"
        assert drug_interv.description == "2.4 mg SC weekly"
        placebo_interv = result.interventions[1]
        assert placebo_interv.intervention_type == "Other"
        assert placebo_interv.intervention_name == "Placebo"
        assert placebo_interv.description is None

        assert len(result.primary_outcomes) == 2
        outcome_1 = result.primary_outcomes[0]
        assert outcome_1.measure == "NASH resolution without worsening fibrosis"
        assert outcome_1.time_frame == "72 weeks"
        outcome_2 = result.primary_outcomes[1]
        assert outcome_2.measure == "Fibrosis improvement by >=1 stage"
        assert outcome_2.time_frame == "72 weeks"


def _make_trial(
    nct_id: str,
    phase: str,
    overall_status: str,
    sponsor: str,
    drug_name: str,
    drug_type: str = "Drug",
    enrollment: int | None = None,
    start_date: str | None = None,
) -> Trial:
    """Helper to build a Trial with a single drug intervention."""
    return Trial(
        nct_id=nct_id,
        title=f"Trial {nct_id}",
        phase=phase,
        overall_status=overall_status,
        sponsor=sponsor,
        enrollment=enrollment,
        start_date=start_date,
        interventions=[
            Intervention(
                intervention_type=drug_type,
                intervention_name=drug_name,
            ),
        ],
    )


class TestAggregateLandscape:
    """Tests for ClinicalTrialsClient._aggregate_landscape."""

    def test_aggregate_landscape(self):
        """_aggregate_landscape should group trials by sponsor+drug, rank by phase then enrollment."""
        trials = [
            # Pfizer has two trials for DrugA — Phase 3 (recruiting) and Phase 2 (completed)
            _make_trial(
                nct_id="NCT00000001",
                phase="Phase 3",
                overall_status="RECRUITING",
                sponsor="Pfizer",
                drug_name="DrugA",
                enrollment=500,
                start_date="2024-05-01",
            ),
            _make_trial(
                nct_id="NCT00000002",
                phase="Phase 2",
                overall_status="COMPLETED",
                sponsor="Pfizer",
                drug_name="DrugA",
                enrollment=200,
                start_date="2022-01-15",
            ),
            # Roche has one Phase 3 trial for DrugB (biological), smaller enrollment
            _make_trial(
                nct_id="NCT00000003",
                phase="Phase 3",
                overall_status="ACTIVE_NOT_RECRUITING",
                sponsor="Roche",
                drug_name="DrugB",
                drug_type="Biological",
                enrollment=300,
                start_date="2024-08-10",
            ),
            # Novartis has a Phase 1 trial — should rank last
            _make_trial(
                nct_id="NCT00000004",
                phase="Phase 1",
                overall_status="RECRUITING",
                sponsor="Novartis",
                drug_name="DrugC",
                enrollment=50,
                start_date="2025-01-01",
            ),
            # Trial with only a Device intervention — should be skipped entirely
            Trial(
                nct_id="NCT00000005",
                title="Device Trial",
                phase="Phase 3",
                overall_status="RECRUITING",
                sponsor="Medtronic",
                enrollment=1000,
                start_date="2024-03-01",
                interventions=[
                    Intervention(
                        intervention_type="Device",
                        intervention_name="StentX",
                    ),
                ],
            ),
        ]

        client = ClinicalTrialsClient.__new__(ClinicalTrialsClient)
        result = client._aggregate_landscape(trials, top_n=50)

        assert isinstance(result, ConditionLandscape)

        # total_trial_count includes ALL trials passed in (including the device trial)
        assert result.total_trial_count == 5

        # Phase distribution only counts drug/biological trials (device trial excluded)
        assert result.phase_distribution == {
            "Phase 3": 2,
            "Phase 2": 1,
            "Phase 1": 1,
        }

        # Recent starts: trials with start_date >= "2024" (3 drug trials qualify)
        assert len(result.recent_starts) == 3
        recent_nct_ids = {rs.nct_id for rs in result.recent_starts}
        assert recent_nct_ids == {"NCT00000001", "NCT00000003", "NCT00000004"}
        # Verify all fields on one recent start
        pfizer_recent = next(
            rs for rs in result.recent_starts if rs.nct_id == "NCT00000001"
        )
        assert pfizer_recent.sponsor == "Pfizer"
        assert pfizer_recent.drug == "DrugA"
        assert pfizer_recent.phase == "Phase 3"

        # 3 competitors: Pfizer|DrugA, Roche|DrugB, Novartis|DrugC
        assert len(result.competitors) == 3

        # Ranked by phase then enrollment:
        # 1. Pfizer|DrugA  — Phase 3, enrollment 700
        # 2. Roche|DrugB   — Phase 3, enrollment 300
        # 3. Novartis|DrugC — Phase 1, enrollment 50
        pfizer = result.competitors[0]
        assert pfizer.sponsor == "Pfizer"
        assert pfizer.drug_name == "DrugA"
        assert pfizer.drug_type == "Drug"
        assert pfizer.max_phase == "Phase 3"
        assert pfizer.trial_count == 2
        assert pfizer.statuses == {"RECRUITING", "COMPLETED"}
        assert pfizer.total_enrollment == 700
        assert pfizer.most_recent_start == "2024-05-01"

        roche = result.competitors[1]
        assert roche.sponsor == "Roche"
        assert roche.drug_name == "DrugB"
        assert roche.drug_type == "Biological"
        assert roche.max_phase == "Phase 3"
        assert roche.trial_count == 1
        assert roche.statuses == {"ACTIVE_NOT_RECRUITING"}
        assert roche.total_enrollment == 300
        assert roche.most_recent_start == "2024-08-10"

        novartis = result.competitors[2]
        assert novartis.sponsor == "Novartis"
        assert novartis.drug_name == "DrugC"
        assert novartis.drug_type == "Drug"
        assert novartis.max_phase == "Phase 1"
        assert novartis.trial_count == 1
        assert novartis.statuses == {"RECRUITING"}
        assert novartis.total_enrollment == 50
        assert novartis.most_recent_start == "2025-01-01"
