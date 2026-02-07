"""Unit tests for Clinical Trials models."""

import pytest
from pydantic import ValidationError

from indication_scout.models.model_clinical_trials import (
    CompetitorEntry,
    ConditionLandscape,
    Intervention,
    NearMiss,
    PrimaryOutcome,
    TerminatedTrial,
    Trial,
    WhitespaceResult,
)


class TestInterventionAndPrimaryOutcome:
    """Tests for Intervention and PrimaryOutcome models."""

    @pytest.mark.parametrize(
        "intervention_type,intervention_name,description",
        [
            ("Drug", "Semaglutide", "GLP-1 receptor agonist"),
            ("Biological", "Pembrolizumab", "PD-1 inhibitor antibody"),
            ("Device", "Insulin Pump", None),
            ("Procedure", "Bariatric Surgery", "Weight loss surgery"),
        ],
    )
    def test_intervention_with_various_types(
        self, intervention_type, intervention_name, description
    ):
        """Intervention should accept various intervention types."""
        intervention = Intervention(
            intervention_type=intervention_type,
            intervention_name=intervention_name,
            description=description,
        )
        assert intervention.intervention_type == intervention_type
        assert intervention.intervention_name == intervention_name
        assert intervention.description == description

    def test_intervention_requires_type_and_name(self):
        """Intervention should require intervention_type and intervention_name."""
        with pytest.raises(ValidationError):
            Intervention(intervention_name="Semaglutide")  # missing type

        with pytest.raises(ValidationError):
            Intervention(intervention_type="Drug")  # missing name

    @pytest.mark.parametrize(
        "measure,time_frame",
        [
            ("Change in HbA1c from baseline", "24 weeks"),
            ("Overall survival", "5 years"),
            ("Tumor response rate", None),
        ],
    )
    def test_primary_outcome_with_various_measures(self, measure, time_frame):
        """PrimaryOutcome should accept various measures and time frames."""
        outcome = PrimaryOutcome(measure=measure, time_frame=time_frame)
        assert outcome.measure == measure
        assert outcome.time_frame == time_frame

    def test_primary_outcome_requires_measure(self):
        """PrimaryOutcome should require measure field."""
        with pytest.raises(ValidationError):
            PrimaryOutcome(time_frame="24 weeks")  # missing measure


class TestTrial:
    """Tests for Trial model."""

    @pytest.fixture
    def sample_trial(self):
        """Create a sample Trial for testing."""
        return Trial(
            nct_id="NCT04375669",
            title="A Study of Semaglutide in Participants With Type 2 Diabetes",
            brief_summary="This study evaluates the efficacy of semaglutide.",
            phase="Phase 3",
            overall_status="Completed",
            conditions=["Type 2 Diabetes Mellitus", "Obesity"],
            interventions=[
                Intervention(
                    intervention_type="Drug",
                    intervention_name="Semaglutide",
                    description="2.4mg subcutaneous injection",
                ),
                Intervention(
                    intervention_type="Drug",
                    intervention_name="Placebo",
                    description="Matching placebo",
                ),
            ],
            sponsor="Novo Nordisk",
            collaborators=["NIH", "FDA"],
            enrollment=1961,
            start_date="2020-03-15",
            completion_date="2022-06-30",
            study_type="Interventional",
            primary_outcomes=[
                PrimaryOutcome(
                    measure="Change in body weight",
                    time_frame="68 weeks",
                ),
            ],
            results_posted=True,
            references=["34567890", "34567891"],
        )

    def test_trial_all_fields(self, sample_trial):
        """Trial should store all fields correctly."""
        assert sample_trial.nct_id == "NCT04375669"
        assert (
            sample_trial.title
            == "A Study of Semaglutide in Participants With Type 2 Diabetes"
        )
        assert (
            sample_trial.brief_summary
            == "This study evaluates the efficacy of semaglutide."
        )
        assert sample_trial.phase == "Phase 3"
        assert sample_trial.overall_status == "Completed"
        assert sample_trial.why_stopped is None
        assert sample_trial.conditions == ["Type 2 Diabetes Mellitus", "Obesity"]
        assert len(sample_trial.interventions) == 2
        assert sample_trial.interventions[0].intervention_name == "Semaglutide"
        assert sample_trial.interventions[1].intervention_name == "Placebo"
        assert sample_trial.sponsor == "Novo Nordisk"
        assert sample_trial.collaborators == ["NIH", "FDA"]
        assert sample_trial.enrollment == 1961
        assert sample_trial.start_date == "2020-03-15"
        assert sample_trial.completion_date == "2022-06-30"
        assert sample_trial.study_type == "Interventional"
        assert len(sample_trial.primary_outcomes) == 1
        assert sample_trial.primary_outcomes[0].measure == "Change in body weight"
        assert sample_trial.results_posted is True
        assert sample_trial.references == ["34567890", "34567891"]

    def test_trial_minimal_required_fields(self):
        """Trial should work with only required fields."""
        trial = Trial(
            nct_id="NCT12345678",
            title="Minimal Trial",
            phase="Phase 1",
            overall_status="Recruiting",
            sponsor="Test Sponsor",
        )
        assert trial.nct_id == "NCT12345678"
        assert trial.title == "Minimal Trial"
        assert trial.phase == "Phase 1"
        assert trial.overall_status == "Recruiting"
        assert trial.sponsor == "Test Sponsor"
        # Defaults
        assert trial.brief_summary is None
        assert trial.why_stopped is None
        assert trial.conditions == []
        assert trial.interventions == []
        assert trial.collaborators == []
        assert trial.enrollment is None
        assert trial.start_date is None
        assert trial.completion_date is None
        assert trial.study_type == "Interventional"
        assert trial.primary_outcomes == []
        assert trial.results_posted is False
        assert trial.references == []

    def test_trial_with_why_stopped(self):
        """Trial should store why_stopped for terminated trials."""
        trial = Trial(
            nct_id="NCT99999999",
            title="Terminated Trial",
            phase="Phase 2",
            overall_status="Terminated",
            why_stopped="Lack of efficacy",
            sponsor="Test Pharma",
        )
        assert trial.why_stopped == "Lack of efficacy"
        assert trial.overall_status == "Terminated"

    @pytest.mark.parametrize(
        "phase",
        [
            "Early Phase 1",
            "Phase 1",
            "Phase 1/Phase 2",
            "Phase 2",
            "Phase 2/Phase 3",
            "Phase 3",
            "Phase 4",
            "Not Applicable",
        ],
    )
    def test_trial_accepts_various_phases(self, phase):
        """Trial should accept various phase formats."""
        trial = Trial(
            nct_id="NCT00000001",
            title="Test Trial",
            phase=phase,
            overall_status="Recruiting",
            sponsor="Test",
        )
        assert trial.phase == phase

    @pytest.mark.parametrize(
        "status",
        [
            "Not yet recruiting",
            "Recruiting",
            "Enrolling by invitation",
            "Active, not recruiting",
            "Suspended",
            "Terminated",
            "Completed",
            "Withdrawn",
            "Unknown status",
        ],
    )
    def test_trial_accepts_various_statuses(self, status):
        """Trial should accept various overall_status values."""
        trial = Trial(
            nct_id="NCT00000001",
            title="Test Trial",
            phase="Phase 2",
            overall_status=status,
            sponsor="Test",
        )
        assert trial.overall_status == status


class TestNearMissAndWhitespaceResult:
    """Tests for NearMiss and WhitespaceResult models."""

    @pytest.fixture
    def sample_near_miss(self):
        """Create a sample NearMiss."""
        return NearMiss(
            nct_id="NCT04375669",
            drug_name="Semaglutide",
            condition="NASH",
            phase="Phase 3",
            status="Active, not recruiting",
        )

    def test_near_miss_all_fields(self, sample_near_miss):
        """NearMiss should store all fields correctly."""
        assert sample_near_miss.nct_id == "NCT04375669"
        assert sample_near_miss.drug_name == "Semaglutide"
        assert sample_near_miss.condition == "NASH"
        assert sample_near_miss.phase == "Phase 3"
        assert sample_near_miss.status == "Active, not recruiting"

    def test_near_miss_requires_all_fields(self):
        """NearMiss should require all fields."""
        with pytest.raises(ValidationError):
            NearMiss(
                nct_id="NCT04375669",
                drug_name="Semaglutide",
                condition="NASH",
                phase="Phase 3",
                # missing status
            )

    def test_whitespace_result_is_whitespace_true(self):
        """WhitespaceResult should handle is_whitespace=True case."""
        result = WhitespaceResult(
            is_whitespace=True,
            exact_match_count=0,
            drug_only_trials=15,
            condition_only_trials=42,
            near_misses=[],
        )
        assert result.is_whitespace is True
        assert result.exact_match_count == 0
        assert result.drug_only_trials == 15
        assert result.condition_only_trials == 42
        assert result.near_misses == []

    def test_whitespace_result_is_whitespace_false_with_near_misses(self):
        """WhitespaceResult should store near_misses when not whitespace."""
        near_miss = NearMiss(
            nct_id="NCT04375669",
            drug_name="Semaglutide",
            condition="NAFLD",  # close to NASH
            phase="Phase 2",
            status="Completed",
        )
        result = WhitespaceResult(
            is_whitespace=False,
            exact_match_count=3,
            drug_only_trials=10,
            condition_only_trials=25,
            near_misses=[near_miss],
        )
        assert result.is_whitespace is False
        assert result.exact_match_count == 3
        assert len(result.near_misses) == 1
        assert result.near_misses[0].nct_id == "NCT04375669"
        assert result.near_misses[0].condition == "NAFLD"

    def test_whitespace_result_defaults_near_misses_to_empty(self):
        """WhitespaceResult should default near_misses to empty list."""
        result = WhitespaceResult(
            is_whitespace=True,
            exact_match_count=0,
            drug_only_trials=5,
            condition_only_trials=10,
        )
        assert result.near_misses == []


class TestCompetitorEntryAndConditionLandscape:
    """Tests for CompetitorEntry and ConditionLandscape models."""

    @pytest.fixture
    def sample_competitor(self):
        """Create a sample CompetitorEntry."""
        return CompetitorEntry(
            sponsor="Eli Lilly",
            drug_name="Tirzepatide",
            drug_type="Biological",
            max_phase="Phase 3",
            trial_count=12,
            statuses={"Recruiting", "Active, not recruiting", "Completed"},
            total_enrollment=8500,
            most_recent_start="2023-06-15",
        )

    def test_competitor_entry_all_fields(self, sample_competitor):
        """CompetitorEntry should store all fields correctly."""
        assert sample_competitor.sponsor == "Eli Lilly"
        assert sample_competitor.drug_name == "Tirzepatide"
        assert sample_competitor.drug_type == "Biological"
        assert sample_competitor.max_phase == "Phase 3"
        assert sample_competitor.trial_count == 12
        assert sample_competitor.statuses == {
            "Recruiting",
            "Active, not recruiting",
            "Completed",
        }
        assert sample_competitor.total_enrollment == 8500
        assert sample_competitor.most_recent_start == "2023-06-15"

    def test_competitor_entry_optional_fields(self):
        """CompetitorEntry should allow optional fields to be None."""
        competitor = CompetitorEntry(
            sponsor="Unknown Pharma",
            drug_name="Drug X",
            drug_type=None,
            max_phase="Phase 1",
            trial_count=1,
            statuses={"Recruiting"},
            total_enrollment=50,
            most_recent_start=None,
        )
        assert competitor.drug_type is None
        assert competitor.most_recent_start is None

    def test_condition_landscape_all_fields(self, sample_competitor):
        """ConditionLandscape should store all fields correctly."""
        recent_trial = {
            "nct_id": "NCT05000001",
            "title": "Recent Trial",
            "start_date": "2023-09-01",
        }
        landscape = ConditionLandscape(
            total_trial_count=150,
            competitors=[sample_competitor],
            phase_distribution={
                "Phase 1": 20,
                "Phase 2": 55,
                "Phase 3": 60,
                "Phase 4": 15,
            },
            recent_starts=[recent_trial],
        )
        assert landscape.total_trial_count == 150
        assert len(landscape.competitors) == 1
        assert landscape.competitors[0].sponsor == "Eli Lilly"
        assert landscape.phase_distribution == {
            "Phase 1": 20,
            "Phase 2": 55,
            "Phase 3": 60,
            "Phase 4": 15,
        }
        assert len(landscape.recent_starts) == 1
        assert landscape.recent_starts[0]["nct_id"] == "NCT05000001"

    def test_condition_landscape_empty_competitors(self):
        """ConditionLandscape should accept empty competitors list."""
        landscape = ConditionLandscape(
            total_trial_count=0,
            competitors=[],
            phase_distribution={},
            recent_starts=[],
        )
        assert landscape.total_trial_count == 0
        assert landscape.competitors == []
        assert landscape.phase_distribution == {}
        assert landscape.recent_starts == []


class TestTerminatedTrial:
    """Tests for TerminatedTrial model."""

    @pytest.fixture
    def sample_terminated_trial(self):
        """Create a sample TerminatedTrial."""
        return TerminatedTrial(
            nct_id="NCT03456789",
            title="A Study That Was Stopped Early",
            drug_name="Failed Drug",
            condition="Type 2 Diabetes",
            phase="Phase 3",
            why_stopped="Interim analysis showed lack of efficacy",
            stop_category="efficacy",
            enrollment=2500,
            sponsor="Big Pharma Inc",
            start_date="2019-01-15",
            termination_date="2021-06-30",
            references=["35000001", "35000002"],
        )

    def test_terminated_trial_all_fields(self, sample_terminated_trial):
        """TerminatedTrial should store all fields correctly."""
        assert sample_terminated_trial.nct_id == "NCT03456789"
        assert sample_terminated_trial.title == "A Study That Was Stopped Early"
        assert sample_terminated_trial.drug_name == "Failed Drug"
        assert sample_terminated_trial.condition == "Type 2 Diabetes"
        assert sample_terminated_trial.phase == "Phase 3"
        assert (
            sample_terminated_trial.why_stopped
            == "Interim analysis showed lack of efficacy"
        )
        assert sample_terminated_trial.stop_category == "efficacy"
        assert sample_terminated_trial.enrollment == 2500
        assert sample_terminated_trial.sponsor == "Big Pharma Inc"
        assert sample_terminated_trial.start_date == "2019-01-15"
        assert sample_terminated_trial.termination_date == "2021-06-30"
        assert sample_terminated_trial.references == ["35000001", "35000002"]

    def test_terminated_trial_minimal_fields(self):
        """TerminatedTrial should work with only required fields."""
        trial = TerminatedTrial(
            nct_id="NCT00000001",
            title="Minimal Terminated Trial",
        )
        assert trial.nct_id == "NCT00000001"
        assert trial.title == "Minimal Terminated Trial"
        # Defaults
        assert trial.drug_name is None
        assert trial.condition is None
        assert trial.phase is None
        assert trial.why_stopped is None
        assert trial.stop_category == "unknown"
        assert trial.enrollment is None
        assert trial.sponsor is None
        assert trial.start_date is None
        assert trial.termination_date is None
        assert trial.references == []

    @pytest.mark.parametrize(
        "stop_category,why_stopped",
        [
            ("safety", "Serious adverse events observed"),
            ("efficacy", "Primary endpoint not met"),
            ("business", "Strategic decision by sponsor"),
            ("enrollment", "Insufficient enrollment"),
            ("other", "COVID-19 pandemic impact"),
            ("unknown", None),
        ],
    )
    def test_terminated_trial_stop_categories(self, stop_category, why_stopped):
        """TerminatedTrial should accept various stop_category values."""
        trial = TerminatedTrial(
            nct_id="NCT00000001",
            title="Test Terminated Trial",
            stop_category=stop_category,
            why_stopped=why_stopped,
        )
        assert trial.stop_category == stop_category
        assert trial.why_stopped == why_stopped
