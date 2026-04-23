"""Integration tests for ClinicalTrialsClient."""

import logging

import pytest

from indication_scout.data_sources.base_client import DataSourceError
from indication_scout.models.model_clinical_trials import MeshTerm, TrialOutcomes

logger = logging.getLogger(__name__)


# --- Main functionality ---


async def test_get_landscape(clinical_trials_client):
    """Test get_landscape returns competitive landscape for an indication.

    get_landscape filters to Drug/Biological interventions, excludes vaccines,
    groups by sponsor + drug, and ranks by phase (desc) then most_recent_start (desc).
    """
    landscape = await clinical_trials_client.get_landscape("gastroparesis", top_n=10)

    # Verify IndicationLandscape.total_trial_count - gastroparesis has ~300+ trials
    assert landscape.total_trial_count > 200

    # Verify IndicationLandscape.competitors - requested top_n=10
    assert len(landscape.competitors) == 10

    # Verify IndicationLandscape.phase_distribution
    assert 10 < landscape.phase_distribution["Phase 2"] < 100
    assert 5 < landscape.phase_distribution["Phase 3"] < 50
    assert 1 < landscape.phase_distribution["Phase 4"] < 10

    # Verify IndicationLandscape.recent_starts - find a known 2024+ trial
    assert len(landscape.recent_starts) >= 1
    [tradipitant] = [rs for rs in landscape.recent_starts if rs.nct_id == "NCT06836557"]
    assert tradipitant.nct_id == "NCT06836557"
    assert tradipitant.sponsor == "Vanda Pharmaceuticals"
    assert tradipitant.drug == "Tradipitant"
    assert tradipitant.phase == "Phase 3"

    # --- Ranking: phase first, then most_recent_start descending ---
    # Phase 4 competitors must all appear before Phase 3 competitors
    phases = [c.max_phase for c in landscape.competitors]
    phase_4_indices = [i for i, p in enumerate(phases) if p == "Phase 4"]
    phase_3_indices = [i for i, p in enumerate(phases) if p == "Phase 3"]
    assert phase_4_indices, "Expected at least one Phase 4 competitor in top 10"
    assert phase_3_indices, "Expected at least one Phase 3 competitor in top 10"
    assert max(phase_4_indices) < min(phase_3_indices)

    # Within Phase 4, competitors are ordered by most_recent_start descending
    phase_4_competitors = [c for c in landscape.competitors if c.max_phase == "Phase 4"]
    starts = [c.most_recent_start for c in phase_4_competitors]
    assert starts == sorted(starts, reverse=True)

    # --- Vaccine exclusion ---
    # No competitor name should match vaccine keywords
    for c in landscape.competitors:
        name_lower = c.drug_name.lower()
        assert "vaccine" not in name_lower
        assert "vax" not in name_lower
        assert "immuniz" not in name_lower

    # --- most_recent_start field is populated ---
    # Find Vanda Pharmaceuticals / Tradipitant (Phase 3, two trials, most recent 2024-01-09)
    [tradipitant_comp] = [
        c
        for c in landscape.competitors
        if c.sponsor == "Vanda Pharmaceuticals" and c.drug_name == "Tradipitant"
    ]
    assert tradipitant_comp.sponsor == "Vanda Pharmaceuticals"
    assert tradipitant_comp.drug_name == "Tradipitant"
    assert tradipitant_comp.drug_type == "Drug"
    assert tradipitant_comp.max_phase == "Phase 3"
    assert tradipitant_comp.trial_count == 2
    assert tradipitant_comp.statuses == {"COMPLETED", "RECRUITING"}
    assert tradipitant_comp.total_enrollment == 1092
    assert tradipitant_comp.most_recent_start == "2024-01-09"


async def test_get_trial(clinical_trials_client):
    """Test get_trial returns a single trial by NCT ID."""
    # Fetch NCT00127933 - XeNA Study (Roche breast cancer trial)
    trial = await clinical_trials_client.get_trial("NCT00127933")

    # Verify all Trial fields with exact values
    assert trial.nct_id == "NCT00127933"
    assert (
        trial.title
        == "XeNA Study - A Study of Xeloda (Capecitabine) in Patients With Invasive Breast Cancer"
    )
    assert trial.phase == "Phase 4"
    assert trial.overall_status == "COMPLETED"
    assert trial.why_stopped is None
    assert trial.indications == ["Breast Cancer"]
    assert trial.sponsor == "Hoffmann-La Roche"
    assert trial.enrollment == 157
    assert trial.start_date == "2005-08"
    assert trial.completion_date == "2009-04"
    assert trial.references == []

    # Verify interventions - trial has 5 drug interventions
    assert len(trial.interventions) == 5
    [herceptin] = [
        i
        for i in trial.interventions
        if i.intervention_name == "Herceptin (HER2-neu positive patients only)"
    ]
    assert herceptin.intervention_type == "Drug"

    # Verify primary_outcomes
    assert len(trial.primary_outcomes) == 1
    assert (
        trial.primary_outcomes[0].measure
        == "Percentage of Participants Assessed for Pathological Complete Response (pCR) Plus Near Complete (npCR) in Primary Breast Tumor at Time of Definitive Surgery"
    )
    assert (
        trial.primary_outcomes[0].time_frame
        == "at the time of definitive surgery; after four 3-week cycles (3-4 months)"
    )


async def test_search_trials_drug_only(clinical_trials_client):
    """Test search_trials returns trials for a drug without specifying indication."""
    trials = await clinical_trials_client.search_trials(
        drug="semaglutide",

    )

    # Semaglutide has many trials across diabetes, obesity, NASH, etc.
    assert len(trials) >= 20

    # Verify trials span multiple indications
    all_indications = set()
    for trial in trials:
        all_indications.update(trial.indications)
    assert len(all_indications) >= 10


async def test_search_trials_nash_trial_fields(clinical_trials_client):
    """Verify all Trial fields (including MeSH ancestors) for NCT04971785.

    Covers both full-field parsing and conditionBrowseModule.ancestors
    extraction in a single live call.
    """
    trials = await clinical_trials_client.search_trials(
        drug="semaglutide",
        indication="nonalcoholic steatohepatitis",

    )

    [nash_trial] = [t for t in trials if t.nct_id == "NCT04971785"]

    assert nash_trial.nct_id == "NCT04971785"
    assert (
        nash_trial.title
        == "Study of Semaglutide, and Cilofexor/Firsocostat, Alone and in Combination, in Adults With Cirrhosis Due to Nonalcoholic Steatohepatitis (NASH)"
    )
    assert nash_trial.phase == "Phase 2"
    assert nash_trial.overall_status == "COMPLETED"
    assert nash_trial.why_stopped is None
    assert nash_trial.indications == ["Nonalcoholic Steatohepatitis"]
    assert nash_trial.sponsor == "Gilead Sciences"
    assert nash_trial.enrollment == 457
    assert nash_trial.start_date == "2021-08-09"
    assert nash_trial.completion_date == "2024-11-12"
    assert nash_trial.references == []

    assert len(nash_trial.interventions) == 4
    [sema] = [
        i
        for i in nash_trial.interventions
        if i.intervention_name == "Semaglutide (SEMA)"
    ]
    assert sema.intervention_type == "Drug"

    assert len(nash_trial.primary_outcomes) == 1
    assert nash_trial.primary_outcomes[0].measure.startswith(
        "Percentage of Participants Who Achieved"
    )

    # MeSH conditions + ancestors from conditionBrowseModule
    assert nash_trial.mesh_conditions == [
        MeshTerm(id="D065626", term="Non-alcoholic Fatty Liver Disease"),
    ]
    ancestor_pairs = [(m.id, m.term) for m in nash_trial.mesh_ancestors]
    assert ancestor_pairs == [
        ("D005234", "Fatty Liver"),
        ("D008107", "Liver Diseases"),
        ("D004066", "Digestive System Diseases"),
    ]


async def test_search_trials_indication_only(clinical_trials_client):
    """Test search_trials returns trials for an indication without specifying drug."""
    # Search for gastroparesis trials across all drugs
    trials = await clinical_trials_client.search_trials(
        drug="",
        indication="gastroparesis",

        phase_filter="PHASE4",
    )

    # Gastroparesis has multiple Phase 4 trials
    assert len(trials) >= 5

    # Find NCT00492622 - University of Louisville omeprazole PK study
    [pk_trial] = [t for t in trials if t.nct_id == "NCT00492622"]

    # Verify all Trial fields with exact values
    assert pk_trial.nct_id == "NCT00492622"
    assert (
        pk_trial.title
        == "Pharmacokinetics of Immediate-Release vs. Delayed-Release Omeprazole in Gastroparesis"
    )
    assert pk_trial.phase == "Phase 4"
    assert pk_trial.overall_status == "COMPLETED"
    assert pk_trial.why_stopped is None
    assert pk_trial.indications == [
        "Gastroparesis",
        "Gastroesophageal Reflux Disease",
    ]
    assert pk_trial.sponsor == "University of Louisville"
    assert pk_trial.enrollment == 12
    assert pk_trial.start_date == "2007-06"
    assert pk_trial.completion_date == "2008-12"
    assert pk_trial.references == ["19925497"]

    # Verify interventions - trial has 2 drug interventions
    assert len(pk_trial.interventions) == 2
    intervention_names = [i.intervention_name for i in pk_trial.interventions]
    assert "Immediate-release omeprazole" in intervention_names
    assert "Delayed-release omeprazole" in intervention_names

    # Verify primary_outcomes
    assert len(pk_trial.primary_outcomes) == 3
    assert (
        pk_trial.primary_outcomes[0].measure
        == "Time to Maximal Omeprazole Concentration (Tmax)"
    )
    assert (
        pk_trial.primary_outcomes[0].time_frame
        == "10, 20, 30, 45, 60, 90, 120, 150, 180, 210, 240 and 300 min after the study drug was ingested on day 7 of treatment"
    )


async def test_search_trials_phase_filter(clinical_trials_client):
    """Test that phase_filter only returns trials matching the specified phase."""
    # Search for Phase 3 trials only
    trials = await clinical_trials_client.search_trials(
        drug="semaglutide",
        indication="diabetes",

        phase_filter="PHASE3",
    )

    # All returned trials must be Phase 3
    assert len(trials) >= 5  # semaglutide + diabetes has many Phase 3 trials
    for trial in trials:
        assert "Phase 3" in trial.phase


async def test_get_terminated(clinical_trials_client):
    """get_terminated returns terminated trials split into scope-labelled lists.

    Verifies indication_wide retains business terminations (only drug_wide applies
    the safety/efficacy stop_category filter).

    Uses "dilated gastrojejunostomy" — a narrow condition whose only terminated
    trial on CT.gov is NCT00394212 (stopped "For business reasons"). This keeps
    the trial in the top-N indication cap regardless of overall terminated volume.
    The drug arg is irrelevant for indication_wide (no drug filter is applied there).
    """
    outcomes = await clinical_trials_client.get_terminated(
        "semaglutide", "dilated gastrojejunostomy"
    )

    [business_trial] = [
        t for t in outcomes.indication_wide if t.nct_id == "NCT00394212"
    ]
    assert business_trial.stop_category == "business"


async def test_detect_whitespace(clinical_trials_client):
    """Test detect_whitespace identifies unexplored drug-indication pairs.

    When is_whitespace=True, returns indication_drugs (other drugs being tested
    for this indication) ranked by phase (desc) then active status, deduplicated
    by drug_name.
    When is_whitespace=False, indication_drugs is empty.
    """
    # Tirzepatide + Huntington disease = whitespace (no exact matches)
    result = await clinical_trials_client.detect_whitespace(
        "tirzepatide", "Huntington disease"
    )

    # Verify WhitespaceResult fields
    assert result.is_whitespace is True
    assert result.exact_match_count == 0
    assert 150 < result.drug_only_trials < 300
    assert 200 < result.indication_only_trials < 400

    # Verify indication_drugs
    assert 40 < len(result.indication_drugs) < 60

    # Verify deduplication: all drug_names should be unique
    drug_names = [cd.drug_name for cd in result.indication_drugs]
    assert len(drug_names) == len(set(drug_names))

    # Verify known drugs are found (Memantine and Tetrabenazine are Phase 4 HD drugs)
    assert "Memantine" in drug_names
    assert "Tetrabenazine" in drug_names

    # Verify ranking: first drugs should be Phase 4 (highest phase)
    assert result.indication_drugs[0].phase == "Phase 4"
    assert result.indication_drugs[1].phase == "Phase 4"
    assert result.indication_drugs[2].phase == "Phase 4"

    # Find Memantine (Phase 4 completed trial) and verify all IndicationDrug fields
    [memantine] = [cd for cd in result.indication_drugs if cd.drug_name == "Memantine"]
    assert memantine.nct_id == "NCT00652457"
    assert memantine.drug_name == "Memantine"
    assert memantine.indication == "Huntington's Disease"
    assert memantine.phase == "Phase 4"
    assert memantine.status == "COMPLETED"

    # Verify Phase 2+ filter: no Phase 1 or Early Phase 1 trials
    for cd in result.indication_drugs:
        assert "Phase 1" not in cd.phase
        assert "Early Phase 1" not in cd.phase


async def test_detect_whitespace_not_whitespace(clinical_trials_client):
    """Test detect_whitespace when exact matches exist (is_whitespace=False).

    When trials exist for the drug-indication pair, is_whitespace=False
    and indication_drugs is empty (no need to show competitors).
    """
    # Semaglutide + diabetes = NOT whitespace (many exact matches)
    result = await clinical_trials_client.detect_whitespace("semaglutide", "diabetes")

    # Verify WhitespaceResult fields for non-whitespace case
    assert result.is_whitespace is False
    assert result.exact_match_count >= 10  # semaglutide + diabetes has many trials
    assert 400 < result.drug_only_trials < 800
    assert 10000 < result.indication_only_trials < 100000

    # indication_drugs should be empty when not whitespace
    assert result.indication_drugs == []


# --- Edge cases and weird inputs ---


async def test_get_trial_nonexistent_nct_id_raises_error(clinical_trials_client):
    """Test that a nonexistent NCT ID raises DataSourceError."""
    with pytest.raises(DataSourceError, match="404") as exc_info:
        await clinical_trials_client.get_trial("NCT99999999")

    assert "NCT99999999" in str(exc_info.value)


@pytest.mark.parametrize(
    "drug, indication, expect_empty",
    [
        ("xyzzy_not_a_real_drug_12345", "diabetes", True),
        ("semaglutide", "xyzzy_fake_disease_99999", True),
        # Empty drug acts as no filter — may return results; just verify no error.
        ("", "diabetes", False),
    ],
    ids=["nonexistent_drug", "nonexistent_indication", "empty_drug"],
)
async def test_search_trials_invalid_input_returns_list(
    clinical_trials_client, drug, indication, expect_empty
):
    """search_trials returns a list (empty for unresolvable terms) rather than raising."""
    trials = await clinical_trials_client.search_trials(
        drug=drug,
        indication=indication,
    )
    assert isinstance(trials, list)
    if expect_empty:
        assert trials == []


async def test_get_landscape_nonexistent_indication_returns_empty(
    clinical_trials_client,
):
    """Test that nonexistent indication returns empty landscape."""
    landscape = await clinical_trials_client.get_landscape(
        "xyzzy_fake_indication_99999", top_n=10
    )

    assert landscape.total_trial_count == 0
    assert landscape.competitors == []
    assert landscape.phase_distribution == {}


async def test_get_terminated_nonexistent_query_returns_empty(clinical_trials_client):
    """Test that nonexistent drug and indication returns empty TrialOutcomes."""
    outcomes = await clinical_trials_client.get_terminated(
        "xyzzy_not_a_real_term_12345",
        "xyzzy_not_a_real_indication_12345",
    )

    assert outcomes == TrialOutcomes()


async def test_get_terminated_business_trial_excluded_from_drug_query(
    clinical_trials_client,
):
    """drug_wide applies a safety/efficacy filter that drops business terminations.

    NCT04012255 (semaglutide, Overweight, Phase 1) was terminated for strategic/business
    reasons. It must not appear in drug_wide (wrong stop_category), but it is retained
    in pair_specific by design (that scope keeps all stop_categories).
    """
    outcomes = await clinical_trials_client.get_terminated(
        "semaglutide", "type 2 diabetes"
    )

    drug_wide_ids = {t.nct_id for t in outcomes.drug_wide}
    assert "NCT04012255" not in drug_wide_ids


@pytest.mark.parametrize(
    "nct_id, why_stopped_fragment, expected_category",
    [
        (
            "NCT00109577",
            "no adverse events",  # negated safety phrase
            "other",
        ),
        (
            "NCT06134661",
            "unrelated to safety",  # negated safety phrase
            "enrollment",
        ),
    ],
)
async def test_classify_stop_reason_negation_on_live_data(
    clinical_trials_client, nct_id, why_stopped_fragment, expected_category
):
    """_classify_stop_reason does not misclassify negated safety phrases as 'safety'.

    Fetches real terminated trials whose why_stopped text contains a negated
    safety phrase and asserts the category is not 'safety'.
    """
    from indication_scout.data_sources.clinical_trials import _classify_stop_reason

    trial = await clinical_trials_client.get_trial(nct_id)

    assert trial.why_stopped is not None
    assert why_stopped_fragment in trial.why_stopped.lower()
    assert _classify_stop_reason(trial.why_stopped) == expected_category
    assert _classify_stop_reason(trial.why_stopped) != "safety"


async def test_get_landscape_total_count_exceeds_fetch_cap(clinical_trials_client):
    """total_trial_count reflects the real API count, not the number of fetched trials.

    CLINICAL_TRIALS_LANDSCAPE_MAX_TRIALS caps the fetch at 50 trials, but the API
    reports the true total. For type 2 diabetes, thousands of trials exist.
    """
    from indication_scout.config import get_settings

    landscape = await clinical_trials_client.get_landscape("type 2 diabetes", top_n=5)

    assert landscape.total_trial_count > get_settings().clinical_trials_landscape_max_trials


async def test_detect_whitespace_nonexistent_drug_is_whitespace(clinical_trials_client):
    """Test that nonexistent drug returns is_whitespace=True."""
    result = await clinical_trials_client.detect_whitespace(
        "xyzzy_fake_drug_99999", "diabetes"
    )

    assert result.is_whitespace is True
    assert result.exact_match_count == 0
    assert result.drug_only_trials == 0
    # Indication-only trials should still exist for a real indication
    assert result.indication_only_trials > 0


async def test_detect_whitespace_nonexistent_indication_is_whitespace(
    clinical_trials_client,
):
    """Test that nonexistent indication returns is_whitespace=True."""
    result = await clinical_trials_client.detect_whitespace(
        "semaglutide", "xyzzy_fake_disease_99999"
    )

    assert result.is_whitespace is True
    assert result.exact_match_count == 0
    # Drug-only trials should still exist for a real drug
    assert result.drug_only_trials > 0
    assert result.indication_only_trials == 0
    assert result.indication_drugs == []


# --- MeSH post-filter (Phase 2) ---


async def test_search_trials_mesh_filter_drops_noise(clinical_trials_client):
    """target_mesh_id drops trials whose MeSH tags don't include the target.

    A loose CT.gov search for metformin + "hypertension" returns Essie noise:
    pre-eclampsia, pulmonary hypertension, prehypertension, masked hypertension.
    Post-filtering by D006973 ("Hypertension") keeps only trials whose
    mesh_conditions OR mesh_ancestors contain D006973.

    Also covers the ancestor-only case: NCT02503943 has mesh_conditions =
    [Masked Hypertension (D059468), ...] and mesh_ancestors includes
    Hypertension (D006973); the post-filter must keep it.
    """
    unfiltered = await clinical_trials_client.search_trials(
        drug="metformin", indication="hypertension"
    )
    filtered = await clinical_trials_client.search_trials(
        drug="metformin", indication="hypertension", target_mesh_id="D006973"
    )

    # Filter must be strictly narrowing on this known-noisy query
    assert len(filtered) < len(unfiltered)
    assert len(filtered) > 0

    # Every kept trial contains D006973 in mesh_conditions or mesh_ancestors
    for t in filtered:
        cond_ids = {m.id for m in t.mesh_conditions}
        anc_ids = {m.id for m in t.mesh_ancestors}
        assert "D006973" in cond_ids | anc_ids, (
            f"{t.nct_id} passed filter but has no D006973 tag"
        )

    # Every dropped trial lacks D006973 (or has no MeSH tags at all)
    kept_ids = {t.nct_id for t in filtered}
    dropped = [t for t in unfiltered if t.nct_id not in kept_ids]
    assert len(dropped) > 0  # must be noise to drop
    for t in dropped:
        cond_ids = {m.id for m in t.mesh_conditions}
        anc_ids = {m.id for m in t.mesh_ancestors}
        assert "D006973" not in cond_ids | anc_ids, (
            f"{t.nct_id} was dropped but has D006973 — filter bug"
        )

    # Ancestor-only spot check: NCT02503943 is kept via mesh_ancestors alone.
    [ancestor_only] = [t for t in filtered if t.nct_id == "NCT02503943"]
    ancestor_cond_ids = {m.id for m in ancestor_only.mesh_conditions}
    ancestor_anc_ids = {m.id for m in ancestor_only.mesh_ancestors}
    assert "D006973" not in ancestor_cond_ids
    assert "D006973" in ancestor_anc_ids


async def test_count_trials_with_mesh_narrows_to_real_matches(clinical_trials_client):
    """_count_trials with target_mesh_id returns post-filter count, not API totalCount.

    "huntington disease" on CT.gov returns ~300 trials total — Essie sweeps in some
    unrelated hits. D006816 ("Huntington Disease") post-filtering narrows the set
    to trials genuinely tagged with that MeSH id in conditions or ancestors.
    """
    unfiltered = await clinical_trials_client._count_trials(
        drug=None, indication="huntington disease"
    )
    filtered = await clinical_trials_client._count_trials(
        drug=None,
        indication="huntington disease",
        target_mesh_id="D006816",
    )

    # Filter is strictly no-wider than the unfiltered total
    assert filtered <= unfiltered
    # And produces a meaningful number of matches for a real disease
    assert filtered > 100
    # Unfiltered totalCount includes Essie noise, so baseline must be large
    assert unfiltered > 200


async def test_get_landscape_with_mesh_filter_narrows(clinical_trials_client):
    """get_landscape with target_mesh_id fetches unbounded pre-filter then caps post-filter.

    Covers Fix #1: for a noisy indication like "hypertension", the unfiltered
    Essie query returns pulmonary / portal / pre-eclampsia hypertension trials.
    With target_mesh_id=D006973 ("Hypertension"), the MeSH filter must be
    applied BEFORE the landscape cap — so every competitor's representative
    trial set must carry D006973 in mesh_conditions or mesh_ancestors, and the
    filtered total_trial_count must be <= the unfiltered one.
    """
    unfiltered = await clinical_trials_client.get_landscape(
        "hypertension", top_n=10
    )
    filtered = await clinical_trials_client.get_landscape(
        "hypertension", top_n=10, target_mesh_id="D006973"
    )

    # Both queries return competitors
    assert len(unfiltered.competitors) > 0
    assert len(filtered.competitors) > 0

    # Filter can only narrow the count
    assert filtered.total_trial_count <= unfiltered.total_trial_count

    # For a truly noisy query, filter should produce a strictly smaller count
    assert filtered.total_trial_count < unfiltered.total_trial_count


async def test_count_trials_mesh_saturation_on_broad_indication(
    clinical_trials_client, caplog
):
    """_count_trials with target_mesh_id on a broad indication hits the page cap.

    Covers Fix #3: "neoplasms" has tens of thousands of trials on CT.gov. An
    uncapped walk would trigger hundreds of HTTP calls; the page cap must stop
    the walk at CLINICAL_TRIALS_COUNT_PAGE_CAP pages and log saturation. We
    can't assert an exact count, but we assert:
      - the call returns a positive int (not a timeout / exception)
      - the count is strictly less than the API's unfiltered totalCount
      - the saturation warning fires
    """
    from indication_scout.constants import CLINICAL_TRIALS_COUNT_PAGE_CAP

    unfiltered = await clinical_trials_client._count_trials(
        drug=None, indication="neoplasms"
    )
    assert unfiltered > 10000, (
        f"expected 'neoplasms' to return >10k trials, got {unfiltered}"
    )

    with caplog.at_level(logging.WARNING, logger="indication_scout.data_sources.clinical_trials"):
        filtered = await clinical_trials_client._count_trials(
            drug=None,
            indication="neoplasms",
            target_mesh_id="D009369",  # Neoplasms
        )

    # Count is bounded by the page cap: at most CAP * PAGE_SIZE (100) survivors
    assert 0 < filtered <= CLINICAL_TRIALS_COUNT_PAGE_CAP * 100
    # Saturation log fired (cap was hit on such a broad indication)
    assert any(
        "page cap" in rec.message and "lower bound" in rec.message
        for rec in caplog.records
    ), f"expected saturation warning; got {[r.message for r in caplog.records]}"


