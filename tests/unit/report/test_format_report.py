"""Unit tests for the markdown report formatter.

Covers the clinical-trials section rewrite (search / completed / terminated /
landscape / approval) and the top-level format_report assembly.
"""

from indication_scout.agents.clinical_trials.clinical_trials_output import (
    ClinicalTrialsOutput,
)
from indication_scout.agents.literature.literature_output import LiteratureOutput
from indication_scout.agents.mechanism.mechanism_output import (
    MechanismCandidate,
    MechanismOutput,
)
from indication_scout.agents.supervisor.supervisor_output import (
    CandidateFindings,
    SupervisorOutput,
)
from indication_scout.models.model_clinical_trials import (
    ApprovalCheck,
    CompetitorEntry,
    CompletedTrialsResult,
    IndicationLandscape,
    SearchTrialsResult,
    TerminatedTrialsResult,
    Trial,
)
from indication_scout.models.model_evidence_summary import EvidenceSummary
from indication_scout.report.format_report import (
    _fmt_clinical_trials,
    format_report,
)


def test_fmt_clinical_trials_empty_returns_placeholder():
    out = ClinicalTrialsOutput()
    rendered = _fmt_clinical_trials(out)
    assert rendered == "_No clinical trials data available._"


def test_fmt_clinical_trials_summary_only():
    out = ClinicalTrialsOutput(summary="Drug already approved for this indication.")
    rendered = _fmt_clinical_trials(out)
    assert rendered == "Drug already approved for this indication."


def test_fmt_clinical_trials_approval_is_approved():
    out = ClinicalTrialsOutput(
        approval=ApprovalCheck(
            is_approved=True,
            label_found=True,
            matched_indication="Type 2 Diabetes Mellitus",
            drug_names_checked=["semaglutide", "ozempic"],
        )
    )
    rendered = _fmt_clinical_trials(out)
    assert "**FDA approval:** Approved (Type 2 Diabetes Mellitus)" in rendered


def test_fmt_clinical_trials_approval_label_found_not_approved():
    out = ClinicalTrialsOutput(
        approval=ApprovalCheck(
            is_approved=False,
            label_found=True,
            matched_indication=None,
            drug_names_checked=["semaglutide"],
        )
    )
    rendered = _fmt_clinical_trials(out)
    assert "**FDA approval:** Not found on FDA label for this indication" in rendered


def test_fmt_clinical_trials_approval_no_label():
    out = ClinicalTrialsOutput(
        approval=ApprovalCheck(
            is_approved=False,
            label_found=False,
            matched_indication=None,
            drug_names_checked=["aducanumab", "aduhelm"],
        )
    )
    rendered = _fmt_clinical_trials(out)
    assert "No FDA label found for aducanumab, aduhelm" in rendered
    assert "status undetermined" in rendered


def test_fmt_clinical_trials_search_with_status_breakdown():
    out = ClinicalTrialsOutput(
        search=SearchTrialsResult(
            total_count=12,
            by_status={"RECRUITING": 5, "ACTIVE_NOT_RECRUITING": 4, "WITHDRAWN": 3},
            trials=[],
        )
    )
    rendered = _fmt_clinical_trials(out)
    assert "**Trial activity:** 12 total trial(s) for this pair" in rendered
    assert "5 recruiting" in rendered
    assert "4 active not recruiting" in rendered
    assert "3 withdrawn" in rendered


def test_fmt_clinical_trials_search_whitespace():
    out = ClinicalTrialsOutput(
        search=SearchTrialsResult(total_count=0, by_status={}, trials=[])
    )
    rendered = _fmt_clinical_trials(out)
    assert "**Trial activity:** 0 total trial(s) for this pair" in rendered
    assert "Whitespace: no trials found for this drug × indication pair." in rendered


def test_fmt_clinical_trials_completed_renders_count_and_top_trials():
    trial = Trial(
        nct_id="NCT04567890",
        title="Semaglutide in NASH",
        phase="Phase 3",
        overall_status="Completed",
    )
    out = ClinicalTrialsOutput(
        completed=CompletedTrialsResult(
            total_count=7,
            phase3_count=2,
            trials=[trial],
        )
    )
    rendered = _fmt_clinical_trials(out)
    assert "**Completed trials (7 total, 2 Phase 3):**" in rendered
    assert (
        "[NCT04567890](https://clinicaltrials.gov/study/NCT04567890) — Semaglutide in NASH (Phase 3, Completed)"
        in rendered
    )


def test_fmt_clinical_trials_completed_caps_at_ten():
    trials = [
        Trial(nct_id=f"NCT{i:08d}", title=f"Trial {i}", phase="Phase 2", overall_status="Completed")
        for i in range(15)
    ]
    out = ClinicalTrialsOutput(
        completed=CompletedTrialsResult(total_count=15, phase3_count=0, trials=trials)
    )
    rendered = _fmt_clinical_trials(out)
    assert "NCT00000009" in rendered
    assert "NCT00000010" not in rendered


def test_fmt_clinical_trials_terminated_with_why_stopped():
    trial = Trial(
        nct_id="NCT01112233",
        title="Cardio Trial",
        phase="Phase 2",
        why_stopped="Sponsor decision due to slow enrollment",
    )
    out = ClinicalTrialsOutput(
        terminated=TerminatedTrialsResult(total_count=1, trials=[trial])
    )
    rendered = _fmt_clinical_trials(out)
    assert "**Terminated trials (1):**" in rendered
    assert (
        "[NCT01112233](https://clinicaltrials.gov/study/NCT01112233) Cardio Trial (Phase 2)"
        " [enrollment] — *Sponsor decision due to slow enrollment*"
        in rendered
    )


def test_fmt_clinical_trials_terminated_unknown_when_no_why_stopped():
    trial = Trial(nct_id="NCT09998888", title="Mystery Stop", phase="Phase 1")
    out = ClinicalTrialsOutput(
        terminated=TerminatedTrialsResult(total_count=1, trials=[trial])
    )
    rendered = _fmt_clinical_trials(out)
    assert (
        "[NCT09998888](https://clinicaltrials.gov/study/NCT09998888) Mystery Stop (Phase 1) [unknown]"
        in rendered
    )


def test_fmt_clinical_trials_terminated_zero_count_renders_nothing_for_section():
    out = ClinicalTrialsOutput(
        terminated=TerminatedTrialsResult(total_count=0, trials=[])
    )
    rendered = _fmt_clinical_trials(out)
    assert "Terminated trials" not in rendered


def test_fmt_clinical_trials_landscape_renders_competitors():
    out = ClinicalTrialsOutput(
        landscape=IndicationLandscape(
            total_trial_count=42,
            competitors=[
                CompetitorEntry(
                    sponsor="Novo Nordisk",
                    drug_name="Semaglutide",
                    max_phase="Phase 3",
                    trial_count=8,
                ),
                CompetitorEntry(
                    sponsor="Eli Lilly",
                    drug_name="Tirzepatide",
                    max_phase="Phase 3",
                    trial_count=5,
                ),
            ],
        )
    )
    rendered = _fmt_clinical_trials(out)
    assert "**Competitive landscape (2 competitors):**" in rendered
    assert "Semaglutide (Novo Nordisk) — Phase 3, 8 trial(s)" in rendered
    assert "Tirzepatide (Eli Lilly) — Phase 3, 5 trial(s)" in rendered


def test_format_report_full_assembly():
    """End-to-end: confirms all sections render, ordering is correct, and the
    mechanism-only-uninvestigated tail surfaces a candidate not in findings."""
    output = SupervisorOutput(
        drug_name="semaglutide",
        candidates=["NASH", "Alzheimer's disease"],
        summary="Semaglutide shows promise for NASH; Alzheimer's evidence is weaker.",
        mechanism=MechanismOutput(
            summary="GLP-1 receptor agonist with metabolic and CNS effects.",
            drug_targets={"GLP1R": "ENSG00000112164"},
            candidates=[
                MechanismCandidate(
                    target_symbol="GLP1R",
                    action_type="AGONIST",
                    disease_name="NASH",
                    disease_description="Non-alcoholic steatohepatitis.",
                    target_function="GLP-1 receptor.",
                ),
                MechanismCandidate(
                    target_symbol="GLP1R",
                    action_type="AGONIST",
                    disease_name="Parkinson's disease",
                    disease_description="Neurodegenerative disorder.",
                    target_function="GLP-1 receptor.",
                ),
            ],
        ),
        findings=[
            CandidateFindings(
                disease="NASH",
                source="both",
                literature=LiteratureOutput(
                    evidence_summary=EvidenceSummary(
                        summary="Multiple Phase 2 trials show histological improvement.",
                        study_count=4,
                        strength="moderate",
                        key_findings=["MASH resolution in ~60% of patients"],
                        supporting_pmids=["12345678"],
                    )
                ),
                clinical_trials=ClinicalTrialsOutput(
                    summary="Active development pipeline.",
                    search=SearchTrialsResult(
                        total_count=5,
                        by_status={"RECRUITING": 2, "ACTIVE_NOT_RECRUITING": 3},
                        trials=[],
                    ),
                ),
            ),
        ],
    )

    rendered = format_report(output)

    assert "# IndicationScout Report: semaglutide" in rendered
    assert "## Summary" in rendered
    assert "Semaglutide shows promise for NASH" in rendered
    assert "## Candidate Diseases" in rendered
    assert "- NASH" in rendered
    assert "- Alzheimer's disease" in rendered
    assert "## Mechanistic Analysis" in rendered
    assert "**Molecular targets:** GLP1R" in rendered
    assert "GLP1R (AGONIST) → NASH" in rendered
    assert "## Candidate Findings" in rendered
    assert "### NASH _(source: both)_" in rendered
    assert "#### Literature" in rendered
    assert "**Evidence strength:** moderate" in rendered
    assert "[12345678](https://pubmed.ncbi.nlm.nih.gov/12345678/)" in rendered
    assert "#### Clinical Trials" in rendered
    assert "**Trial activity:** 5 total trial(s) for this pair" in rendered
    assert "### Other mechanism candidates (promoted, not investigated)" in rendered
    assert "- Parkinson's disease" in rendered


def test_format_report_no_candidates_or_findings():
    output = SupervisorOutput(drug_name="metformin")
    rendered = format_report(output)
    assert "# IndicationScout Report: metformin" in rendered
    assert "_No summary produced._" in rendered
    assert "_No candidates surfaced._" in rendered
    assert "_Mechanism analysis not run._" in rendered
    assert "_No candidate findings produced._" in rendered


def test_format_report_unknown_drug_name_default():
    output = SupervisorOutput()
    rendered = format_report(output)
    assert "# IndicationScout Report: Unknown drug" in rendered
