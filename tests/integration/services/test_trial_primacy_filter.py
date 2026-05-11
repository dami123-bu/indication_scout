"""Integration test for services.trial_primacy_filter — hits real Haiku."""

import logging

from indication_scout.models.model_clinical_trials import MeshTerm, Trial
from indication_scout.services.trial_primacy_filter import classify_trials_by_primacy

logger = logging.getLogger(__name__)


def _trial(nct_id: str, title: str, indications: list[str], mesh: list[str]) -> Trial:
    return Trial(
        nct_id=nct_id,
        title=title,
        indications=indications,
        mesh_conditions=[MeshTerm(id="", term=m) for m in mesh],
    )


async def test_classify_drops_pcos_primary_for_cardiovascular_query():
    """Real Haiku must drop NCT00842140 (PCOS contraceptives safety, with CV
    only as MeSH enrichment) when the query is cardiovascular disease, and
    keep a real CV-primary trial (NCT07209527).

    Reproduces the case the user surfaced: CT.gov's
    `AREA[ConditionMeshTerm]"Cardiovascular Diseases"` returns NCT00842140
    because the trial's derived mesh includes CV — but the trial's sponsor-
    declared condition is only PCOS, so the LLM should classify it as
    SECONDARY.
    """
    pcos_trial = _trial(
        "NCT00842140",
        title="Safety of Hormonal Contraceptives Therapies in Polycystic Ovary Syndrome (PCOS) Women",
        indications=["Polycystic Ovary Syndrome"],
        mesh=["Polycystic Ovary Syndrome", "Cardiovascular Diseases"],
    )
    real_cv_trial = _trial(
        "NCT07209527",
        title="Truway Diagnostic Tools in Primary Care",
        indications=[
            "Chronic Disease",
            "Type 2 Diabetes Mellitus (T2DM), Obesity, Overweight",
            "Hypertension",
            "Cardiovascular Disease Acute",
            "Hyperlipidemia",
            "Peripheral Artery Disease",
            "Metabolic Syndrome",
            "Prediabetes",
        ],
        mesh=["Chronic Disease", "Diabetes Mellitus, Type 2", "Cardiovascular Diseases"],
    )

    drop_set = await classify_trials_by_primacy(
        trials=[pcos_trial, real_cv_trial],
        queried_indication="cardiovascular disease",
        mesh_term="Cardiovascular Diseases",
    )

    assert "NCT00842140" in drop_set
    assert "NCT07209527" not in drop_set


async def test_classify_drops_pcos_primary_for_gestational_diabetes_query():
    """Real Haiku must drop NCT00994812 (PCOS/miscarriage trial that lists
    Gestational Diabetes as a tracked condition, far from primary focus).
    """
    pcos_gd_trial = _trial(
        "NCT00994812",
        title="The Effects of Metformin on Pregnancy and Miscarriage Rates in "
        "Polycystic Ovary Syndrome (PCOS)",
        indications=[
            "Polycystic Ovary Syndrome",
            "Miscarriage",
            "Infertility",
            "Toxemia",
            "Gestational Diabetes",
        ],
        mesh=[
            "Polycystic Ovary Syndrome",
            "Abortion, Spontaneous",
            "Infertility",
            "Toxemia",
            "Diabetes, Gestational",
            "Pregnancy Complications",
        ],
    )
    real_gd_trial = _trial(
        "NCT05479214",
        title="Effect of Adding Metformin to Insulin in Uncontrolled Diabetic Patients During the "
        "3rd Trimester of Pregnancy",
        indications=["Diabetes Mellitus Pregnancy"],
        mesh=["Diabetes, Gestational"],
    )

    drop_set = await classify_trials_by_primacy(
        trials=[pcos_gd_trial, real_gd_trial],
        queried_indication="gestational diabetes",
        mesh_term="Diabetes, Gestational",
    )

    assert "NCT00994812" in drop_set
    assert "NCT05479214" not in drop_set


async def test_classify_handles_synonyms_keeps_nash_trial_for_nafld_query():
    """Real Haiku must keep NCT02970942 (semaglutide for non-alcoholic
    steatohepatitis) even though the sponsor wrote 'Non-alcoholic
    Steatohepatitis' rather than the exact preferred term 'Non-alcoholic
    Fatty Liver Disease' — NASH and NAFLD are synonyms in MeSH (NASH is an
    entry term under D065626). A regex-only filter would drop this.
    """
    nash_trial = _trial(
        "NCT02970942",
        title="Investigation of Efficacy and Safety of Three Dose Levels of Subcutaneous "
        "Semaglutide Once Daily Versus Placebo in Subjects With Non-alcoholic Steatohepatitis.",
        indications=["Hepatobiliary Disorders", "Non-alcoholic Steatohepatitis"],
        mesh=["Digestive System Diseases", "Non-alcoholic Fatty Liver Disease"],
    )

    drop_set = await classify_trials_by_primacy(
        trials=[nash_trial],
        queried_indication="nash",
        mesh_term="Non-alcoholic Fatty Liver Disease",
    )

    assert "NCT02970942" not in drop_set
