"""IndicationScout — Streamlit frontend MOCK.

Layout preview only. All data is hardcoded — no agents, no DB, no API calls.

Run with:
    streamlit run app_mock.py
"""

import streamlit as st

st.set_page_config(page_title="IndicationScout (mock)", page_icon="static/favicon.png", layout="wide")


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

MOCK_DRUG = "Metformin"

MOCK_SUMMARY = (
    "Metformin shows mechanistically plausible repurposing potential across three "
    "oncology and metabolic indications, supported by 47 published studies and 12 "
    "active clinical trials. Strongest signal: pancreatic cancer (Phase 2 ongoing, "
    "AMPK pathway rationale)."
)

MOCK_CANDIDATES = [
    "Pancreatic cancer",
    "Polycystic ovary syndrome (PCOS)",
    "Non-alcoholic fatty liver disease (NAFLD)",
    "Glioblastoma",
    "Aging / longevity",
]

MOCK_MECHANISM = {
    "summary": (
        "Metformin's primary mechanism is activation of AMP-activated protein kinase "
        "(AMPK), with secondary effects on mitochondrial complex I and mTOR signaling. "
        "These pathways are dysregulated in multiple cancers and metabolic disorders."
    ),
    "targets": ["PRKAA1", "PRKAA2", "MTOR", "ETC-Complex-I"],
    "moa": [
        {"mechanism": "AMPK activator", "action": "AGONIST", "targets": "PRKAA1, PRKAA2"},
        {"mechanism": "Mitochondrial complex I inhibitor", "action": "INHIBITOR", "targets": "MT-ND1"},
        {"mechanism": "mTOR pathway suppression (indirect)", "action": "INHIBITOR", "targets": "MTOR"},
    ],
    "candidates": [
        {
            "target": "PRKAA1",
            "action": "AGONIST",
            "disease": "Pancreatic cancer",
            "disease_desc": "AMPK activation suppresses pancreatic tumor growth via mTOR inhibition.",
            "target_function": "Catalytic subunit of AMPK; energy-sensing kinase.",
        },
        {
            "target": "MTOR",
            "action": "INHIBITOR",
            "disease": "Glioblastoma",
            "disease_desc": "mTOR hyperactivation is common in GBM; indirect suppression may slow proliferation.",
            "target_function": "Master regulator of cell growth and protein synthesis.",
        },
    ],
}

MOCK_FINDINGS = {
    "Pancreatic cancer": {
        "source": "mechanism",
        "literature": {
            "strength": "Moderate",
            "study_count": 18,
            "summary": (
                "Multiple retrospective cohort studies suggest improved survival in diabetic "
                "pancreatic cancer patients on metformin. Two small Phase 2 trials show "
                "modest benefit when combined with standard chemotherapy."
            ),
            "key_findings": [
                "Diabetic patients on metformin had 15% improved 2-year OS in a 2019 meta-analysis.",
                "AMPK activation reduces pancreatic stellate cell desmoplasia in PDX models.",
                "Phase 2 combo with gemcitabine showed PFS benefit (HR 0.76, p=0.04).",
            ],
            "pmids": ["31234567", "30123456", "29876543", "32109876", "33445566"],
        },
        "clinical_trials": {
            "summary": "Active development with 4 completed and 3 ongoing trials, mostly Phase 2 combinations.",
            "approval": {"status": "not_approved", "label": "Not found on FDA label for pancreatic cancer"},
            "search": {"total": 12, "by_status": {"COMPLETED": 4, "RECRUITING": 3, "TERMINATED": 2, "UNKNOWN": 3}},
            "completed": [
                {"nct": "NCT01210911", "title": "Metformin + Gemcitabine in Advanced PDAC", "phase": "Phase 2", "status": "COMPLETED"},
                {"nct": "NCT02005419", "title": "Metformin in Resectable Pancreatic Cancer", "phase": "Phase 2", "status": "COMPLETED"},
                {"nct": "NCT01666730", "title": "FOLFIRINOX + Metformin", "phase": "Phase 1/2", "status": "COMPLETED"},
            ],
            "terminated": [
                {"nct": "NCT01971034", "title": "Metformin Adjuvant Therapy", "phase": "Phase 3", "reason": "Slow accrual", "category": "non_safety"},
            ],
            "competitors": [
                {"drug": "Pembrolizumab", "sponsor": "Merck", "phase": "Phase 3", "trials": 24},
                {"drug": "Olaparib", "sponsor": "AstraZeneca", "phase": "Phase 3", "trials": 11},
            ],
        },
    },
    "Polycystic ovary syndrome (PCOS)": {
        "source": "literature",
        "literature": {
            "strength": "Strong",
            "study_count": 22,
            "summary": (
                "Metformin is widely studied in PCOS with consistent benefit on insulin "
                "sensitivity, ovulation rates, and menstrual regularity."
            ),
            "key_findings": [
                "Cochrane review confirms improved ovulation vs placebo (OR 2.12).",
                "Combination with clomiphene increases live birth rate.",
            ],
            "pmids": ["28765432", "27654321", "26543210"],
        },
        "clinical_trials": {
            "summary": "Mature trial landscape with established efficacy signals.",
            "approval": {"status": "not_approved", "label": "Off-label widely used; not on FDA label for PCOS"},
            "search": {"total": 38, "by_status": {"COMPLETED": 22, "RECRUITING": 5, "TERMINATED": 3, "UNKNOWN": 8}},
            "completed": [
                {"nct": "NCT00188331", "title": "Metformin vs Clomiphene in PCOS", "phase": "Phase 3", "status": "COMPLETED"},
            ],
            "terminated": [],
            "competitors": [
                {"drug": "Inositol", "sponsor": "Various", "phase": "Phase 3", "trials": 18},
            ],
        },
    },
    "Glioblastoma": {
        "source": "mechanism",
        "literature": {
            "strength": "Weak",
            "study_count": 7,
            "summary": "Limited preclinical data; one small pilot trial inconclusive.",
            "key_findings": [
                "In vitro AMPK activation reduces GBM cell proliferation.",
                "Single-arm Phase 1 (n=12) showed acceptable safety but no efficacy signal.",
            ],
            "pmids": ["34567890", "33456789"],
        },
        "clinical_trials": {
            "summary": "Sparse trial activity; mostly investigator-initiated pilots.",
            "approval": {"status": "no_label", "label": "No FDA label found for glioblastoma indication"},
            "search": {"total": 3, "by_status": {"COMPLETED": 1, "TERMINATED": 1, "UNKNOWN": 1}},
            "completed": [
                {"nct": "NCT02780024", "title": "Metformin Pilot in Recurrent GBM", "phase": "Phase 1", "status": "COMPLETED"},
            ],
            "terminated": [
                {"nct": "NCT01430351", "title": "Metformin + TMZ", "phase": "Phase 2", "reason": "Lack of efficacy at interim", "category": "efficacy"},
            ],
            "competitors": [],
        },
    },
}


# ---------------------------------------------------------------------------
# Sidebar — inputs and run control
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("static/favicon.png", width=60)
    st.markdown("## Indication Scout")
    st.caption("Drug repurposing analysis powered by AI agents.")

    st.markdown("---")

    drug_input = st.text_input(
        "Drug name",
        value=MOCK_DRUG,
        placeholder="e.g. metformin",
        help="Enter the name of the drug to analyse for repurposing opportunities.",
    )

    run_clicked = st.button("Analyse", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("### Disease focus")
    st.caption("Select a candidate disease to drill into clinical & literature evidence.")

    investigated = list(MOCK_FINDINGS.keys())
    selected_disease = st.radio(
        "Investigated diseases",
        options=investigated,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.download_button(
        label="Download report (.md)",
        data="# Mock report\n\nThis is a placeholder.",
        file_name=f"indication_scout_{MOCK_DRUG.lower()}_mock.md",
        mime="text/markdown",
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Main area — header and tabs
# ---------------------------------------------------------------------------

st.title(f"Indication Scout — {MOCK_DRUG}")
st.caption("_Generated 2026-04-27 14:31 UTC_ · **MOCK PREVIEW** (no agents run)")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Candidate diseases", len(MOCK_CANDIDATES))
m2.metric("Investigated", len(MOCK_FINDINGS))
m3.metric("Total trials", sum(f["clinical_trials"]["search"]["total"] for f in MOCK_FINDINGS.values()))
m4.metric("Total studies", sum(f["literature"]["study_count"] for f in MOCK_FINDINGS.values()))

st.markdown("---")

tab_overview, tab_mech, tab_trials, tab_lit = st.tabs(
    ["Overview", "Mechanism", "Clinical Trials", "Literature"]
)


# ----- Overview tab -----
with tab_overview:
    st.markdown("### Summary")
    st.write(MOCK_SUMMARY)

    st.markdown("### Candidate diseases")
    for c in MOCK_CANDIDATES:
        investigated_marker = "✓ investigated" if c in MOCK_FINDINGS else "_not investigated_"
        st.markdown(f"- **{c}** — {investigated_marker}")

    st.info(
        f"Use the sidebar to switch the focus disease. Currently viewing: "
        f"**{selected_disease}** in the Clinical Trials and Literature tabs."
    )


# ----- Mechanism tab -----
with tab_mech:
    st.markdown("### Mechanistic analysis")
    st.write(MOCK_MECHANISM["summary"])

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("#### Molecular targets")
        for t in MOCK_MECHANISM["targets"]:
            st.markdown(f"- `{t}`")

    with col_b:
        st.markdown("#### Mechanisms of action")
        st.dataframe(
            MOCK_MECHANISM["moa"],
            hide_index=True,
            use_container_width=True,
            column_config={
                "mechanism": "Mechanism",
                "action": "Action type",
                "targets": "Targets",
            },
        )

    st.markdown("#### Repurposing candidates from mechanism")
    for c in MOCK_MECHANISM["candidates"]:
        with st.container(border=True):
            st.markdown(f"**{c['target']} ({c['action']}) → {c['disease']}**")
            st.caption(c["disease_desc"])
            st.caption(f"_Target function:_ {c['target_function']}")


# ----- Clinical Trials tab -----
with tab_trials:
    finding = MOCK_FINDINGS[selected_disease]
    ct = finding["clinical_trials"]

    st.markdown(f"### Clinical trials — {selected_disease}")
    st.caption(f"Source: {finding['source']}")

    st.write(ct["summary"])

    a1, a2, a3 = st.columns(3)
    a1.metric("Total trials", ct["search"]["total"])
    a1.caption("for this drug × indication")
    a2.metric("Completed", ct["search"]["by_status"].get("COMPLETED", 0))
    a3.metric("Recruiting", ct["search"]["by_status"].get("RECRUITING", 0))

    approval = ct["approval"]
    if approval["status"] == "approved":
        st.success(f"**FDA approval:** {approval['label']}")
    elif approval["status"] == "not_approved":
        st.warning(f"**FDA approval:** {approval['label']}")
    else:
        st.info(f"**FDA approval:** {approval['label']}")

    st.markdown("#### Status breakdown")
    st.bar_chart(ct["search"]["by_status"], horizontal=True)

    if ct["completed"]:
        st.markdown(f"#### Completed trials ({len(ct['completed'])})")
        completed_rows = [
            {
                "NCT": f"[{t['nct']}](https://clinicaltrials.gov/study/{t['nct']})",
                "Title": t["title"],
                "Phase": t["phase"],
                "Status": t["status"],
            }
            for t in ct["completed"]
        ]
        st.dataframe(completed_rows, hide_index=True, use_container_width=True,
                     column_config={"NCT": st.column_config.LinkColumn("NCT", display_text=r"NCT\d+")})

    if ct["terminated"]:
        st.markdown(f"#### Terminated trials ({len(ct['terminated'])})")
        for t in ct["terminated"]:
            with st.container(border=True):
                st.markdown(
                    f"[{t['nct']}](https://clinicaltrials.gov/study/{t['nct']}) — {t['title']}"
                )
                st.caption(f"{t['phase']} · **{t['category']}** · _{t['reason']}_")

    if ct["competitors"]:
        st.markdown(f"#### Competitive landscape ({len(ct['competitors'])})")
        st.dataframe(
            ct["competitors"],
            hide_index=True,
            use_container_width=True,
            column_config={
                "drug": "Drug",
                "sponsor": "Sponsor",
                "phase": "Max phase",
                "trials": st.column_config.NumberColumn("Trials"),
            },
        )


# ----- Literature tab -----
with tab_lit:
    finding = MOCK_FINDINGS[selected_disease]
    lit = finding["literature"]

    st.markdown(f"### Literature — {selected_disease}")
    st.caption(f"Source: {finding['source']}")

    l1, l2 = st.columns(2)
    l1.metric("Evidence strength", lit["strength"])
    l2.metric("Study count", lit["study_count"])

    st.markdown("#### Summary")
    st.write(lit["summary"])

    st.markdown("#### Key findings")
    for f in lit["key_findings"]:
        st.markdown(f"- {f}")

    if lit["pmids"]:
        st.markdown(f"#### Supporting PMIDs ({len(lit['pmids'])})")
        pmid_links = " · ".join(
            f"[{p}](https://pubmed.ncbi.nlm.nih.gov/{p}/)" for p in lit["pmids"]
        )
        st.markdown(pmid_links)
