"""Integration tests for OpenTargetsopen_targets_client."""

import logging

import pytest

from indication_scout.data_sources.base_client import DataSourceError

logger = logging.getLogger(__name__)



# --- get_drug ---


@pytest.mark.asyncio
async def test_sildenafil_drug_data(open_targets_client):
    """Test fetching drug data and indications for semaglutide."""
    drug = await open_targets_client.get_drug("Semaglutide")
    indications = drug.indications
    match = [i for i in indications if "kidney" in i.disease_name.lower()]
    approved = [a for a in match if a.disease_id in drug.approved_disease_ids]
    logger.info(drug.indications)


@pytest.mark.asyncio
async def test_semaglutide_drug_data(open_targets_client):
    """Test fetching drug data and indications for semaglutide."""
    drug = await open_targets_client.get_drug("semaglutide")

    # DrugData top-level fields
    assert drug.chembl_id == "CHEMBL2108724"
    assert drug.name == "SEMAGLUTIDE"
    assert "NN-9535" in drug.synonyms
    assert "Ozempic" in drug.trade_names
    assert "Wegovy" in drug.trade_names
    assert drug.drug_type == "Protein"
    assert drug.is_approved is True
    assert drug.max_clinical_phase == 4.0
    assert drug.year_first_approved == 2017
    assert len(drug.indications) > 5
    assert len(drug.targets) > 0
    assert len(drug.adverse_events) > 5
    assert 38.5 < drug.adverse_events_critical_value < 38.6
    assert len(drug.warnings) == 1

    # Indications - should include type 2 diabetes (approved)
    t2d_indication = next(
        (
            i
            for i in drug.indications
            if "type 2 diabetes" in i.disease_name.lower()
        ),
        None,
    )
    assert t2d_indication is not None
    assert t2d_indication.max_phase == 4.0


@pytest.mark.asyncio
async def test_semaglutide_drug_target(open_targets_client):
    """Test DrugTarget fields for semaglutide's GLP1R target."""
    drug = await open_targets_client.get_drug("semaglutide")
    [glp1r] = [t for t in drug.targets if t.target_symbol == "GLP1R"]

    assert glp1r.target_id == "ENSG00000112164"
    assert glp1r.target_symbol == "GLP1R"
    assert (
        glp1r.mechanism_of_action == "Glucagon-like peptide 1 receptor agonist"
    )
    assert glp1r.action_type == "AGONIST"


@pytest.mark.asyncio
async def test_semaglutide_indication(open_targets_client):
    """Test Indication fields for semaglutide's type 2 diabetes indication."""
    drug = await open_targets_client.get_drug("semaglutide")
    [t2d] = [
        i for i in drug.indications if i.disease_name == "type 2 diabetes mellitus"
    ]

    assert t2d.disease_id == "MONDO_0005148"
    assert t2d.disease_name == "type 2 diabetes mellitus"
    assert t2d.max_phase == 4.0
    assert len(t2d.references) == 4
    fda_ref = next(r for r in t2d.references if r["source"] == "FDA")
    assert "label/2017/209637lbl.pdf" in fda_ref["ids"]


@pytest.mark.asyncio
async def test_trastuzumab_adverse_event(open_targets_client):
    """Test AdverseEvent fields for trastuzumab."""
    drug = await open_targets_client.get_drug("trastuzumab")
    adverse_event = next(
        ae for ae in drug.adverse_events if ae.name == "ejection fraction decreased"
    )

    assert adverse_event.name == "ejection fraction decreased"
    assert adverse_event.meddra_code == "10050528"
    assert adverse_event.count == 1124
    assert 2725.0 < adverse_event.log_likelihood_ratio < 2726.0


@pytest.mark.asyncio
async def test_rofecoxib_drug_warning(open_targets_client):
    """Test DrugWarning fields for rofecoxib (Vioxx) - a withdrawn drug with complete warning data."""
    drug = await open_targets_client.get_drug("rofecoxib")
    assert len(drug.warnings) > 5

    # Find a specific warning with all fields populated
    warning = next(
        w
        for w in drug.warnings
        if w.toxicity_class == "cardiotoxicity"
        and w.efo_id == "EFO:0000612"
        and "serious cardiovascular events" in (w.description or "")
    )

    assert warning.warning_type == "Withdrawn"
    assert (
        warning.description
        == "Increased risk of serious cardiovascular events, such as heart attacks and strokes"
    )
    assert warning.toxicity_class == "cardiotoxicity"
    assert warning.country == "Worldwide"
    assert warning.year == 2004
    assert warning.efo_id == "EFO:0000612"


@pytest.mark.asyncio
async def test_metformin_drug_data(open_targets_client):
    """Simple test for metformin - a different drug than others in the suite."""
    drug = await open_targets_client.get_drug("metformin")

    assert drug.chembl_id == "CHEMBL1431"
    assert drug.name == "METFORMIN"
    assert drug.drug_type == "Small molecule"
    assert drug.is_approved is True
    assert drug.max_clinical_phase == 4.0
    assert drug.year_first_approved == 1995


# --- get_drug error handling ---


@pytest.mark.asyncio
async def test_nonexistent_drug_raises_error(open_targets_client):
    """Test that a nonexistent drug raises DataSourceError."""
    with pytest.raises(DataSourceError) as exc_info:
        await open_targets_client.get_drug("xyzzy_not_a_real_drug_12345")

    assert exc_info.value.source == "open_targets"
    assert "No drug found" in str(exc_info.value)


@pytest.mark.asyncio
async def test_empty_drug_name_raises_error(open_targets_client):
    """Test that empty drug name raises DataSourceError."""
    with pytest.raises(DataSourceError) as exc_info:
        await open_targets_client.get_drug("")

    assert exc_info.value.source == "open_targets"


@pytest.mark.asyncio
async def test_special_characters_drug_name_raises_error(open_targets_client):
    """Test that special characters in drug name raises DataSourceError."""
    with pytest.raises(DataSourceError) as exc_info:
        await open_targets_client.get_drug("!!!@@@###$$$")

    assert exc_info.value.source == "open_targets"


# --- get_disease_synonyms ---



@pytest.mark.asyncio
async def test_get_disease_synonyms_nep(open_targets_client):

    result = await open_targets_client.get_disease_synonyms("type 2 diabetes nephropathy")
    assert result

@pytest.mark.asyncio
async def test_get_disease_synonyms(open_targets_client):
    """Test fetching disease synonyms for type 2 diabetes mellitus."""
    result = await open_targets_client.get_disease_synonyms("type 2 diabetes mellitus")

    assert result.disease_id == "MONDO_0005148"
    assert result.disease_name == "type 2 diabetes mellitus"
    assert result.parent_names == ["diabetes mellitus"]
    assert len(result.exact) == 25
    assert len(result.related) == 9
    assert len(result.narrow) == 1
    assert "T2DM" in result.exact
    assert "NIDDM" in result.exact
    assert "type 2 diabetes" in result.exact
    assert "maturity-onset diabetes" in result.related
    assert "diabetes mellitus, noninsulin-dependent, 2" in result.narrow


@pytest.mark.asyncio
async def test_get_disease_synonyms_nonexistent(open_targets_client):
    """Test that a nonexistent disease name raises DataSourceError."""
    with pytest.raises(DataSourceError) as exc_info:
        await open_targets_client.get_disease_synonyms("xyzzy_not_a_real_disease_12345")

    assert exc_info.value.source == "open_targets"
    assert "No disease found" in str(exc_info.value)


# --- get_target_data error handling ---


@pytest.mark.asyncio
async def test_nonexistent_target_raises_error(open_targets_client):
    """Test that a nonexistent target ID raises DataSourceError."""
    with pytest.raises(DataSourceError) as exc_info:
        await open_targets_client.get_target_data("ENSG99999999999")

    assert exc_info.value.source == "open_targets"


@pytest.mark.asyncio
async def test_invalid_target_format_raises_error(open_targets_client):
    """Test that invalid target format raises DataSourceError."""
    with pytest.raises(DataSourceError) as exc_info:
        await open_targets_client.get_target_data("not_an_ensembl_id")

    assert exc_info.value.source == "open_targets"


@pytest.mark.asyncio
async def test_empty_target_id_raises_error(open_targets_client):
    """Test that empty target ID raises DataSourceError."""
    with pytest.raises(DataSourceError) as exc_info:
        await open_targets_client.get_target_data("")

    assert exc_info.value.source == "open_targets"


# --- get_target_data ---


@pytest.mark.asyncio
async def test_glp1r_target_associations(open_targets_client):
    """Test fetching associations for GLP1R target."""
    target = await open_targets_client.get_target_data("ENSG00000112164")

    assert len(target.associations) > 10
    [assoc] = [a for a in target.associations if a.disease_name == "gastroparesis"]

    assert assoc is not None
    assert (
        assoc.disease_id.startswith("EFO_") or assoc.disease_id.startswith("MONDO_")
    )
    assert assoc.overall_score > 0.2
    assert 0.4 < assoc.datatype_scores["genetic_association"] < 0.5
    assert 0.2 < assoc.datatype_scores["literature"] < 0.3
    assert "gastrointestinal disease" in assoc.therapeutic_areas


@pytest.mark.asyncio
async def test_glp1r_drug_summary(open_targets_client):
    """Test DrugSummary fields for GLP1R target."""
    target = await open_targets_client.get_target_data("ENSG00000112164")
    liraglutide = next(
        d
        for d in target.drug_summaries
        if d.drug_name == "LIRAGLUTIDE"
        and d.disease_name == "type 2 diabetes mellitus"
    )

    assert liraglutide.drug_id == "CHEMBL4084119"
    assert liraglutide.drug_name == "LIRAGLUTIDE"
    assert liraglutide.disease_id == "MONDO_0005148"
    assert liraglutide.disease_name == "type 2 diabetes mellitus"
    assert liraglutide.phase == 4.0
    assert liraglutide.status is None
    assert (
        liraglutide.mechanism_of_action
        == "Glucagon-like peptide 1 receptor agonist"
    )
    assert isinstance(liraglutide.clinical_trial_ids, list)


@pytest.mark.asyncio
async def test_pdgfrb_target_pathway(open_targets_client):
    """Test Pathway fields for PDGFRB target."""
    target = await open_targets_client.get_target_data("ENSG00000113721")
    [pathway] = [
        p for p in target.pathways if p.pathway_name == "Signaling by PDGF"
    ]

    assert pathway.pathway_id == "R-HSA-186797"
    assert pathway.pathway_name == "Signaling by PDGF"
    assert pathway.top_level_pathway == "Signal Transduction"


@pytest.mark.asyncio
async def test_pdgfrb_target_interaction(open_targets_client):
    """Test Interaction fields for PDGFRB target."""
    target = await open_targets_client.get_target_data("ENSG00000113721")
    interaction = next(
        i
        for i in target.interactions
        if i.interacting_target_symbol == "PLCG1" and i.source_database == "string"
    )

    assert interaction.interacting_target_id == "ENSG00000124181"
    assert interaction.interacting_target_symbol == "PLCG1"
    assert interaction.interaction_score > 0.99
    assert interaction.source_database == "string"
    assert interaction.biological_role == "unspecified role"
    assert interaction.evidence_count == 4


@pytest.mark.asyncio
async def test_atp1a1_target_tissue_expression(open_targets_client):
    """Test TissueExpression, RNAExpression, ProteinExpression, and CellTypeExpression fields."""
    target = await open_targets_client.get_target_data("ENSG00000163399")
    expression = next(e for e in target.expressions if e.tissue_name == "liver")

    # TissueExpression fields
    assert expression.tissue_id == "UBERON_0002107"
    assert expression.tissue_name == "liver"
    assert expression.tissue_anatomical_system == "endocrine system"

    # RNAExpression fields
    assert expression.rna.value == 11819.0
    assert expression.rna.quantile == 5
    assert expression.rna.unit == "TPM"

    # ProteinExpression fields
    assert expression.protein.level == 2
    assert expression.protein.reliability is True
    assert len(expression.protein.cell_types) > 0

    # CellTypeExpression fields
    hepatocytes = next(
        ct for ct in expression.protein.cell_types if ct.name == "hepatocytes"
    )
    assert hepatocytes.name == "hepatocytes"
    assert hepatocytes.level == 1
    assert hepatocytes.reliability is True


@pytest.mark.asyncio
async def test_glp1r_target_mouse_phenotype(open_targets_client):
    """Test MousePhenotype and BiologicalModel fields for GLP1R target."""
    target = await open_targets_client.get_target_data("ENSG00000112164")
    phenotype = next(
        p for p in target.mouse_phenotypes if p.phenotype_id == "MP:0013279"
    )

    # MousePhenotype fields
    assert phenotype.phenotype_id == "MP:0013279"
    assert (
        phenotype.phenotype_label == "increased fasting circulating glucose level"
    )
    assert "homeostasis/metabolism phenotype" in phenotype.phenotype_categories
    assert len(phenotype.biological_models) == 1

    # BiologicalModel fields
    [model] = phenotype.biological_models
    assert model.allelic_composition == "Glp1r<tm1b(KOMP)Mbp> hom early"
    assert model.genetic_background == "C57BL/6NTac"


@pytest.mark.asyncio
async def test_erbb2_target_genetic_constraint(open_targets_client):
    """Test GeneticConstraint fields for ERBB2 target."""
    target = await open_targets_client.get_target_data("ENSG00000141736")
    lof_constraint = next(
        gc for gc in target.genetic_constraint if gc.constraint_type == "lof"
    )

    assert lof_constraint.constraint_type == "lof"
    assert 0.41 < lof_constraint.oe < 0.42
    assert 0.33 < lof_constraint.oe_lower < 0.34
    assert 0.51 < lof_constraint.oe_upper < 0.52
    assert 0.06 < lof_constraint.score < 0.07
    assert lof_constraint.upper_bin == 1


@pytest.mark.asyncio
async def test_atp1a1_target_safety_liability(open_targets_client):
    """Test SafetyLiability and SafetyEffect fields for ATP1A1 target."""
    target = await open_targets_client.get_target_data("ENSG00000163399")
    assert len(target.safety_liabilities) > 5

    # Find a specific safety liability with effects
    liability = next(
        sl
        for sl in target.safety_liabilities
        if sl.event == "cardiac arrhythmia" and sl.event_id == "EFO_0004269"
    )

    # SafetyLiability fields
    assert liability.event == "cardiac arrhythmia"
    assert liability.event_id == "EFO_0004269"
    assert liability.datasource == "Lynch et al. (2017)"
    assert liability.literature == "28216264"
    assert liability.url is None
    assert len(liability.effects) == 1

    # SafetyEffect fields
    [effect] = liability.effects
    assert effect.direction == "Inhibition/Decrease/Downregulation"
    assert effect.dosing == "acute"


SALT_SUFFIXES = [
    " hydrochloride", " hydrobromide", " sulfate", " succinate", " chloride",
    " dimesylate", " tartrate", " citrate", " tosylate", " mesylate", " saccharate",
    " hemihydrate", " maleate", " phosphate", " malate", " esylate", " anhydrous"
]


def normalize_drug_name(name: str) -> str:
    name_lower = name.lower()
    for suffix in SALT_SUFFIXES:
        if name_lower.endswith(suffix):
            return name_lower[: -len(suffix)].strip()
    return name_lower

# TODO remove, for testing only
async def get_drug_competitors(open_targets_client, name) -> dict[str, set[str]]:
    """Fetch phase-4 competitor drugs for bupropion, grouped by disease."""
    name=name.lower()
    bup = await open_targets_client.get_drug(name)
    targets = bup.targets

    siblings: dict[str, set[str]] = {}

    for t in targets:
        logger.info(t.mechanism_of_action)
        summaries = await open_targets_client.get_target_data_drug_summaries(t.target_id)
        drugs=set([normalize_drug_name(s.drug_name.lower()) for s in summaries])
        diseases = set([s.disease_name.lower() for s in summaries])
        for summary in summaries:
            if summary.phase >= 3:
                disease = summary.disease_name
                drug_name = normalize_drug_name(summary.drug_name)
                if disease in siblings:
                    siblings[disease].add(drug_name)
                else:
                    siblings[disease] = {drug_name}

    for key in list(siblings):
        val = siblings[key]
        if name in val:
            del siblings[key]

    sorted_data = dict(
        sorted(siblings.items(), key=lambda item: len(item[1]), reverse=True)
    )
    top_10 = dict(list(sorted_data.items())[:10])
    return top_10

# TODO remove just for testing
def get_pubmed_query(drug_name, disease_name):
    prompt = f"""Convert this disease name to the best PubMed search query. 
    Return ONLY the search term, nothing else.
    Disease: {disease_name}"""

    pubmed_term=disease_name
    #pubmed_term = llm_call(prompt)  # e.g. "diabetic nephropathy"

    return f"{drug_name} AND {pubmed_term}"


# TODO rework
@pytest.mark.asyncio
async def test_surfacing_pipeline(open_targets_client, pubmed_client):
    """Test bupropion competitor pipeline returns top diseases with multiple drugs."""
    drug_name = "Imatinib"
    top_10 = await get_drug_competitors(open_targets_client, drug_name)

    for disease in top_10:
        query=get_pubmed_query(drug_name, disease)
        #query = f"{name} AND {disease}"
        pmids = await pubmed_client.search(query, max_results=10)
        articles = await pubmed_client.fetch_articles(pmids)

        assert query

    assert len(top_10) > 0

# TODO rework
@pytest.mark.asyncio
async def test_get_drug_target_competitors(open_targets_client):
    """Test get_drug_target_competitors returns drugs grouped by target symbol."""
    result = await open_targets_client.get_drug_target_competitors("semaglutide")

    # Semaglutide targets GLP1R (and possibly others)
    assert "GLP1R" in result
    assert len(result["GLP1R"]) > 5

    # LIRAGLUTIDE should be among the GLP1R drugs
    liraglutide = next(
        d
        for d in result["GLP1R"]
        if d.drug_name == "LIRAGLUTIDE"
        and d.disease_name == "type 2 diabetes mellitus"
    )
    assert liraglutide.drug_id == "CHEMBL4084119"
    assert liraglutide.drug_name == "LIRAGLUTIDE"
    assert liraglutide.disease_id == "MONDO_0005148"
    assert liraglutide.disease_name == "type 2 diabetes mellitus"
    assert liraglutide.phase == 4.0
    assert liraglutide.status is None
    assert (
        liraglutide.mechanism_of_action
        == "Glucagon-like peptide 1 receptor agonist"
    )
