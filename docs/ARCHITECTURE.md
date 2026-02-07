# Architecture

## Open Targets Data Structure

The `DrugEvaluation` object is the main data container fetched by `OpenTargetsClient.load()`. It contains everything needed for indication discovery.

```
DrugData
 |-- list[DrugWarning]
 |-- list[Indication]
 |-- list[AdverseEvent]           (FAERS drug-level adverse events)
 +-- list[DrugTarget]
      |
      +-- TargetData (fetched separately, keyed by target_id)
           |-- list[Association]
           |-- list[Pathway]
           |-- list[Interaction]
           |-- list[KnownDrug]
           |-- list[TissueExpression]
           |    |-- RNAExpression
           |    +-- ProteinExpression
           |         +-- list[CellTypeExpression]
           |-- list[MousePhenotype]
           |    +-- list[BiologicalModel]
           |-- list[SafetyLiability]
           |    +-- list[SafetyEffect]
           +-- list[GeneticConstraint]
```

---

## Model Definitions

### DrugData

| Field | Type | Description |
|-------|------|-------------|
| chembl_id | str | ChEMBL identifier, e.g. "CHEMBL3137309" |
| name | str | Drug name, e.g. "semaglutide" |
| synonyms | list[str] | Alternative names |
| trade_names | list[str] | Brand names, e.g. ["Ozempic", "Wegovy"] |
| drug_type | str | "Protein", "Small molecule", etc. |
| is_approved | bool | Whether drug is approved |
| max_clinical_phase | float | Highest phase reached (0-4) |
| year_first_approved | int or None | Year of first approval |
| warnings | list[DrugWarning] | Black box warnings, withdrawals |
| indications | list[Indication] | Diseases this drug targets |
| targets | list[DrugTarget] | Molecular targets |
| adverse_events | list[AdverseEvent] | FAERS drug-level adverse events |
| adverse_events_critical_value | float | LLR threshold for significance |

### DrugTarget

| Field | Type | Description |
|-------|------|-------------|
| target_id | str | Ensembl ID, e.g. "ENSG00000112164" |
| target_symbol | str | Gene symbol, e.g. "GLP1R" |
| mechanism_of_action | str | e.g. "GLP-1 receptor agonist" |
| action_type | str or None | "AGONIST", "INHIBITOR", etc. |

### DrugWarning

| Field | Type | Description |
|-------|------|-------------|
| warning_type | str | "Black Box Warning", "Withdrawn" |
| description | str or None | Warning details |
| toxicity_class | str or None | Classification of toxicity |
| country | str or None | Where warning was issued |
| year | int or None | When warning was issued |
| efo_id | str or None | Disease term for warning |

### Indication

| Field | Type | Description |
|-------|------|-------------|
| disease_id | str | EFO identifier, e.g. "EFO_0001360" |
| disease_name | str | e.g. "type 2 diabetes mellitus" |
| max_phase | float | Highest phase for this indication (0-4) |
| references | list[dict] | Supporting references |

### TargetData

| Field | Type | Description |
|-------|------|-------------|
| target_id | str | Ensembl ID |
| symbol | str | Gene symbol |
| name | str | Full gene name |
| associations | list[Association] | All diseases linked to this target |
| pathways | list[Pathway] | Reactome pathways |
| interactions | list[Interaction] | Protein-protein interactions |
| known_drugs | list[KnownDrug] | Other drugs targeting this protein |
| expressions | list[TissueExpression] | Tissue expression data |
| mouse_phenotypes | list[MousePhenotype] | Mouse model phenotypes |
| safety_liabilities | list[SafetyLiability] | Known target safety effects |
| genetic_constraint | list[GeneticConstraint] | GnomAD constraint scores |

### Association

| Field | Type | Description |
|-------|------|-------------|
| disease_id | str | EFO identifier |
| disease_name | str | Human-readable disease name |
| overall_score | float | Association strength (0-1) |
| datatype_scores | dict[str, float] | Breakdown by evidence type |
| therapeutic_areas | list[str] | Top-level disease categories |

### Pathway

| Field | Type | Description |
|-------|------|-------------|
| pathway_id | str | Reactome ID |
| pathway_name | str | Pathway name |
| top_level_pathway | str | Parent category |

### Interaction

| Field | Type | Description |
|-------|------|-------------|
| interacting_target_id | str | Ensembl ID of partner |
| interacting_target_symbol | str | Gene symbol of partner |
| interaction_score | float | Confidence (0-1) |
| source_database | str | IntAct, STRING, Signor, Reactome |
| biological_role | str | e.g. "enzyme target" |
| evidence_count | int | Number of supporting experiments |

### KnownDrug

| Field | Type | Description |
|-------|------|-------------|
| drug_id | str | ChEMBL ID |
| drug_name | str | Drug name |
| disease_id | str | Target indication |
| disease_name | str | Indication name |
| phase | float | Clinical phase (0-4) |
| status | str or None | "Active", "Completed", "Terminated" |
| mechanism_of_action | str | Mechanism description |
| clinical_trial_ids | list[str] | NCT IDs |

### TissueExpression

| Field | Type | Description |
|-------|------|-------------|
| tissue_id | str | UBERON ID |
| tissue_name | str | e.g. "liver", "pancreas" |
| tissue_anatomical_system | str | e.g. "digestive system" |
| rna | RNAExpression | RNA expression data |
| protein | ProteinExpression | Protein expression data |

### RNAExpression

| Field | Type | Description |
|-------|------|-------------|
| value | float | Expression value in TPM |
| quantile | int | Relative level across tissues |
| unit | str | Unit of measurement (default "TPM") |

### ProteinExpression

| Field | Type | Description |
|-------|------|-------------|
| level | int | Expression level (0-3) |
| reliability | bool | Whether measurement is reliable |
| cell_types | list[CellTypeExpression] | Cell type breakdown |

### CellTypeExpression

| Field | Type | Description |
|-------|------|-------------|
| name | str | Cell type name |
| level | int | Expression level |
| reliability | bool | Whether measurement is reliable |

### MousePhenotype

| Field | Type | Description |
|-------|------|-------------|
| phenotype_id | str | MP ontology ID |
| phenotype_label | str | e.g. "increased circulating glucose" |
| phenotype_categories | list[str] | Top-level MP categories |
| biological_models | list[BiologicalModel] | Mouse models showing this |

### BiologicalModel

| Field | Type | Description |
|-------|------|-------------|
| allelic_composition | str | Allelic composition of the model |
| genetic_background | str | Genetic background strain |
| literature | list[str] | PMIDs for supporting literature |
| model_id | str | MGI model ID |

### SafetyLiability

| Field | Type | Description |
|-------|------|-------------|
| event | str or None | Safety event description |
| event_id | str or None | Unique identifier for safety event |
| effects | list[SafetyEffect] | Reported safety effects |
| datasource | str or None | Source reporting the liability |
| literature | str or None | Reference citations |
| url | str or None | Link for additional details |

### SafetyEffect

| Field | Type | Description |
|-------|------|-------------|
| direction | str | Direction of reported effect (e.g., "increase", "decrease") |
| dosing | str or None | Dosing conditions related to the effect |

### AdverseEvent (Drug-level)

| Field | Type | Description |
|-------|------|-------------|
| name | str | Adverse event term |
| meddra_code | str or None | MedDRA identifier |
| count | int | Number of FAERS reports |
| log_likelihood_ratio | float | Statistical significance |

### GeneticConstraint

| Field | Type | Description |
|-------|------|-------------|
| constraint_type | str | "syn", "mis", or "lof" |
| oe | float or None | Observed/expected ratio |
| oe_lower | float or None | Lower confidence bound |
| oe_upper | float or None | Upper confidence bound |
| score | float or None | Constraint score |
| upper_bin | int or None | 0 = most constrained, 5 = least |

---

## Three Disease Links

| Path | What it answers |
|------|-----------------|
| `drug.indications` | What diseases is **this drug** being tested for? |
| `target.associations` | What diseases is **this target** linked to (by any evidence)? |
| `target.known_drugs` | What **other drugs** target this protein, and for what diseases? |


---

## Finding Whitespace Indications

The system finds repurposing opportunities by:

1. Get `target.associations` - diseases linked to the target
2. Filter out `drug.indications` - diseases already being pursued
3. What remains = potential new indications

Example: If GLP1R has a high association score with NASH, but semaglutide's `indications` list doesn't include NASH, that's a whitespace opportunity.

---

## Helper Properties on DrugEvaluation

```python
evaluation.get_target(target_id)      # Returns TargetData, raises TargetNotFoundError
evaluation.primary_target             # First target (or None if no targets)
evaluation.approved_disease_ids       # set[str] of phase 4+ disease IDs
evaluation.investigated_disease_ids   # set[str] of all disease IDs being pursued
```
