# ChEMBL Client — Data Contracts

One method. Returns molecule-level drug properties from the ChEMBL REST API.

**API:** ChEMBL REST API (`https://www.ebi.ac.uk/chembl/api/data`)
**Auth:** None required
**Rate limit:** No published limit; treat as a public scholarly API (avoid hammering)
**Versioning:** URL-less versioning — the base URL reflects the current stable release

---

## 1. `get_molecule(chembl_id) → MoleculeData`

Fetch molecule properties by ChEMBL compound ID.

**In:**
- `chembl_id` — canonical ChEMBL compound identifier (e.g. `"CHEMBL1118"`)

**API call:** `GET /molecule/{chembl_id}.json`

**Out:**
- `molecule_chembl_id` — the canonical identifier echoed back (e.g. `"CHEMBL1118"`)
- `molecule_type` — compound classification: `"Small molecule"`, `"Protein"`, `"Antibody"`, `"Oligonucleotide"`, `"Oligosaccharide"`, `"Cell"`, `"Unknown"`
- `max_phase` — highest clinical trial phase reached: `"4"` (approved), `"3"`, `"2"`, `"1"`, `"0.5"` (early Phase 1), `null` (preclinical / no trials)
- `atc_classifications` — list of WHO ATC codes (e.g. `["A10BA02"]`). Empty list if none assigned
- `black_box_warning` — `1` if the drug carries an FDA black box warning, `0` otherwise
- `first_approval` — year of first regulatory approval as int (e.g. `1994`), `null` if not approved
- `oral` — `true` if the drug has an oral formulation

**Response field mapping:**
```
molecule_chembl_id   → molecule_chembl_id
molecule_type        → molecule_type
max_phase            → max_phase
atc_classifications  → atc_classifications
black_box_warning    → black_box_warning
first_approval       → first_approval
oral                 → oral
```

**Error behavior:**
- **Molecule not found:** Raises `DataSourceError` (HTTP 404 propagated from `_rest_get`)
- **Malformed response:** Raises `DataSourceError` with message `"Unexpected response shape for '{chembl_id}'"`
- **Unexpected exception:** Wrapped and re-raised as `DataSourceError`

---

## Data quality notes

**`max_phase` is a string, not an int.** The API returns it as a string (`"4"`, `"3"`, etc.) or `null`. Do not compare as an integer without explicit casting.

**ATC codes are sparse.** Many approved drugs have no ATC code assigned in ChEMBL, especially biologics and recently approved compounds. An empty `atc_classifications` list does not mean the drug lacks a therapeutic class.

**`first_approval` reflects any regulatory body.** This is not necessarily FDA approval — it could be EMA, PMDA, or another agency, whichever came first.

**`molecule_type` matters for mechanism agents.** A `"Small molecule"` has very different developability properties than an `"Antibody"`. The mechanism agent should use this to qualify repurposing feasibility.

---

## Pydantic model

```python
class MoleculeData(BaseModel):
    molecule_chembl_id: str
    molecule_type: str
    max_phase: str | None
    atc_classifications: list[str]
    black_box_warning: int
    first_approval: int | None
    oral: bool
```

---

## Data flow from ChEMBL into the agents

```
Mechanism Agent
  │
  ├── get_molecule("CHEMBL1118")      # metformin
  │     → molecule_type, max_phase, atc_classifications, black_box_warning
  │
  └── OUTPUT: drug property summary
        - Is this a small molecule or biologic?
        - How far has it been in trials (max_phase)?
        - Does it carry a black box warning?
        - What therapeutic class (ATC)?
        - When was it first approved?
```

---

## Agent-to-method mapping

| Agent | Method | What it gets |
|-------|--------|--------------|
| **Mechanism** | `get_molecule` | Molecule type, clinical maturity (`max_phase`), ATC class, safety flag (`black_box_warning`) |