# Plan: ATC Codes via ChEMBL Molecule Endpoint

## Steps

- [x] 1. Add `CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"` to `constants.py`

- [x] 2. Create `src/indication_scout/models/model_chembl.py` with `MoleculeData` Pydantic model
  - Fields: `atc_classifications: list[str]`, `molecule_type: str`, `max_phase: str`, `black_box_warning: int`, `first_approval: int | None`, `oral: bool`

- [x] 3. Implement `ChEMBLClient.get_molecule(chembl_id: str) -> MoleculeData` in `chembl.py`
  - `GET {CHEMBL_BASE_URL}/molecule/{chembl_id}.json` via `_rest_get()`
  - Parse response into `MoleculeData`
  - Raise `DataSourceError` on failure

- [x] 4. Add `atc_classifications: list[str] = []` to `DrugData` in `model_open_targets.py`

- [x] 5. Create `src/indication_scout/services/drug_service.py` with `get_drug_with_atc(drug_name: str) -> DrugData`
  - Call `OpenTargetsClient.get_drug(drug_name)`
  - Call `ChEMBLClient.get_molecule(drug.chembl_id)`
  - Return `DrugData` with `atc_classifications` populated
  - On `DataSourceError` from ChEMBL, log warning and return `DrugData` with empty `atc_classifications`

- [x] 6. Unit tests in `tests/unit/test_chembl.py`
  - Mock HTTP response with real API response shape
  - Assert `MoleculeData` fields parse correctly including `atc_classifications`

- [x] 7. Integration tests in `tests/integration/test_chembl.py`
  - Call `get_molecule("CHEMBL894")` against live API
  - Assert `atc_classifications == ["N06AX12"]` and other known Bupropion fields
