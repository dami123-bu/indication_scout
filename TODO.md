# TODO

## RAG Pipeline

- [ ] Implement `synthesize()` in `services/retrieval.py` -- last remaining stub in the RAG pipeline
- [ ] Define `EvidenceSummary` Pydantic model in `models/model_evidence_summary.py`
- [ ] Write `prompts/synthesize.txt` prompt template
- [ ] Implement `run_rag()` end-to-end orchestration in `services/retrieval.py`
- [ ] Unit and integration tests for `synthesize` and `run_rag`

## Agents

- [ ] Implement `Orchestrator.run()` in `agents/orchestrator.py`
- [ ] Implement `LiteratureAgent.run()` in `agents/literature.py`
- [ ] Implement `ClinicalTrialsAgent.run()` in `agents/clinical_trials.py`
- [ ] Implement `MechanismAgent.run()` in `agents/mechanism.py`
- [ ] Implement `SafetyAgent.run()` in `agents/safety.py`

## Data Sources

- [ ] Implement `DrugBankClient.get_drug()` and `get_interactions()` in `data_sources/drugbank.py`

## API

- [ ] Define API routes in `api/routes/`
- [ ] Define request/response schemas in `api/schemas/`

## CLI

- [ ] Implement CLI module (`indication_scout.cli.cli`) referenced in `pyproject.toml`

## Infrastructure

- [ ] Add connection pooling singleton to `db/session.py` (currently creates new engine per call)

## Code Quality

- [ ] Fix `runners/pubmed_runner.py` to use `logging` instead of `print()`
- [ ] Remove superseded tests in `tests/integration/services/test_pubmed_query.py` (two tests marked `# TODO delete`)
- [ ] Review `get_drug_competitors()` in Open Targets client (marked `# TODO needs rework`)
