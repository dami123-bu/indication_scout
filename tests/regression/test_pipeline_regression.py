"""Full-pipeline regression test against a committed golden SupervisorOutput.

Marked `regression` and excluded from the default `pytest` run via the
addopts marker filter in `pytest.ini`. To run:

    pytest -m regression                                # replay (default mode)
    SCOUT_CASSETTE_MODE=record  pytest -m regression    # re-record
    SCOUT_CASSETTE_MODE=live    pytest -m regression    # bypass cassette

The test requires a live database connection (Postgres + pgvector) just like
the rest of the integration suite — the cassette only stubs external HTTP /
LLM traffic.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from indication_scout.regression.harness import compare_reports, has_errors, render_diffs

from tests.regression.cassette import use_cassette
from tests.regression.constants import CASSETTE_DIR, GOLDEN_DIR

PINNED_DRUGS = ["metformin"]


@pytest.mark.regression
@pytest.mark.parametrize("drug", PINNED_DRUGS)
async def test_pipeline_regression(drug: str) -> None:
    golden_path = GOLDEN_DIR / f"{drug}.json"
    cassette_path = CASSETTE_DIR / drug / "pipeline.yaml"

    if not golden_path.exists():
        pytest.skip(
            f"No golden snapshot at {golden_path}. Record one with "
            f"SCOUT_CASSETTE_MODE=record pytest -m regression -k {drug}"
        )

    from indication_scout.agents.supervisor.supervisor_output import SupervisorOutput

    golden = SupervisorOutput.model_validate_json(golden_path.read_text())

    current = await _run_pipeline(drug, cassette_path)

    # In record mode, overwrite the golden snapshot with the freshly recorded
    # run so cassette + golden stay in lockstep. This is the only place that
    # writes to the golden directory; the test still runs the comparison after
    # so a record-mode run can catch a structural regression in the golden.
    if os.environ.get("SCOUT_CASSETTE_MODE", "").lower() == "record":
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(current.model_dump_json(indent=2))
        golden = current

    diffs = compare_reports(golden, current)
    error_diffs = [d for d in diffs if d.severity == "error"]
    assert not has_errors(diffs), (
        f"\nRegression detected for {drug}:\n{render_diffs(error_diffs)}\n"
        f"(Run `scout diff-report {golden_path} <current.json>` for a full diff.)"
    )


async def _run_pipeline(drug: str, cassette_path: Path):
    """Run the full supervisor pipeline under the cassette context.

    Mirrors `_run_for_drug` in cli/cli.py but skips the markdown/file output —
    we want the SupervisorOutput object directly.
    """
    from langchain_anthropic import ChatAnthropic

    from indication_scout.agents.supervisor.supervisor_agent import (
        build_supervisor_agent,
        run_supervisor_agent,
    )
    from indication_scout.constants import DEFAULT_CACHE_DIR
    from indication_scout.db.session import get_db
    from indication_scout.helpers.drug_helpers import normalize_drug_name
    from indication_scout.services.retrieval import RetrievalService

    drug = normalize_drug_name(drug)
    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        temperature=0,
        max_tokens=4096,
        api_key=os.environ.get("ANTHROPIC_API_KEY", "test-key-unused-in-replay"),
    )
    db = next(get_db())
    svc = RetrievalService(DEFAULT_CACHE_DIR)

    with use_cassette(cassette_path):
        agent, get_merged_allowlist, get_auto_findings = build_supervisor_agent(
            llm=llm, svc=svc, db=db, date_before=None
        )
        return await run_supervisor_agent(
            agent,
            get_merged_allowlist,
            drug,
            get_auto_findings=get_auto_findings,
        )
