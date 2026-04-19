"""Integration tests for the mechanism agent.

Hits real Anthropic and Open Targets APIs.
Expected values verified by a live run on 2026-04-12.
"""

import logging

from langchain_anthropic import ChatAnthropic

from indication_scout.agents.mechanism.mechanism_agent import (
    build_mechanism_agent,
    run_mechanism_agent,
)
from indication_scout.agents.mechanism.mechanism_output import MechanismOutput


# ------------------------------------------------------------------
# Imatinib
#
# Expected values verified by a live run on 2026-04-12.
# ------------------------------------------------------------------

# Targets that must appear in drug_targets (imatinib's known kinase targets)
_EXPECTED_TARGET_SYMBOLS = {"ABL1", "BCR", "KIT", "PDGFRB"}

# Symbols for which associations must be fetched
_EXPECTED_ASSOCIATION_SYMBOLS = {"ABL1", "KIT", "PDGFRB"}

# Top diseases expected per key target
_EXPECTED_TOP_DISEASES = {
    "ABL1": "chronic myelogenous leukemia",
    "KIT": "gastrointestinal stromal tumor",
}

# Top-level pathway terms that must appear across all targets
_EXPECTED_TOP_LEVEL_PATHWAYS = {"Signal Transduction"}


async def test_metformin_mechanism_agent():
    """End-to-end: mechanism agent produces correct MechanismOutput for imatinib.

    Verifies:
    - known kinase targets (ABL1, BCR, KIT, PDGFRB) are in drug_targets
    - shaped_associations contains entries for each expected target
    - top disease per key target (by overall_score) matches expected values
    - pathways are fetched per target (dict keyed by symbol)
    - Signal Transduction appears as a top-level pathway term across targets
    - narrative summary is non-empty and mentions relevant biology
    """
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0, max_tokens=4096)
    agent = build_mechanism_agent(llm)

    output = await run_mechanism_agent(agent, "imatinib")

    assert isinstance(output, MechanismOutput)

    # --- drug_targets ---
    assert len(output.drug_targets) >= 4
    assert _EXPECTED_TARGET_SYMBOLS.issubset(set(output.drug_targets.keys()))
    for symbol, ensembl_id in output.drug_targets.items():
        assert ensembl_id.startswith("ENSG"), f"Expected Ensembl ID for {symbol}, got {ensembl_id!r}"

    # --- shaped_associations ---
    symbols_with_associations = {s.target_symbol for s in output.shaped_associations}
    assert _EXPECTED_ASSOCIATION_SYMBOLS.issubset(symbols_with_associations)
    # Verify top disease per key target — use highest overall_score entry per symbol
    for symbol, expected_disease in _EXPECTED_TOP_DISEASES.items():
        per_symbol = [s for s in output.shaped_associations if s.target_symbol == symbol]
        assert per_symbol, f"Expected shaped_associations for {symbol}"
        top = max(per_symbol, key=lambda s: s.overall_score or 0.0).disease_name
        assert top == expected_disease, f"Expected top disease for {symbol} to be {expected_disease!r}, got {top!r}"

    # --- pathways ---
    assert len(output.pathways) >= 3
    all_top_level = {p.top_level_pathway for paths in output.pathways.values() for p in paths}
    assert _EXPECTED_TOP_LEVEL_PATHWAYS.issubset(all_top_level)

    # --- summary ---
    assert len(output.summary) > 200
    summary_lower = output.summary.lower()
    assert any(kw in summary_lower for kw in ("abl", "kit", "pdgfr", "kinase", "imatinib"))
