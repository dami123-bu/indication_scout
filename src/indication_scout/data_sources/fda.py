"""
openFDA FAERS (Drug Adverse Event) client.

Two methods:
  1. get_top_reactions — Aggregated adverse reaction counts for a drug
  2. get_events        — Individual adverse event records for a drug
"""

from __future__ import annotations

import logging
from typing import Any

from indication_scout.config import get_settings
from indication_scout.constants import (
    OPENFDA_BASE_URL,
    OPENFDA_MAX_LIMIT,
    REACTION_OUTCOME_MAP,
)
from indication_scout.data_sources.base_client import BaseClient
from indication_scout.models.model_fda import FAERSEvent, FAERSReactionCount

logger = logging.getLogger("indication_scout.data_sources.fda")


class FDAClient(BaseClient):
    """Client for querying the openFDA Drug Adverse Event API."""

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__()
        self._api_key = api_key if api_key is not None else get_settings().openfda_api_key

    @property
    def _source_name(self) -> str:
        return "openfda"

    # -- Public methods -------------------------------------------------------

    async def get_top_reactions(
        self, drug_name: str, limit: int = 10
    ) -> list[FAERSReactionCount]:
        """Return the most frequently reported adverse reactions for a drug."""
        params = self._build_params(drug_name, limit)
        params["count"] = "patient.reaction.reactionmeddrapt.exact"

        data = await self._rest_get(OPENFDA_BASE_URL, params)
        results = data.get("results", [])

        return [self._parse_reaction_count(r) for r in results]

    async def get_events(
        self, drug_name: str, limit: int = 10
    ) -> list[FAERSEvent]:
        """Return individual adverse event records for a drug."""
        params = self._build_params(drug_name, limit)

        data = await self._rest_get(OPENFDA_BASE_URL, params)
        results = data.get("results", [])

        return [self._parse_event(r) for r in results]

    # -- Private helpers ------------------------------------------------------

    def _build_params(self, drug_name: str, limit: int) -> dict[str, str]:
        """Build common query parameters for the openFDA API."""
        params: dict[str, str] = {
            "search": f'patient.drug.medicinalproduct:"{drug_name}"',
            "limit": str(min(limit, OPENFDA_MAX_LIMIT)),
        }
        if self._api_key:
            params["api_key"] = self._api_key
        return params

    @staticmethod
    def _parse_reaction_count(raw: dict[str, Any]) -> FAERSReactionCount:
        """Parse a single count result into FAERSReactionCount."""
        return FAERSReactionCount(
            term=raw["term"],
            count=raw["count"],
        )

    @staticmethod
    def _parse_event(raw: dict[str, Any]) -> FAERSEvent:
        """Parse a single event result into FAERSEvent."""
        patient = raw.get("patient", {})
        drugs = patient.get("drug", [])
        reactions = patient.get("reaction", [])

        first_drug = drugs[0] if drugs else {}
        first_reaction = reactions[0] if reactions else {}

        outcome_code = first_reaction.get("reactionoutcome")
        reaction_outcome = REACTION_OUTCOME_MAP.get(str(outcome_code)) if outcome_code else None

        return FAERSEvent(
            medicinal_product=first_drug.get("medicinalproduct", ""),
            drug_indication=first_drug.get("drugindication"),
            reaction=first_reaction.get("reactionmeddrapt", ""),
            reaction_outcome=reaction_outcome,
            serious=raw.get("serious"),
            company_numb=raw.get("companynumb"),
        )
