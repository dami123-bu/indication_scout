"""Mechanism tools.

Uses content_and_artifact so typed Python objects are preserved on
msg.artifact. Tools share inter-call data via a closure-scoped store dict.
No InjectedState, no LangGraph state machinery.
"""

from langchain_core.tools import tool

from indication_scout.data_sources.open_targets import OpenTargetsClient
from indication_scout.models.model_open_targets import Association, Pathway


def build_mechanism_tools() -> list:
    """Build tools that share data via a closure-scoped store dict.

    get_drug_targets must be called first — it populates the store with the
    symbol→target_id map that get_target_associations and get_target_pathways
    use to resolve target symbols.
    """

    store: dict = {}

    @tool(response_format="content_and_artifact")
    async def get_drug_targets(drug_name: str) -> tuple[str, dict[str, str]]:
        """Fetch the molecular targets for a drug.

        Returns a mapping of gene symbol to Ensembl target ID
        (e.g. {"GLP1R": "ENSG00000112164"}). Call this first — subsequent
        tools resolve target symbols using this result.
        """
        async with OpenTargetsClient() as client:
            drug = await client.get_drug(drug_name)

        target_map = {
            t.target_symbol: t.target_id
            for t in drug.targets
            if t.target_symbol and t.target_id
        }
        store["target_ids"] = target_map
        return (
            f"Found {len(target_map)} targets for {drug_name}: {', '.join(target_map)}",
            target_map,
        )

    @tool(response_format="content_and_artifact")
    async def get_target_associations(
        target_symbol: str,
    ) -> tuple[str, list[Association]]:
        """Fetch disease associations for a target, ranked by overall score.

        Returns the top 20 disease associations with per-datatype evidence
        scores (genetic_association, literature, etc.). Call get_drug_targets
        first so the target ID is available.
        """
        target_ids: dict[str, str] = store.get("target_ids", {})
        target_id = target_ids.get(target_symbol)
        if not target_id:
            return f"Target '{target_symbol}' not found in store — call get_drug_targets first", []

        async with OpenTargetsClient() as client:
            associations = await client.get_target_data_associations(target_id)

        top = sorted(associations, key=lambda a: a.overall_score or 0, reverse=True)[:20]
        return (
            f"{len(top)} associations for {target_symbol} (top: {top[0].disease_name if top else 'none'})",
            top,
        )

    @tool(response_format="content_and_artifact")
    async def get_target_pathways(
        target_symbol: str,
    ) -> tuple[str, list[Pathway]]:
        """Fetch Reactome pathways for a target.

        Returns the pathways this target participates in, grouped by top-level
        pathway term. Useful for mechanistic reasoning about disease relevance.
        Call get_drug_targets first so the target ID is available.
        """
        target_ids: dict[str, str] = store.get("target_ids", {})
        target_id = target_ids.get(target_symbol)
        if not target_id:
            return f"Target '{target_symbol}' not found in store — call get_drug_targets first", []

        async with OpenTargetsClient() as client:
            pathways = await client.get_target_data_pathways(target_id)

        top_level = {p.top_level_pathway for p in pathways if p.top_level_pathway}
        return (
            f"{len(pathways)} pathways for {target_symbol} across {len(top_level)} top-level terms",
            pathways,
        )

    return [get_drug_targets, get_target_associations, get_target_pathways]
