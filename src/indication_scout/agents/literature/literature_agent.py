import logging

from indication_scout.agents.literature.literature_output import LiteratureOutput
from indication_scout.agents.literature.literature_state import LiteratureState

logger = logging.getLogger(__name__)
from langgraph.graph import StateGraph, START, END


def build_literature_graph(
    svc, db, drug_profile, date_before=None, max_search_results=None, num_top_k=5
):

    # ====================== NODES ======================

    async def expand_node(state: LiteratureState):
        result = await svc.expand_search_terms(
            state.drug_name, state.disease_name, drug_profile
        )
        logger.debug("expand_node: generated %d search terms", len(result))
        return {"expanded_search_results": result}

    async def fetch_node(state: LiteratureState):
        pmids = await svc.fetch_and_cache(
            state.expanded_search_results,
            db,
            date_before=date_before,
            max_results=max_search_results,
        )
        logger.debug("fetch_node: cached %d abstracts", len(pmids))
        return {"pmids": pmids}

    async def search_node(state: LiteratureState):
        results = await svc.semantic_search(
            state.disease_name, state.drug_name, state.pmids, db, num_top_k
        )
        logger.debug("search_node: retrieved %d abstracts", len(results))
        return {"semantic_search_results": results}

    async def synthesize_node(state: LiteratureState):
        result = await svc.synthesize(
            state.drug_name, state.disease_name, state.semantic_search_results
        )
        logger.debug("synthesize_node: strength=%s", result.strength)
        return {"evidence_summary": result}

    def assemble_node(state: LiteratureState):
        final_output = LiteratureOutput(
            search_results=state.expanded_search_results,
            pmids=state.pmids,
            semantic_search_results=state.semantic_search_results,
            evidence_summary=state.evidence_summary,
            summary=state.evidence_summary.summary if state.evidence_summary else "",
        )
        return {"final_output": final_output}

    # ====================== BUILD GRAPH ======================

    workflow = StateGraph(LiteratureState)

    workflow.add_node("expand", expand_node)
    workflow.add_node("fetch", fetch_node)
    workflow.add_node("search", search_node)
    workflow.add_node("synthesize", synthesize_node)
    workflow.add_node("assemble", assemble_node)

    workflow.add_edge(START, "expand")
    workflow.add_edge("expand", "fetch")
    workflow.add_edge("fetch", "search")
    workflow.add_edge("search", "synthesize")
    workflow.add_edge("synthesize", "assemble")
    workflow.add_edge("assemble", END)

    return workflow.compile()
