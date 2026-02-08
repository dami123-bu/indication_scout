"""
LiteratureIndex for semantic search over pre-indexed PubMed content.

Uses ChromaDB + BioLinkBERT for vector search.
"""

from datetime import date

from indication_scout.models.model_pubmed import RAGResult


class LiteratureIndex:
    """
    Semantic search over pre-indexed PubMed abstracts.

    This searches a ChromaDB collection that was previously populated
    by fetching and embedding PubMed content. The indexing pipeline
    (fetch → chunk → embed → store) is a separate concern.
    """

    async def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        date_before: date | None = None,
    ) -> list[RAGResult]:
        """
        Search for conceptually related publications using vector similarity.

        Parameters
        ----------
        query : str
            Natural language query (not PubMed syntax).
            e.g. "incretin-based therapy hepatic steatosis reduction"
        top_k : int
            Maximum number of results to return.
        date_before : date, optional
            Only return publications before this date (for temporal holdout).

        Returns
        -------
        list[RAGResult]
            Chunks with similarity scores and full Publication metadata.
        """
        raise NotImplementedError
