"""Evidence scoring service."""

from indication_scout.models.evidence import Evidence, EvidenceStrength


class EvidenceScorer:
    """Service for scoring evidence quality and relevance."""

    def score(self, evidence: Evidence) -> float:
        """Calculate a score for the given evidence."""
        base_scores = {
            EvidenceStrength.STRONG: 0.9,
            EvidenceStrength.MODERATE: 0.6,
            EvidenceStrength.WEAK: 0.3,
            EvidenceStrength.ANECDOTAL: 0.1,
        }
        return base_scores.get(evidence.strength, 0.0)

    def aggregate_scores(self, evidences: list[Evidence]) -> float:
        """Aggregate scores from multiple pieces of evidence."""
        if not evidences:
            return 0.0
        scores = [self.score(e) for e in evidences]
        return sum(scores) / len(scores)
