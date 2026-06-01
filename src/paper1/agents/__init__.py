"""Three agents: Extractor, Critic, Consolidator."""

from paper1.agents.consolidator import ConsolidatorAgent
from paper1.agents.critic import CriticAgent
from paper1.agents.extractor import ExtractorAgent

__all__ = ["ConsolidatorAgent", "CriticAgent", "ExtractorAgent"]
