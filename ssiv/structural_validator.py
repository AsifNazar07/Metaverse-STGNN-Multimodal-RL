import networkx as nx
import numpy as np


class StructuralValidator:
    def __init__(self, alpha: float = 0.5):
        """
        Initialize structural validator.

        Args:
            alpha (float): edge importance weight (0–1)
        """
        self.alpha = alpha

    def validate(
        self,
        policy_graph: nx.Graph,
        forecast_graph: nx.Graph,
    ) -> float:
        """
        Compute structural consistency score between policy and forecast graphs.

        Args:
            policy_graph (nx.Graph): Graph derived from IBN policy
            forecast_graph (nx.Graph): Graph derived from ST-GNN forecast

        Returns:
            float: consistency score in [0, 1]
        """
        if policy_graph is None or forecast_graph is None:
            return 0.0

        # Node consistency
        policy_nodes = set(policy_graph.nodes())
        forecast_nodes = set(forecast_graph.nodes())

        if len(policy_nodes) == 0:
            return 0.0

        node_overlap = len(policy_nodes & forecast_nodes) / len(policy_nodes)

        # Edge consistency (GAT-like attention)
        policy_edges = set(policy_graph.edges())
        forecast_edges = set(forecast_graph.edges())

        if len(policy_edges) == 0:
            edge_overlap = 1.0
        else:
            edge_overlap = len(policy_edges & forecast_edges) / len(policy_edges)

        # Weighted consistency score
        consistency_score = (
            self.alpha * node_overlap
            + (1.0 - self.alpha) * edge_overlap
        )

        return float(np.clip(consistency_score, 0.0, 1.0))


# --- Functional interface (used by SSIV score) ---

_structural_validator = StructuralValidator(alpha=0.5)


def structural_consistency(
    policy_graph: nx.Graph,
    forecast_graph: nx.Graph,
) -> float:
    """
    Lightweight functional wrapper for SSIV.

    Used in:
        S_t = λ1 * sim_BERT(...) + λ2 * consistency_GAT(...)
    """
    return _structural_validator.validate(policy_graph, forecast_graph)
