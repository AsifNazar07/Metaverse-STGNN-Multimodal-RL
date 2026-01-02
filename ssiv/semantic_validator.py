from sentence_transformers import SentenceTransformer, util
import torch


class SemanticValidator:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = None,
    ):
        """
        Initialize Sentence-BERT model.

        Args:
            model_name (str): Pretrained SBERT model name
            device (str): 'cpu' or 'cuda'
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)

    def similarity(self, intent_text: str, policy_text: str) -> float:
        """
        Compute semantic similarity between intent and policy text.

        Args:
            intent_text (str): Natural language intent (from multimodal fusion)
            policy_text (str): IBN-translated policy description

        Returns:
            float: cosine similarity score in [0, 1]
        """
        if not intent_text or not policy_text:
            return 0.0

        embeddings = self.model.encode(
            [intent_text, policy_text],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        sim_score = util.cos_sim(embeddings[0], embeddings[1]).item()
        return float(sim_score)


# --- Functional interface (used by SSIV score) ---

_semantic_validator = SemanticValidator()


def semantic_similarity(intent_text: str, policy_text: str) -> float:
    """
    Lightweight functional wrapper for SSIV.

    Used in:
        S_t = λ1 * sim_BERT(ϕ_t, π_t) + λ2 * consistency_GAT(...)
    """
    return _semantic_validator.similarity(intent_text, policy_text)
