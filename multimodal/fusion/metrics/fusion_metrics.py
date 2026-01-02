import torch
import torch.nn.functional as F


# ------------------------------------------------------------------------------
# Cosine Similarity (Metric)
# ------------------------------------------------------------------------------
def cosine_similarity(x, y):
    """
    Computes cosine similarity between two vectors.
    """
    x_norm = F.normalize(x, dim=-1)
    y_norm = F.normalize(y, dim=-1)
    return (x_norm * y_norm).sum(dim=-1)


# ------------------------------------------------------------------------------
# Contrastive Loss
# ------------------------------------------------------------------------------
def contrastive_loss(embeddings, labels, margin=1.0):
    batch_size = embeddings.size(0)

    losses = []

    for i in range(batch_size):
        for j in range(i + 1, batch_size):

            sim = cosine_similarity(embeddings[i], embeddings[j])
            label = labels[i] == labels[j]

            if label:  # same class → maximize similarity
                loss = 1 - sim
            else:       # different classes → enforce margin
                loss = torch.clamp(sim - margin, min=0)

            losses.append(loss)

    if len(losses) == 0:
        return torch.tensor(0.0, requires_grad=True)

    return torch.stack(losses).mean()


# ------------------------------------------------------------------------------
# Triplet Loss
# ------------------------------------------------------------------------------
def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Triplet Loss:
        L = max(0, d(a,p) - d(a,n) + margin)

    Args:
        anchor   : embedding tensor
        positive : embedding tensor (same class)
        negative : embedding tensor (different class)
    """

    pos_dist = 1 - cosine_similarity(anchor, positive)
    neg_dist = 1 - cosine_similarity(anchor, negative)

    loss = torch.clamp(pos_dist - neg_dist + margin, min=0)
    return loss.mean()


# ------------------------------------------------------------------------------
# Embedding Variance (diagnostic metric)
# ------------------------------------------------------------------------------
def embedding_variance(embeddings):
    """
    Returns the average variance across embedding dimensions.
    Useful for checking embedding collapse or over-dispersion.
    """
    return torch.var(embeddings, dim=0).mean().item()
