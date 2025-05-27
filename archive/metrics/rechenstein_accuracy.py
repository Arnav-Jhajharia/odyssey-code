from itertools import permutations

def clustering_accuracy(C, C_tilde, gamma=0.8):
    """
    Compute clustering accuracy A_γ(C, C_tilde) as defined in the DNA data storage paper.

    Args:
        C (list of sets): Ground truth clusters.
        C_tilde (list of sets): Predicted clusters.
        gamma (float): Fraction threshold, must be in (0.5, 1].

    Returns:
        float: Accuracy score in [0, 1].
    """
    assert 0.5 < gamma <= 1.0, "γ must be in the range (0.5, 1]"
    
    k = len(C)
    m = len(C_tilde)
    max_matches = 0

    # Try all injective mappings π: {1..m} → {1..k}
    for pi in permutations(range(k), min(k, m)):
        matches = 0
        for i, j in enumerate(pi):
            C_pred = C_tilde[i]
            C_true = C[j]
            # Conditions: subset and sufficient overlap
            if C_pred.issubset(C_true) and len(C_pred & C_true) >= gamma * len(C_true):
                matches += 1
        max_matches = max(max_matches, matches)

    return max_matches / k
