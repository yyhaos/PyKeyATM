"""Small output helpers used by the runnable examples."""


def print_fit_summary(fit):
    """Print the main fitted objects without dumping internal sampler state."""
    theta = fit["theta"]
    phi = fit["phi"]
    print(f"Model: {fit['model']}")
    print(f"Documents: {fit['N']} | Vocabulary: {fit['V']} | Topics: {theta.shape[1]}")
    print("\nDocument-topic proportions (theta):")
    print(theta.round(3).to_string())
    print("\nTopic-word probabilities (phi):")
    print(phi.round(3).to_string())
    print("\nTop words by topic:")
    for topic, row in phi.iterrows():
        top_words = row.sort_values(ascending=False).head(3).index.tolist()
        print(f"  {topic}: {', '.join(top_words)}")
    if not fit["model_fit"].empty:
        print("\nSampling diagnostics:")
        print(fit["model_fit"].tail(1).round(3).to_string(index=False))

