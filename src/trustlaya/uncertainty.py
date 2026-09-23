def confidence(probabilities):
    # Decision certainty is separate from risk magnitude.
    return max(0.0, min(1.0, 2 * max(abs(float(p)-0.5) for p in probabilities)))
def should_abstain(confidence_value, threshold=0.60):
    return confidence_value < threshold
