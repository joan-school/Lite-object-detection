def confidence_gate(results, threshold=0.5):
    return [r for r in results if r["confidence"] > threshold]
