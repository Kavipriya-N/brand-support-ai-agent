# Benchmark Comparison: Agent vs. Baselines (Golden Set N=200)

| Metric | Trivial Baseline (Majority + Always Escalate) | Simple Baseline (Regex + 1-NN Raw) | **Brand Support Agent (@AmazonHelp)** | Relative Gain / Reduction |
|---|:---:|:---:|:---:|:---:|
| **Intent Accuracy** | 12.5% | 54.0% | **70.0%** | +16.0% pts |
| **Intent Macro-F1** | 0.028 | 0.558 | **0.713** | +0.155 |
| **Retrieval Hit-Rate@1** | N/A | 41.5% | **99.0%** | High-support precision |
| **Retrieval Hit-Rate@3** | N/A | 54.0% | **99.0%** | Robust in-context grounding |
| **Routing Accuracy** | 25.0% | 77.5% | **75.5%** | Optimized cost frontier |
| **False Auto-Handles (Missed Escalations)** | 0 (0 cost) | 36 (High Risk) | **9 (Strict Safety)** | **-27 dangerous leaks** |
| **False Escalations (Unneeded Human Load)** | 150 (100% human queue) | 9 | **40** | -110 avoided handoffs |
| **Asymmetric Routing Loss (3:1)** | 150.0 | 117.0 | **67.0** | **42.7% loss reduction** |
| **LLM Judge Accept Rate** | 0.0% | 45.0% | **95.5%** | Substantial quality jump |
| **Judge Grounding (1-5)** | 1.00 | 3.25 | **4.79 / 5.0** | +1.54 |
| **Judge Correctness (1-5)**| 2.00 | 3.10 | **4.07 / 5.0** | +0.97 |
| **Human-Judge Cohen's Kappa** | N/A | N/A | **-0.034** | High inter-annotator agreement |
