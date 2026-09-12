# Decision Log: Brand Support-Agent System

This log documents non-obvious engineering decisions, trade-offs, and design rationales made incrementally throughout development.

| # | Decision | Rationale |
|---|---|---|
| 1 | **Selected `@AmazonHelp` as target brand** | Empirical profiling showed #1 volume (>500k in TWCS), highest thread depth (55.4% multi-turn), and a rich 79.8% self-serve vs. 20.2% escalation distribution compared to AppleSupport (74.5% direct DM redirect). |
| 2 | **Use a streaming subsample of TWCS (~45MB / 5,000 brand pairs)** | Enables full reproduction in <15 minutes while preserving statistical validity and representative intent coverage. |
| 3 | **Exclude non-English tweets (`en` filter + ASCII heuristic)** | @AmazonHelp provides multilingual support (ES, DE, JA, FR), but standardizing on English prevents multilingual confounding in intent clustering. |
| 4 | **Thread reconstruction strategy: first inbound to first direct brand reply** | Captures the core issue-to-resolution handoff without noise from prolonged customer-agent debate or delayed post-resolution chatter. |
| 5 | **Scrub customer handles and order numbers via regex redaction** | Strips `@115858` placeholders and 17-digit order IDs (`###-#######-#######`) to prevent lexical memorization in retrieval. |
| 6 | **8-class induced intent taxonomy** | 8 intents balance specificity (e.g., separating delay vs. damage vs. return) with class sample density, avoiding high-variance micro-categories. |
| 7 | **Zero-leakage split: separate 200-sample golden set before retrieval indexing** | 200 evaluation items were completely excised from the 4,800 candidate corpus before building vector retrieval to prevent test contamination. |
| 8 | **Asymmetric routing loss function ($3\times$ False Auto-Handle penalty)** | False Auto-Handle (missed escalation requiring human) is penalized at $3\times$ the cost of False Escalation (unnecessary human review) due to customer churn risk. |
| 9 | **Explicit inspectable routing reasons instead of opaque confidence floats** | Production support operations require clear audit trails (e.g., rule triggered, keyword matched) for compliance and supervisor review. |
| 10 | **Disjoint LLM Judge architecture with external prompt file** | Rubric evaluation uses an isolated prompt (`prompts/judge_rubric.txt`) and independent scoring anchors, preventing generator self-preference bias. |
| 11 | **Local offline-capable vector retrieval & evaluation fallback** | Ensures skeptical reviewers can reproduce all headline numbers in under 15 minutes with zero API key or network prerequisites. |
| 12 | **Calibrated human agreement study on 40 stratified samples** | Evaluated inter-rater agreement to measure judge reliability, which empirically surfaced the judge's affirmative leniency bias ($\kappa = -0.034$). |
| 13 | **Unconventional visual design system (Ink-teal `#0B3B36` & Bone Paper `#FAF6EE`)** | Replaced generic framework defaults (Bootstrap blue, Tailwind slate) with an authoritative, brand-appropriate ops palette with high-contrast data density. |
| 14 | **Asymmetric route badge styling (Teal `#0E6251` vs Amber `#B45309`)** | Avoided cliché green/red traffic-light styling in favor of operational state indicators that distinguish routine auto-handling from human escalation without alarmism. |
