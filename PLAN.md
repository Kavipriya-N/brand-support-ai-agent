# Project Execution Plan: Brand Support-Agent System (@AmazonHelp)

## Phase 1: Data Ingestion, Thread Reconstruction & Cleaning
- [x] 1.1 Download and slice Kaggle TWCS dataset (target ~25,000 to 50,000 @AmazonHelp interactions)
- [x] 1.2 Thread reconstruction: reconstruct conversation trees, pairing inbound customer tweets with brand resolution replies
- [x] 1.3 Cleaning & filtering: handle PII/usernames/order numbers, language filtering, deduplication
- [x] 1.4 Export processed issue-resolution dataset to `data/processed/amazonhelp_pairs.parquet`

## Phase 2: Intent Taxonomy Induction & Golden Set Construction
- [x] 2.1 Induce 8 brand-specific intents from clustering + semantic analysis of @AmazonHelp issues
- [x] 2.2 Construct golden evaluation set: 200 hand-audited, stratified examples (easy vs ambiguous, across all 8 intents) with gold intent, gold route (auto vs escalate), explicit routing reason, and gold reference resolution
- [x] 2.3 Document sampling methodology and labeling protocol in `data/sampling_and_labeling_note.md`

## Phase 3: Core Pipeline Implementation (Classifier, Retrieval Index, Router, Drafter)
- [x] 3.1 Intent Classifier: hybrid semantic classifier against the 8-class brand taxonomy
- [x] 3.2 Retrieval Index: vector store / cosine retrieval over historical issue->resolution pairs
- [x] 3.3 Routing Engine: multi-factor router (confidence, policy rules for PII/orders/account security, retrieval support score) with explicit human-inspectable reason string
- [x] 3.4 Grounded Reply Drafter: prompt-engineered drafter strictly conditioned on retrieved historical resolutions + intent

## Phase 4: Evaluation Harness, Baselines & LLM Judge
- [x] 4.1 Implement automated metrics: Intent accuracy/macro-F1, retrieval hit-rate@k, asymmetric routing precision/recall (cost of false auto-handle vs false escalate)
- [x] 4.2 Implement Trivial Baseline (majority class + always escalate) & Simple Baseline (keyword/regex matcher + raw nearest neighbor reply)
- [x] 4.3 Implement LLM-as-judge with explicit multi-criteria rubric (`prompts/judge_rubric.txt`: grounding, correctness, brand tone, actionability)
- [x] 4.4 Human-judge agreement evaluation on 40 sampled replies: calculate Cohen's kappa and confusion breakdown

## Phase 5: Dashboard & Ops Reviewer Tool
- [x] 5.1 FastAPI backend exposing `/api/inspect`, `/api/eval-results`, `/api/golden-samples`
- [x] 5.2 Dense ops review frontend with deliberate visual design: deep ink-teal (`#0B3B36`), bone background (`#FAF6EE`), amber accent (`#E0A63C`), distinctive typography, queue + full evidence inspection panel

## Phase 6: Report, Documentation, Decision Log & Verification
- [x] 6.1 Comprehensive `REPORT.md` (Problem framing & scope cuts, Results vs baselines, Top 5 failure analysis, "What is misleading about my headline number?", Next week roadmap)
- [x] 6.2 `DECISION_LOG.md` with 10-15 non-obvious engineering decisions logged as made
- [x] 6.3 `CITATIONS.md` with inline/source credits
- [x] 6.4 `Makefile` with <15 min reproduction commands (`make ingest`, `make eval`, `make dashboard`)
- [x] 6.5 Final self-audit against Definition of Done checklist
