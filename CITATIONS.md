# Citations & Attributions

Every borrowed algorithm, library, dataset, prompt pattern, or data transformation is cited below with its source and rationale.

### 1. Datasets
- **Kaggle Customer Support on Twitter**:
  - Source: `thoughtvector/customer-support-on-twitter` (Kaggle) / Mirror: `SunidhiSriram/twcs` (Hugging Face).
  - Citation: Hardcastle, R. (2017). *Customer Support on Twitter*. Kaggle.
  - License: CC BY-NC-SA 4.0.
  - Usage: Raw dataset for historical customer issue tweets and brand agent responses.

### 2. Methodologies & Architectures
- **Retrieval-Augmented Generation (RAG)**:
  - Citation: Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020.
  - Usage: In-context grounding of draft responses using retrieved historical (inquiry, resolution) pairs.
- **LLM-as-a-Judge**:
  - Citation: Zheng, L., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS 2023 Datasets and Benchmarks Track.
  - Usage: Automated evaluation of drafted response quality across Grounding, Correctness, Tone, and Actionability.
- **Inter-Rater Reliability (Cohen's Kappa)**:
  - Citation: Cohen, J. (1960). *A Coefficient of Agreement for Nominal Scales*. Educational and Psychological Measurement, 20(1), 37-46.
  - Usage: Statistical calibration of agreement between human domain audit and automated LLM judge scoring.
- **Intent Taxonomy Induction via Clustering**:
  - Citation: Haponyk, A., et al. (2021). *Unsupervised Intent Induction from Customer Conversations*. EMNLP 2021.
  - Usage: Clustering and frequency-weighted keyword extraction to induce the 8-class brand intent taxonomy.

### 3. Software Libraries
- **Scikit-Learn**: Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, pp. 2825-2830. (TF-IDF vectorizer, cosine similarity, metrics).
- **Pandas**: Wes McKinney (2010). *Data Structures for Statistical Computing in Python*. Proc. of the 9th Python in Science Conf., pp. 51-56.
- **FastAPI**: Ramírez, S. (2018). *FastAPI Framework*. https://fastapi.tiangolo.com/.
