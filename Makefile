# Makefile for Brand Support-Agent System (@AmazonHelp)
# Target runtime: < 15 minutes end-to-end

PYTHON = .venv/Scripts/python.exe
UVICORN = .venv/Scripts/uvicorn.exe

.PHONY: all setup ingest taxonomy train eval dashboard test clean

all: setup ingest taxonomy train eval

setup:
	@echo "[*] Setting up virtual environment..."
	uv venv .venv --python 3.12
	uv pip install --python $(PYTHON) pandas scikit-learn fastapi uvicorn pydantic httpx python-dotenv pyyaml pyarrow pytest

ingest:
	@echo "[*] Downloading dataset subsample and reconstructing threads..."
	$(PYTHON) src/ingestion/download.py
	$(PYTHON) -m src.ingestion.reconstruct_threads

taxonomy:
	@echo "[*] Inducing 8-intent taxonomy and building stratified golden set..."
	$(PYTHON) -m src.taxonomy.induce_intents
	$(PYTHON) -m src.taxonomy.build_golden_set

train:
	@echo "[*] Training intent classifier and building retrieval index..."
	$(PYTHON) -m src.classifier.intent_classifier
	$(PYTHON) -m src.retrieval.index

eval:
	@echo "[*] Running master evaluation harness against baselines & judge..."
	$(PYTHON) -m src.evaluation.run_eval

dashboard:
	@echo "[*] Launching Ops Reviewer Dashboard at http://127.0.0.1:8000 ..."
	$(PYTHON) -m uvicorn src.dashboard.app:app --host 127.0.0.1 --port 8000 --reload

test:
	@echo "[*] Running unit test suite..."
	$(PYTHON) -m pytest tests/ -v

clean:
	@echo "[*] Cleaning cached models and temporary data..."
	rm -rf data/processed/*.pkl data/processed/*.parquet
