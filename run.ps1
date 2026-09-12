# Quick Task Runner for Windows PowerShell (run.ps1)
param(
    [string]$Task = "eval"
)

$Python = ".venv\Scripts\python.exe"

switch ($Task) {
    "ingest" {
        Write-Host "[*] Ingestion & Thread Reconstruction..." -ForegroundColor Cyan
        & $Python src\ingestion\download.py
        & $Python -m src.ingestion.reconstruct_threads
    }
    "taxonomy" {
        Write-Host "[*] Taxonomy & Golden Set Construction..." -ForegroundColor Cyan
        & $Python -m src.taxonomy.induce_intents
        & $Python -m src.taxonomy.build_golden_set
    }
    "train" {
        Write-Host "[*] Classifier & Vector Index..." -ForegroundColor Cyan
        & $Python -m src.classifier.intent_classifier
        & $Python -m src.retrieval.index
    }
    "eval" {
        Write-Host "[*] Running Master Evaluation Harness..." -ForegroundColor Cyan
        & $Python -m src.evaluation.run_eval
    }
    "dashboard" {
        Write-Host "[*] Launching Ops Reviewer Dashboard..." -ForegroundColor Cyan
        & $Python -m uvicorn src.dashboard.app:app --host 127.0.0.1 --port 8000
    }
    "test" {
        Write-Host "[*] Running Test Suite..." -ForegroundColor Cyan
        & $Python -m pytest tests\ -v
    }
    Default {
        Write-Host "Usage: .\run.ps1 [ingest | taxonomy | train | eval | dashboard | test]" -ForegroundColor Yellow
    }
}
