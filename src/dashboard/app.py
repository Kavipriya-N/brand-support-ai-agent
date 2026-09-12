"""
FastAPI Backend for Brand Support Ops Inspector.

Exposes REST APIs to inspect tweets end-to-end, retrieve benchmark metrics,
and view the golden set queue.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.agent import BrandSupportAgent
from src.evaluation.judge import LLMSupportJudge

app = FastAPI(title="Brand Support Ops Inspector (@AmazonHelp)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model singletons
agent: Optional[BrandSupportAgent] = None
judge: Optional[LLMSupportJudge] = None

def get_agent() -> BrandSupportAgent:
    global agent
    if agent is None:
        agent = BrandSupportAgent()
    return agent

def get_judge() -> LLMSupportJudge:
    global judge
    if judge is None:
        judge = LLMSupportJudge()
    return judge

class InspectRequest(BaseModel):
    text: str
    gold_intent: Optional[str] = None
    gold_route: Optional[str] = None
    reference_resolution: Optional[str] = None

@app.get("/api/health")
def health():
    return {"status": "healthy", "brand": "AmazonHelp"}

@app.get("/api/eval-results")
def get_eval_results():
    results_path = Path("data/processed/eval_results.json")
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="Evaluation results not found. Run make eval first.")
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/api/samples")
def get_samples(limit: int = 50):
    golden_path = Path("data/golden_set.jsonl")
    if not golden_path.exists():
        raise HTTPException(status_code=404, detail="Golden set not found.")
    samples = []
    with open(golden_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                samples.append(json.loads(line))
            if len(samples) >= limit:
                break
    return {"total": len(samples), "samples": samples}

@app.post("/api/inspect")
def inspect_message(req: InspectRequest):
    support_agent = get_agent()
    support_judge = get_judge()

    # Step 1-4: Agent Processing
    res = support_agent.process(req.text)

    # Step 5: Judge Evaluation
    gold_intent = req.gold_intent or res["intent"]["intent"]
    ref_res = req.reference_resolution or "Provide helpful, verified Amazon resolution instructions."
    
    judge_eval = support_judge.evaluate_reply(
        customer_text=req.text,
        gold_intent=gold_intent,
        reference_resolution=ref_res,
        route_decision=res["routing"]["decision"],
        route_reason=res["routing"]["reason"],
        historical_evidence=res["retrieval"]["top_hits"],
        candidate_reply=res["reply"]["draft"]
    )

    return {
        "customer_text": req.text,
        "gold_metadata": {
            "gold_intent": req.gold_intent,
            "gold_route": req.gold_route,
            "reference_resolution": req.reference_resolution
        },
        "intent": res["intent"],
        "retrieval": res["retrieval"],
        "routing": res["routing"],
        "reply": res["reply"],
        "judge": judge_eval
    }

# Mount static files if web directory exists
web_dir = Path("web")
if web_dir.exists():
    app.mount("/", StaticFiles(directory="web", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
