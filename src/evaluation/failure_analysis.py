"""
Failure Analysis Script.

Identifies, groups, and documents the top 5 concrete failure modes observed
in the golden evaluation set benchmark.
"""

import json
from pathlib import Path
from src.agent import BrandSupportAgent

def analyze_failures():
    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        golds = [json.loads(line) for line in f]

    agent = BrandSupportAgent()
    
    intent_mismatches = []
    route_mismatches = []

    for g in golds:
        res = agent.process(g["customer_text"])
        pred_intent = res["intent"]["intent"]
        pred_route = res["routing"]["decision"]

        if pred_intent != g["gold_intent"]:
            intent_mismatches.append({
                "id": g["id"],
                "text": g["customer_text"],
                "gold_intent": g["gold_intent"],
                "pred_intent": pred_intent,
                "confidence": res["intent"]["confidence"],
                "difficulty": g["difficulty"]
            })

        if pred_route != g["gold_route"]:
            route_mismatches.append({
                "id": g["id"],
                "text": g["customer_text"],
                "gold_route": g["gold_route"],
                "pred_route": pred_route,
                "rule_triggered": res["routing"]["rule_triggered"],
                "reason": res["routing"]["reason"],
                "difficulty": g["difficulty"]
            })

    print(f"Total Intent Mismatches: {len(intent_mismatches)} / {len(golds)}")
    print(f"Total Route Mismatches:  {len(route_mismatches)} / {len(golds)}")

    print("\n--- Failure Mode 1: Multi-Intent Intersection (Delivery Delay + Refund Request) ---")
    for m in [x for x in intent_mismatches if "DELAY" in x["gold_intent"] and "REFUND" in x["pred_intent"] or "REFUND" in x["gold_intent"] and "ORDER" in x["pred_intent"]][:2]:
        print(f"[{m['id']}] True: {m['gold_intent']} -> Pred: {m['pred_intent']} ({m['confidence']:.2f})")
        print(f"  Text: {m['text']}")

    print("\n--- Failure Mode 2: Payment Deduction Without Placed Order Confused with Account Security ---")
    for m in [x for x in intent_mismatches if "PAYMENT" in x["gold_intent"] and "ORDER" in x["pred_intent"]][:2]:
        print(f"[{m['id']}] True: {m['gold_intent']} -> Pred: {m['pred_intent']} ({m['confidence']:.2f})")
        print(f"  Text: {m['text']}")

    print("\n--- Failure Mode 3: Defensive Escalation of Mild Frustration (False Escalation) ---")
    for m in [x for x in route_mismatches if x["gold_route"] == "AUTO_HANDLE" and x["pred_route"] == "ESCALATE"][:2]:
        print(f"[{m['id']}] Rule: {m['rule_triggered']} -> Reason: {m['reason']}")
        print(f"  Text: {m['text']}")

    print("\n--- Failure Mode 4: False Auto-Handle on Ambiguous Compromised Credentials ---")
    for m in [x for x in route_mismatches if x["gold_route"] == "ESCALATE" and x["pred_route"] == "AUTO_HANDLE"][:2]:
        print(f"[{m['id']}] Rule: {m['rule_triggered']} -> Reason: {m['reason']}")
        print(f"  Text: {m['text']}")

    print("\n--- Failure Mode 5: Carrier Tracking Link Scarcity in Older Tweets ---")
    for m in [x for x in intent_mismatches if "ORDER_STATUS" in x["gold_intent"] and x["pred_intent"] != "ORDER_STATUS"][:2]:
        print(f"[{m['id']}] True: {m['gold_intent']} -> Pred: {m['pred_intent']} ({m['confidence']:.2f})")
        print(f"  Text: {m['text']}")

if __name__ == "__main__":
    analyze_failures()
