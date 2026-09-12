"""
Baseline Models for Comparative Evaluation.

Implements:
1. TrivialBaseline: Majority-class intent + Always-Escalate route + Generic static canned response.
2. SimpleBaseline: Keyword/regex intent matcher + Heuristic keyword routing + 1-NN raw historical reply without LLM drafting.
"""

import re
from typing import Dict, Any, List
from src.retrieval.index import ResolutionRetrievalIndex

class TrivialBaseline:
    """Trivial Baseline: Predicts majority class and always escalates to human."""
    def __init__(self, majority_intent: str = "ORDER_STATUS_TRACKING"):
        self.majority_intent = majority_intent

    def process(self, text: str) -> Dict[str, Any]:
        return {
            "intent": {"intent": self.majority_intent, "confidence": 0.50},
            "routing": {
                "decision": "ESCALATE",
                "reason": "Trivial Baseline: Always-escalate policy.",
                "rule_triggered": "ALWAYS_ESCALATE_BASELINE"
            },
            "reply": {
                "draft": "[USER] Thank you for contacting Amazon Help. Please check your order status online at [LINK] or contact our help team.",
                "generation_mode": "trivial-canned-response"
            }
        }

class SimpleBaseline:
    """Simple Baseline: Lexical regex intent matcher + simple heuristic router + 1-NN raw reply."""
    def __init__(self, retrieval_index: ResolutionRetrievalIndex = None):
        self.retriever = retrieval_index or ResolutionRetrievalIndex().build_or_load()
        self.rules = [
            ("ACCOUNT_LOGIN_SECURITY", re.compile(r"\b(password|login|otp|locked|hacked|sign in)\b", re.IGNORECASE)),
            ("RETURN_REFUND_REQUEST", re.compile(r"\b(return|refund|pickup|returned)\b", re.IGNORECASE)),
            ("DELIVERY_DELAY_COMPLAINT", re.compile(r"\b(delay|delayed|late|waiting|yesterday)\b", re.IGNORECASE)),
            ("DAMAGED_DEFECTIVE_ITEM", re.compile(r"\b(damaged|broken|seal|defective|wrong item)\b", re.IGNORECASE)),
            ("DIGITAL_KINDLE_PRIME_STREAMING", re.compile(r"\b(prime video|kindle|stream|ebook|music)\b", re.IGNORECASE)),
            ("PAYMENT_BILLING_INQUIRY", re.compile(r"\b(charge|payment|card|decline|deducted|billing)\b", re.IGNORECASE)),
            ("GENERAL_POLICY_FEEDBACK", re.compile(r"\b(worst|terrible|rude|agent|policy|feedback)\b", re.IGNORECASE)),
            ("ORDER_STATUS_TRACKING", re.compile(r"\b(order|track|tracking|status|dispatch|where)\b", re.IGNORECASE))
        ]

    def process(self, text: str) -> Dict[str, Any]:
        # Simple regex matcher
        predicted_intent = "ORDER_STATUS_TRACKING"  # fallback
        for intent_name, regex in self.rules:
            if regex.search(text):
                predicted_intent = intent_name
                break

        # Simple keyword router: escalate if contains "human", "agent", "account", or "worst"
        lower = text.lower()
        if any(w in lower for w in ["human", "agent", "account", "worst", "hacked", "stolen", "lawsuit"]):
            route_decision = "ESCALATE"
            route_reason = "Simple Baseline: Escalate keyword detected."
        else:
            route_decision = "AUTO_HANDLE"
            route_reason = "Simple Baseline: Default auto-handle."

        # Raw nearest neighbor reply (no LLM rewriting or safety adaptation)
        hits = self.retriever.query(text, top_k=1)
        raw_reply = hits[0]["brand_reply"] if hits else "[USER] Please visit [LINK] for support."

        return {
            "intent": {"intent": predicted_intent, "confidence": 0.50},
            "routing": {
                "decision": route_decision,
                "reason": route_reason,
                "rule_triggered": "KEYWORD_HEURISTIC_BASELINE"
            },
            "reply": {
                "draft": raw_reply,
                "generation_mode": "1nn-raw-historical-reply"
            }
        }
