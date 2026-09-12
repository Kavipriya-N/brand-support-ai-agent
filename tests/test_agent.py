"""
Sanity and Unit Test Suite for Brand Support Agent Pipeline.
"""

import pytest
from src.ingestion.cleaner import clean_text, is_english_tweet
from src.classifier.intent_classifier import IntentClassifier
from src.retrieval.index import ResolutionRetrievalIndex
from src.router.engine import SupportRouter
from src.drafter.generator import ReplyDrafter
from src.agent import BrandSupportAgent

def test_pii_sanitization():
    raw_tweet = "@115858 My order 123-4567890-1234567 was stolen by user @john! Visit https://amazon.com/track for details."
    cleaned = clean_text(raw_tweet)
    assert "123-4567890-1234567" not in cleaned
    assert "[ORDER_ID]" in cleaned
    assert "[USER]" in cleaned
    assert "[LINK]" in cleaned
    assert "@john" not in cleaned

def test_english_filter():
    assert is_english_tweet("Where is my package? It is late.") == True
    assert is_english_tweet("Hola por favor ayuda con mi pedido") == False

def test_intent_classification():
    clf = IntentClassifier().fit_or_load()
    res = clf.predict("Where is my package? When will it be dispatched?")
    assert res["intent"] == "ORDER_STATUS_TRACKING"
    assert res["confidence"] > 0.40
    assert "ORDER_STATUS_TRACKING" in res["distribution"]

def test_security_escalation_routing():
    router = SupportRouter()
    res = router.route(
        customer_text="Someone hacked my password and made fraudulent charges!",
        intent_info={"intent": "ACCOUNT_LOGIN_SECURITY", "confidence": 0.85},
        retrieval_hits=[{"similarity": 0.35, "brand_reply": "Reach out directly"}]
    )
    assert res["decision"] == "ESCALATE"
    assert "hacked" in res["reason"]
    assert res["rule_triggered"] == "SECURITY_KEYWORD_MATCH"

def test_end_to_end_agent():
    agent = BrandSupportAgent()
    res = agent.process("How do I return a damaged shirt for a refund?")
    assert "intent" in res
    assert "retrieval" in res
    assert "routing" in res
    assert "reply" in res
    assert len(res["reply"]["draft"]) > 10
    assert "[USER]" in res["reply"]["draft"] or "@AmazonHelp" in res["reply"]["draft"]
