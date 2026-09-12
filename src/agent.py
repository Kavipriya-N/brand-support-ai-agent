"""
Unified Support Agent Pipeline for @AmazonHelp.

Executes end-to-end processing for an incoming customer inquiry:
1. Intent Classification
2. Historical Resolution Retrieval
3. Rule + Confidence + Support Routing
4. Grounded Reply Drafting
"""

from typing import Dict, Any
from src.classifier.intent_classifier import IntentClassifier
from src.retrieval.index import ResolutionRetrievalIndex
from src.router.engine import SupportRouter
from src.drafter.generator import ReplyDrafter

class BrandSupportAgent:
    def __init__(self, config_path: str = "configs/default.yaml"):
        self.classifier = IntentClassifier(config_path=config_path).fit_or_load()
        self.retriever = ResolutionRetrievalIndex(config_path=config_path).build_or_load()
        self.router = SupportRouter(config_path=config_path)
        self.drafter = ReplyDrafter(config_path=config_path)

    def process(self, customer_text: str) -> Dict[str, Any]:
        """
        Executes end-to-end agent decision pipeline for a single customer inquiry.
        """
        # Step 1: Classify intent
        intent_info = self.classifier.predict(customer_text)

        # Step 2: Retrieve historical evidence
        retrieval_hits = self.retriever.query(customer_text, top_k=3)

        # Step 3: Route decision (Auto-handle vs. Escalate)
        route_info = self.router.route(customer_text, intent_info, retrieval_hits)

        # Step 4: Draft grounded reply
        draft_info = self.drafter.draft(customer_text, intent_info, route_info, retrieval_hits)

        return {
            "customer_text": customer_text,
            "intent": intent_info,
            "retrieval": {
                "top_hits": retrieval_hits,
                "support_score": route_info["support_score"]
            },
            "routing": route_info,
            "reply": draft_info
        }

if __name__ == "__main__":
    agent = BrandSupportAgent()
    sample = "Where is my package? The tracking page has not updated for 3 days."
    res = agent.process(sample)
    print("\n=== End-to-End Agent Response ===")
    print(f"Customer: {res['customer_text']}")
    print(f"Intent: {res['intent']['intent']} (Conf: {res['intent']['confidence']})")
    print(f"Route: {res['routing']['decision']} -> {res['routing']['reason']}")
    print(f"Draft: {res['reply']['draft']}")
