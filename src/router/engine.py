"""
Routing Engine Module.

Makes deterministic, inspectable routing decisions (AUTO_HANDLE vs. ESCALATE)
based on intent classification confidence, policy rules, high-risk security keywords,
and retrieval evidence support scores.
"""

import re
import yaml
from typing import Dict, Any, List

class SupportRouter:
    def __init__(self, config_path: str = "configs/default.yaml"):
        self.config_path = config_path
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.conf_threshold = self.cfg["router"].get("auto_handle_threshold", 0.40)
        self.min_support = self.cfg["retrieval"].get("min_relevance_threshold", 0.18)
        self.escalation_keywords = self.cfg["router"].get("escalation_keywords", [
            "hacked", "unauthorized", "lawsuit", "fraud", "police", "lawyer", "scam", "urgent", "stolen card"
        ])
        self.keyword_regex = re.compile(r"\b(" + "|".join(re.escape(k) for k in self.escalation_keywords) + r")\b", re.IGNORECASE)

        # Policy intent mappings
        self.policy_escalate_intents = {"ACCOUNT_LOGIN_SECURITY", "GENERAL_POLICY_FEEDBACK"}

    def route(self, customer_text: str, intent_info: Dict[str, Any], retrieval_hits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Determines routing decision and attaches human-inspectable explanation.
        """
        intent = intent_info.get("intent", "ORDER_STATUS_TRACKING")
        confidence = float(intent_info.get("confidence", 0.0))
        top_similarity = float(retrieval_hits[0]["similarity"]) if retrieval_hits else 0.0

        # Rule 1: High-risk security/legal keywords
        kw_match = self.keyword_regex.search(customer_text)
        if kw_match:
            matched_kw = kw_match.group(0).lower()
            return {
                "decision": "ESCALATE",
                "rule_triggered": "SECURITY_KEYWORD_MATCH",
                "reason": f"Escalated to human: High-risk security/legal keyword '{matched_kw}' detected requiring human compliance review.",
                "confidence": confidence,
                "support_score": top_similarity
            }

        # Rule 2: Policy-mandated escalation intents
        if intent in self.policy_escalate_intents and confidence >= 0.35:
            return {
                "decision": "ESCALATE",
                "rule_triggered": "POLICY_MANDATED_ESCALATION",
                "reason": f"Escalated to human: Intent '{intent}' mandates authenticated human specialist handling per Amazon policy.",
                "confidence": confidence,
                "support_score": top_similarity
            }

        # Rule 3: Low Intent Confidence (Ambiguity)
        if confidence < self.conf_threshold:
            return {
                "decision": "ESCALATE",
                "rule_triggered": "LOW_CONFIDENCE_FALLBACK",
                "reason": f"Escalated to human: Classification confidence ({confidence:.2f}) below auto-handle threshold ({self.conf_threshold:.2f}); inquiry is ambiguous.",
                "confidence": confidence,
                "support_score": top_similarity
            }

        # Rule 4: Weak retrieval evidence grounding
        if top_similarity < self.min_support:
            return {
                "decision": "ESCALATE",
                "rule_triggered": "POOR_RETRIEVAL_SUPPORT",
                "reason": f"Escalated to human: Historical retrieval support ({top_similarity:.2f}) below threshold ({self.min_support:.2f}); insufficient precedent for automated reply.",
                "confidence": confidence,
                "support_score": top_similarity
            }

        # Rule 5: Standard verified auto-handle
        return {
            "decision": "AUTO_HANDLE",
            "rule_triggered": "STANDARD_AUTO_HANDLE",
            "reason": f"Auto-handled: High-confidence ({confidence:.2f}) '{intent}' inquiry with strong historical precedent (similarity {top_similarity:.2f}).",
            "confidence": confidence,
            "support_score": top_similarity
        }

if __name__ == "__main__":
    router = SupportRouter()
    sample_text = "Someone hacked my account and placed unauthorized orders!"
    sample_intent = {"intent": "ACCOUNT_LOGIN_SECURITY", "confidence": 0.88}
    sample_hits = [{"similarity": 0.52, "brand_reply": "Please visit amazon.com/help"}]
    res = router.route(sample_text, sample_intent, sample_hits)
    print(f"Sample Route Result:\n{res}")
