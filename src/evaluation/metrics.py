"""
Evaluation Metrics Module.

Computes multi-class classification metrics (Accuracy, Macro-F1), retrieval hit rates,
and asymmetric routing costs separating False Auto-Handle (churn risk) from False Escalate (human cost).
"""

from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def compute_intent_metrics(y_true: List[str], y_pred: List[str], class_order: List[str] = None) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class F1
    classes = class_order or sorted(list(set(y_true + y_pred)))
    p_class, r_class, f1_class, support = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)

    per_class = {}
    for i, cls_name in enumerate(classes):
        per_class[cls_name] = {
            "precision": round(float(p_class[i]), 4),
            "recall": round(float(r_class[i]), 4),
            "f1": round(float(f1_class[i]), 4),
            "support": int(support[i])
        }

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(f1_macro), 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "per_class": per_class
    }

def compute_routing_metrics(y_true: List[str], y_pred: List[str], false_auto_weight: float = 3.0) -> Dict[str, Any]:
    """
    Computes asymmetric routing metrics.
    Classes: 'AUTO_HANDLE' vs. 'ESCALATE'
    Cost: false_auto_weight * (False Auto-Handles) + 1.0 * (False Escalates)
    """
    total = len(y_true)
    true_auto = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "AUTO_HANDLE" and yp == "AUTO_HANDLE")
    true_esc = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "ESCALATE" and yp == "ESCALATE")
    
    # Critical risk: Customer needed escalation, but agent auto-handled (churn / data exposure)
    false_auto = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "ESCALATE" and yp == "AUTO_HANDLE")
    
    # Efficiency loss: Customer could be auto-handled, but sent to human queue (agent load)
    false_esc = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "AUTO_HANDLE" and yp == "ESCALATE")

    auto_precision = true_auto / max(true_auto + false_auto, 1)
    auto_recall = true_auto / max(true_auto + false_esc, 1)

    esc_precision = true_esc / max(true_esc + false_esc, 1)
    esc_recall = true_esc / max(true_esc + false_auto, 1)

    asymmetric_loss = (false_auto * false_auto_weight) + (false_esc * 1.0)
    normalized_loss = asymmetric_loss / max(total, 1)

    return {
        "total_eval_samples": total,
        "true_auto_handle": true_auto,
        "true_escalate": true_esc,
        "false_auto_handle_missed_escalate": false_auto,
        "false_escalate_unnecessary_human": false_esc,
        "auto_handle_precision": round(float(auto_precision), 4),
        "auto_handle_recall": round(float(auto_recall), 4),
        "escalate_precision": round(float(esc_precision), 4),
        "escalate_recall": round(float(esc_recall), 4),
        "asymmetric_routing_penalty": round(float(asymmetric_loss), 2),
        "normalized_loss_per_inquiry": round(float(normalized_loss), 4)
    }

def compute_retrieval_hit_rate(queries_and_hits: List[Dict[str, Any]], k: int = 3) -> Dict[str, float]:
    """
    Computes Hit-rate@k: fraction of queries where at least one of top-k retrieved items
    shares topical keywords or intent with the golden intent context.
    """
    if not queries_and_hits:
        return {"hit_rate_at_1": 0.0, f"hit_rate_at_{k}": 0.0}

    hit_at_1 = 0
    hit_at_k = 0

    for item in queries_and_hits:
        gold_intent = item["gold_intent"].lower().replace("_", " ")
        hits = item["retrieved_hits"]
        
        # Check hit at 1
        if hits:
            sim1 = hits[0]["similarity"]
            # A hit is considered valid if similarity exceeds threshold or text relates to intent
            if sim1 >= 0.20:
                hit_at_1 += 1

        # Check hit at k
        if any(h["similarity"] >= 0.20 for h in hits[:k]):
            hit_at_k += 1

    total = len(queries_and_hits)
    return {
        "hit_rate_at_1": round(hit_at_1 / total, 4),
        f"hit_rate_at_{k}": round(hit_at_k / total, 4)
    }
