"""
Intent Taxonomy Induction Module.

Performs unsupervised clustering and keyword extraction on customer inquiries
to derive an empirical 8-intent taxonomy specifically tailored to @AmazonHelp.
"""

import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans

INTENT_DEFINITIONS = {
    "ORDER_STATUS_TRACKING": {
        "description": "Inquiries regarding package location, dispatch status, tracking links, and estimated delivery dates.",
        "default_route": "AUTO_HANDLE",
        "route_reason": "Solvable via automated tracking link lookups and dispatch status summaries.",
        "keywords": ["track", "tracking", "status", "where", "dispatch", "dispatched", "eta", "shipped", "arrive", "when"]
    },
    "DELIVERY_DELAY_COMPLAINT": {
        "description": "Complaints that an order is late, missed guaranteed delivery date, or stuck in transit.",
        "default_route": "AUTO_HANDLE",
        "route_reason": "Can provide carrier delay policy and self-serve re-delivery or carrier inquiry steps.",
        "keywords": ["delay", "delayed", "late", "still not", "yesterday", "supposed", "waiting", "guaranteed", "today"]
    },
    "RETURN_REFUND_REQUEST": {
        "description": "Requests for returning items, tracking refund credit to bank, return labels, or pickup scheduling.",
        "default_route": "AUTO_HANDLE",
        "route_reason": "Standard automated return authorization, return center link, and refund SLA (3-5 business days).",
        "keywords": ["return", "refund", "pickup", "money back", "returned", "credit", "reimbursement", "label"]
    },
    "DIGITAL_KINDLE_PRIME_STREAMING": {
        "description": "Issues with Prime Video playback, Kindle device/books, Amazon Music, or digital subscriptions.",
        "default_route": "AUTO_HANDLE",
        "route_reason": "Can provide standard digital troubleshooting steps (app cache, license refresh, device re-registration).",
        "keywords": ["prime", "video", "kindle", "stream", "streaming", "music", "app", "book", "subscription", "fire"]
    },
    "ACCOUNT_LOGIN_SECURITY": {
        "description": "Account lockouts, OTP/2FA failures, unrecognized transactions, password reset, or hacked accounts.",
        "default_route": "ESCALATE",
        "route_reason": "Requires authenticated account verification and credential security protocol by human specialist.",
        "keywords": ["account", "password", "otp", "login", "locked", "hacked", "unauthorized", "verify", "security", "sign in"]
    },
    "DAMAGED_DEFECTIVE_ITEM": {
        "description": "Reports of damaged goods, broken packaging, defective items, or incorrect item received.",
        "default_route": "AUTO_HANDLE",
        "route_reason": "Can direct to automated replacement flow or replacement order generation without manual agent if within window.",
        "keywords": ["damage", "damaged", "broken", "defective", "wrong item", "faulty", "seal broken", "missing", "smashed"]
    },
    "PAYMENT_BILLING_INQUIRY": {
        "description": "Inquiries about double charges, payment failures, gift card balance, or declined cards.",
        "default_route": "AUTO_HANDLE",
        "route_reason": "Can clarify billing hold policies and provide bank reconciliation timelines before escalation.",
        "keywords": ["charge", "charged", "payment", "bank", "card", "decline", "declined", "gift card", "deducted", "billing"]
    },
    "GENERAL_POLICY_FEEDBACK": {
        "description": "Customer feedback, complaints about customer service reps, seller behavior, or broad store policies.",
        "default_route": "ESCALATE",
        "route_reason": "Subjective complaint or high-friction dispute requiring personalized human de-escalation.",
        "keywords": ["service", "terrible", "worst", "feedback", "agent", "seller", "policy", "experience", "rude", "complaint"]
    }
}

def induce_taxonomy(config_path: str = "configs/default.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    parquet_path = Path(cfg["data"]["processed_pairs_path"])
    print(f"[*] Loading clean pairs from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    texts = df["customer_text_clean"].tolist()
    print(f"    Loaded {len(texts):,} customer inquiries for taxonomy clustering.")

    # Vectorize customer text
    tfidf = TfidfVectorizer(
        max_features=3000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.8
    )
    X = tfidf.fit_transform(texts)
    terms = np.array(tfidf.get_feature_names_out())

    num_clusters = 8
    print(f"[*] Running MiniBatchKMeans (k={num_clusters})...")
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=256)
    cluster_labels = kmeans.fit_predict(X)
    df["cluster_id"] = cluster_labels

    cluster_info = {}
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

    for i in range(num_clusters):
        top_terms = [terms[ind] for ind in order_centroids[i, :8]]
        cluster_texts = df[df["cluster_id"] == i]["customer_text_clean"].head(5).tolist()
        cluster_info[f"cluster_{i}"] = {
            "top_terms": top_terms,
            "sample_inquiries": cluster_texts,
            "size": int((cluster_labels == i).sum())
        }
        print(f"    Cluster {i} ({cluster_info[f'cluster_{i}']['size']} rows): {', '.join(top_terms)}")

    taxonomy_output = {
        "intents": INTENT_DEFINITIONS,
        "empirical_clusters": cluster_info,
        "total_analyzed": len(texts)
    }

    output_path = Path("data/processed/induced_taxonomy.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(taxonomy_output, f, indent=2)

    print(f"[OK] Saved induced taxonomy with 8 brand intents to {output_path}")
    return taxonomy_output

if __name__ == "__main__":
    induce_taxonomy()
