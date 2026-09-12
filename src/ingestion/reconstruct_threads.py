"""
Thread Reconstruction Pipeline for @AmazonHelp.

Reconstructs (Customer Inquiry -> Brand Resolution) conversation pairs from raw TWCS data.
Extracts:
- customer_tweet_id, customer_raw_text, customer_clean_text
- brand_tweet_id, brand_raw_text, brand_clean_text
- thread_depth (single vs multi-turn)
- escalation_flag (did brand suggest DM/phone/call)
"""

import os
import re
import yaml
import pandas as pd
from pathlib import Path
try:
    from src.ingestion.cleaner import clean_text, is_english_tweet
except ModuleNotFoundError:
    from cleaner import clean_text, is_english_tweet

def reconstruct_pairs(config_path: str = "configs/default.yaml") -> pd.DataFrame:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_path = Path(cfg["data"]["raw_sample_path"])
    output_parquet = Path(cfg["data"]["processed_pairs_path"])
    output_csv = output_parquet.with_suffix(".csv")
    brand_name = cfg["brand"]["name"]
    max_pairs = cfg["data"].get("max_pairs_to_process", 5000)

    print(f"[*] Reading raw sample from {raw_path}...")
    df = pd.read_csv(
        raw_path,
        dtype={
            "tweet_id": str,
            "author_id": str,
            "inbound": "boolean",
            "created_at": str,
            "text": str,
            "response_tweet_id": str,
            "in_response_to_tweet_id": str,
        },
        on_bad_lines="skip"
    )
    print(f"    Raw tweets loaded: {len(df):,}")

    # Build tweet lookup dict for quick parent retrieval
    # tweet_id -> row dict
    print("[*] Indexing tweets by tweet_id...")
    tweet_map = df.set_index("tweet_id").to_dict(orient="index")

    # Filter brand replies
    brand_tweets = df[(df["inbound"] == False) & (df["author_id"] == brand_name)]
    print(f"    Found {len(brand_tweets):,} replies by {brand_name}")

    pairs = []
    seen_inquiries = set()

    dm_escalate_regex = re.compile(r"\b(dm|direct message|phone|call us|reach out to us directly|chat with us)\b", re.IGNORECASE)

    for _, row in brand_tweets.iterrows():
        parent_id = str(row["in_response_to_tweet_id"]).strip()
        if not parent_id or parent_id == "nan" or parent_id not in tweet_map:
            continue

        parent = tweet_map[parent_id]
        customer_text = str(parent.get("text", "")).strip()
        brand_reply_text = str(row.get("text", "")).strip()

        # Sanity checks
        if not customer_text or not brand_reply_text or len(customer_text) < 15:
            continue

        # Filter English only
        if not is_english_tweet(customer_text) or not is_english_tweet(brand_reply_text):
            continue

        clean_customer = clean_text(customer_text, keep_brand_handle=False)
        clean_brand = clean_text(brand_reply_text, keep_brand_handle=True, brand_handle=f"@{brand_name}")

        # Deduplicate identical customer text
        if clean_customer.lower() in seen_inquiries:
            continue
        seen_inquiries.add(clean_customer.lower())

        # Check multi-turn depth
        has_followup = pd.notna(row.get("response_tweet_id")) and str(row.get("response_tweet_id")) != "nan"
        is_dm_escalation = bool(dm_escalate_regex.search(brand_reply_text))

        pairs.append({
            "customer_tweet_id": parent_id,
            "brand_tweet_id": str(row["tweet_id"]),
            "customer_text_raw": customer_text,
            "customer_text_clean": clean_customer,
            "brand_reply_raw": brand_reply_text,
            "brand_reply_clean": clean_brand,
            "customer_created_at": str(parent.get("created_at", "")),
            "brand_created_at": str(row.get("created_at", "")),
            "has_followup": has_followup,
            "is_dm_escalation": is_dm_escalation
        })

        if len(pairs) >= max_pairs:
            break

    pairs_df = pd.DataFrame(pairs)
    print(f"[OK] Reconstructed {len(pairs_df):,} high-quality issue-resolution pairs.")
    print(f"    Multi-turn follow-ups: {pairs_df['has_followup'].mean():.1%}")
    print(f"    Direct DM/Phone escalations in brand reply: {pairs_df['is_dm_escalation'].mean():.1%}")

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pairs_df.to_parquet(output_parquet, index=False)
    pairs_df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"[OK] Saved processed pairs to:\n    - {output_parquet}\n    - {output_csv}")

    return pairs_df

if __name__ == "__main__":
    reconstruct_pairs()
