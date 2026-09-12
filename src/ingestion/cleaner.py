"""
Data Cleaning & PII Sanitization Module.

Applies deterministic regex transformations to strip private customer data,
handles, URLs, tracking numbers, and order identifiers before indexing or model ingestion.
"""

import re
from typing import Tuple

# Patterns for scrubbing
HANDLE_PATTERN = re.compile(r"@[A-Za-z0-9_]+")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
AMAZON_ORDER_PATTERN = re.compile(r"\b\d{3}-\d{7}-\d{7}\b|\b\d{17}\b")
TRACKING_CODE_PATTERN = re.compile(r"\b[A-Z]{2}\d{9}[A-Z]{2}\b|\b1Z[0-9A-Z]{16}\b|\b\d{10,14}\b")
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
MULTIPLE_SPACES = re.compile(r"\s+")

def clean_text(text: str, keep_brand_handle: bool = False, brand_handle: str = "@AmazonHelp") -> str:
    """
    Sanitizes raw tweet text:
    - Replaces order IDs with [ORDER_ID]
    - Replaces URLs with [LINK]
    - Replaces emails with [EMAIL]
    - Scrubs handles (@115858 -> [USER], keeps brand if requested)
    - Normalizes whitespace
    """
    if not isinstance(text, str):
        return ""

    # Replace Amazon order numbers first
    text = AMAZON_ORDER_PATTERN.sub("[ORDER_ID]", text)
    
    # Replace URLs
    text = URL_PATTERN.sub("[LINK]", text)

    # Replace emails
    text = EMAIL_PATTERN.sub("[EMAIL]", text)

    # Replace tracking codes
    text = TRACKING_CODE_PATTERN.sub("[TRACKING_CODE]", text)

    # Normalize or scrub handles
    if keep_brand_handle:
        # Keep @AmazonHelp, replace user handles
        def _replace_handle(match):
            h = match.group(0)
            if h.lower() == brand_handle.lower():
                return brand_handle
            return "[USER]"
        text = HANDLE_PATTERN.sub(_replace_handle, text)
    else:
        text = HANDLE_PATTERN.sub("[USER]", text)

    # Normalize whitespace
    text = MULTIPLE_SPACES.sub(" ", text).strip()
    return text

def is_english_tweet(text: str) -> bool:
    """
    Fast heuristic to check if tweet is primarily English:
    Checks printable ASCII character ratio and common English stopwords.
    """
    if not text or len(text.strip()) < 5:
        return False
    
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    if ascii_chars / len(text) < 0.85:
        return False
        
    # Quick token check for high-frequency English functional words
    lower = text.lower()
    english_tokens = {"the", "i", "to", "my", "is", "a", "and", "in", "it", "you", "for", "on", "order", "delivery", "help", "amazon"}
    words = set(re.findall(r"\b[a-z]{2,}\b", lower))
    return len(words.intersection(english_tokens)) >= 1
