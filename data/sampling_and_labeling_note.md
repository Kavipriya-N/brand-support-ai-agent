# Golden Evaluation Set: Sampling Methodology & Annotation Protocol

**Author**: Senior Applied-AI Engineer  
**Dataset**: Kaggle Customer Support on Twitter (`thoughtvector/customer-support-on-twitter` / `@AmazonHelp`)  
**Target Size**: 200 hand-audited, stratified evaluation examples  
**Artifact File**: `data/golden_set.jsonl`

---

## 1. Stratification Strategy

The golden evaluation set is stratified across two orthogonal dimensions:

1. **Intent Taxonomy (8 Classes)**:
   - `ORDER_STATUS_TRACKING`: 25 examples
   - `DELIVERY_DELAY_COMPLAINT`: 25 examples
   - `RETURN_REFUND_REQUEST`: 25 examples
   - `DIGITAL_KINDLE_PRIME_STREAMING`: 25 examples
   - `ACCOUNT_LOGIN_SECURITY`: 25 examples
   - `DAMAGED_DEFECTIVE_ITEM`: 25 examples
   - `PAYMENT_BILLING_INQUIRY`: 25 examples
   - `GENERAL_POLICY_FEEDBACK`: 25 examples
   **Total**: $8 \times 25 = 200$ examples.

2. **Difficulty Spectrum (Easy vs. Ambiguous)**:
   - **Easy (~60% / 120 items)**: Canonical phrasing, prominent intent markers, explicit single-issue complaints (e.g. *"Where is my package with tracking code XYZ?"*).
   - **Ambiguous (~40% / 80 items)**: Multi-intent intersections (e.g. delayed delivery + requesting refund, or wrong item + demanding account compensation), sarcastic remarks, terse elliptical queries, or emotional complaints requiring boundary disambiguation.

---

## 2. Labeling Schema & Ground Truth Definitions

Each item in `data/golden_set.jsonl` contains the following strict schema:

```json
{
  "id": "eval_001",
  "customer_tweet_id": "115858",
  "customer_text": "My package says delivered today on the tracking page but nothing was left at my front door or mailbox!",
  "difficulty": "easy",
  "gold_intent": "ORDER_STATUS_TRACKING",
  "gold_route": "AUTO_HANDLE",
  "gold_route_reason": "Missing package marked delivered can be automated via standard 36-hour carrier grace window protocol and neighbor/safe-place check guidance.",
  "reference_resolution": "Advise checking with neighbors/household members, waiting up to 36 hours for carrier scans, and viewing the delivery photo in Your Orders.",
  "historical_brand_reply": "We'd like to help! Please check with family members or neighbors. If still missing after 36 hours, reach out via [LINK]."
}
```

---

## 3. Routing Policy Guidelines (Auto-Handle vs. Escalate)

To avoid subjective drift, routing decisions follow an explicit policy matrix:

| Condition | Gold Decision | Canonical Reason String |
|---|---|---|
| Inquiries solvable via public URL, tracking instructions, self-service return center, or digital device reset | **`AUTO_HANDLE`** | Standard self-service workflow supported by documented brand protocols. |
| Account security, compromised credentials, 2FA/OTP failures, unrecognized financial transactions | **`ESCALATE`** | Strict account security protocol requires authenticated human specialist verification. |
| Extreme customer frustration, threats of legal action/regulatory complaints, abusive seller conduct | **`ESCALATE`** | High-friction churn risk or severe policy dispute requiring human de-escalation. |
| Highly entangled multi-issue inquiries where self-service actions conflict | **`ESCALATE`** | Ambiguous multi-system edge case exceeding automated boundary certainty. |

---

## 4. Quality Audit & Validation Checks

- **Zero Data Leakage**: The 200 golden eval items are strictly partitioned and removed from the candidate retrieval pool during evaluation to guarantee out-of-sample retrieval integrity.
- **Syntactic Validity**: Verified UTF-8 encoding, valid JSONL schema, non-empty fields, and validated categorical values.
