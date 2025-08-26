In the streamlit_dashboard_app.py we want to add an AI-powered analysis.

Starting with the data that is filtered with the current filter settings we want to create a data set that can be submitted to an LLM for deep analysis and action recommendations.

We will want the ability to choose the LLM - and llm provider.


You’ll get the best results if you **don’t dump 20–50k raw rows into a single prompt**. Instead, let code do the heavy lifting (group-bys, time series deltas, anomaly flags), and use the LLM to **explain what changed, why, and where to look**, optionally pulling a few representative review quotes.

Here’s a battle-tested pattern that scales and stays accurate.

# 1) Pre-aggregate first (weekly or monthly)

Create a “metrics pack” per period × product (or product family). For each slice, compute:

* `n_reviews`, `avg_rating`, `p_pos`, `p_neu`, `p_neg`
* `problem_counts`: dict of {problem\_label: count}
* `positive_counts`: dict of {positive\_label: count}
* `top_terms` (optional): keyphrases from text (TF-IDF or YAKE)
* **deltas vs baseline**: percent change vs prior 4–8 weeks
* **z-scores / significance**: e.g., binomial test on change in negative share
* **anomaly flags**: e.g., any metric |z| ≥ 2 or sudden new label with support ≥ k and growth ≥ x%

Keep the raw reviews outside the prompt, but store **IDs & timestamps** so you can fetch a few example quotes when the LLM asks (or when a slice is flagged).

# 2) Feed the LLM compact JSON, not the whole CSV

Give it 200–1000 rows of **aggregates**, not 50k reviews. Typical unit: **weekly × top 50 SKUs** (or category-level if you have many SKUs). If you need depth, do it **hierarchically** (map-reduce): summarize per SKU → summarize per category → brand-level story.

### Recommended JSON schema (example)

```json
{
  "meta": {
    "brand": "Vuori",
    "period": "2023-06-01..2024-06-30",
    "granularity": "week",
    "min_support": 20,
    "anomaly_rules": {
      "z_abs_threshold": 2.0,
      "min_count": 15,
      "emerging_ratio": 2.0
    }
  },
  "series": [
    {
      "week": "2023-06-01",
      "product": "V302-Kore Short Lined 7\"",
      "sku": "V302BLKLRG",
      "n_reviews": 84,
      "avg_rating": 4.7,
      "p_pos": 0.86,
      "p_neu": 0.08,
      "p_neg": 0.06,
      "problem_counts": {"Fit": 3, "Liner": 5, "Chafe": 2},
      "positive_counts": {"Comfort": 41, "Material": 18, "Design": 12},
      "deltas": {
        "avg_rating_vs_8wk": -0.2,
        "p_neg_vs_8wk": +0.03,
        "Fit_vs_8wk": +150,
        "Liner_vs_8wk": +220
      },
      "z": {
        "p_neg": 2.4,
        "Fit": 2.1,
        "Liner": 2.8
      },
      "anomaly_flags": ["p_neg", "Liner"],
      "examples": {
        "neg_review_ids": ["r9031","r9142","r9160"],
        "pos_review_ids": ["r9050","r9057"]
      }
    }
  ]
}
```

# 3) Use a **structured prompt** with clear tasks & outputs

Give the model a firm role, the rules, the schema, and **exact outputs** (JSON + short narrative). Here’s a template you can drop into GPT/OpenAI-compatible or LM Studio:

**System**

```
You are a senior retail analytics copilot. You ONLY use facts in the provided JSON.
Prioritize statistically meaningful change. Avoid vague claims.
When you cite a reason, name the metric(s) and the period(s) that moved.
```

**User**

```
Goal: Find trends over time, surface anomalies, and explain likely drivers with supporting slices.

Data: <PASTE THE JSON "metrics pack" HERE>

Rules:
- Treat each row as period×product summary.
- An anomaly requires (a) min_support ≥ meta.min_support AND (b) either |z| ≥ meta.anomaly_rules.z_abs_threshold OR emerging topic growth ≥ meta.anomaly_rules.emerging_ratio.
- Prefer changes that persist ≥2 periods.
- Map problems/positives into themes if obvious (e.g., "Liner", "Chafe" → "Comfort/Fit").
- If you need examples, list the IDs in `examples` to fetch (do NOT invent quotes).
- If data are insufficient, say so explicitly.

Output (JSON then brief narrative):
{
  "brand_trends": [
    {"theme":"Comfort","direction":"up","evidence":[{"week":"2023-06-01","support":...}]}
  ],
  "sku_highlights": [
    {"sku":"V302...", "issue":"Liner complaints up", "metric":"Liner", "z":2.8, "delta_pct":"+220%", "weeks":["2023-06-01","2023-06-08"], "ask_for_examples":["r9031","r9142"]}
  ],
  "emerging_topics": [
    {"label":"Chafe","where":["V302"],"trend":"appeared_then_grew","support_path":[2,5,9]}
  ],
  "risk_watchlist": [
    {"sku":"VW303-L...", "reason":"p_neg rising 3w in a row", "action":"inspect sizing guidance & PDP imagery"}
  ],
  "positive_drivers": [
    {"theme":"Comfort","where":["V438","V126"],"evidence":"share ≥ 40% for 6 of last 8 weeks"}
  ]
}

Then add a 8–12 line narrative that tells the story in plain English.
```

> Tip: keep each prompt ≤ \~10–30k tokens by limiting `series` (e.g., last 8–12 weeks, top N SKUs by volume). Run multiple passes (per category) and combine with a final “summary-of-summaries” prompt.

# 4) (Optional) Map–Reduce for text: summaries & exemplars

If you want the model to include **representative quotes**:

* First pass (per week×product): summarize top 3 **problem** and **positive** themes with 1–2 quotes each (limit 200–400 tokens).
* Second pass: roll those summaries up (no raw text) to category → brand level.

This keeps prompts small and stable while preserving color.

# 5) How to compute the flags (minimal pandas sketch)

```python
import pandas as pd
from scipy.stats import binom_test
# df: columns = [date, product, product_description, rating, sentiment, problems_mentioned, positive_mentions]

df["week"] = pd.to_datetime(df["date"]).dt.to_period("W").dt.start_time
agg = (
  df.assign(
      neg=(df["sentiment"]=="Negative"),
      pos=(df["sentiment"]=="Positive"),
      neu=(df["sentiment"]=="Neutral")
  )
  .groupby(["week","product"], as_index=False)
  .agg(n_reviews=("sentiment","size"),
       avg_rating=("rating","mean"),
       neg=("neg","mean"),
       pos=("pos","mean"),
       neu=("neu","mean"))
)

# explode labels to count mentions
def split_labels(s):
    if pd.isna(s) or s=="None": return []
    return [x.strip() for x in str(s).split(";")]
labels = (df.assign(problem=lambda x: x["problems_mentioned"].apply(split_labels))
            .explode("problem")
            .dropna(subset=["problem"]))
lab = labels.groupby(["week","product","problem"]).size().reset_index(name="count")
# pivot to dict per row later

# rolling baselines & z-like scores
agg = agg.sort_values(["product","week"])
agg["neg_roll_mean"] = agg.groupby("product")["neg"].transform(lambda s: s.rolling(8, min_periods=4).mean())
agg["neg_roll_std"]  = agg.groupby("product")["neg"].transform(lambda s: s.rolling(8, min_periods=4).std())
agg["z_neg"] = (agg["neg"] - agg["neg_roll_mean"]) / agg["neg_roll_std"]

# simple significance example (binomial vs baseline)
def pval(row):
    if pd.isna(row.neg_roll_mean) or row.n_reviews < 15: return None
    k = int(round(row.neg*row.n_reviews))
    return binom_test(k, row.n_reviews, row.neg_roll_mean, alternative="two-sided")
agg["p_neg"] = agg.apply(pval, axis=1)
```

From here, assemble the JSON `series` rows, fold in `problem_counts` per (week, product), compute deltas vs 8-week rolling means, and mark anomalies.

# 6) Retrieval for drill-downs (optional but great)

Add an embeddings index (pgvector/FAISS). When the LLM surfaces an anomaly, fetch **3–5 most similar negative reviews** from the flagged slice and re-prompt: “Here are the requested examples. Update the narrative if needed.”

# 7) Guardrails that make outputs reliable

* **Schema-only outputs** (the LLM returns JSON you validate; then ask for the narrative).
* **Min support** (ignore changes on <15–20 reviews).
* **Canonical taxonomies** for `problems_mentioned`/`positive_mentions` (normalize synonyms before aggregation).
* **Encoding cleanup** (`‚Äô` → `'`, strip stray punctuation) before phrase extraction.

---

## TL;DR prompt + data recipe

1. Compute **weekly aggregates + anomaly stats** per SKU (and optionally per category).
2. Pass only those **compact JSON packs** to the LLM (not full text).
3. Ask for **structured JSON outputs** (trends, anomalies, watchlist, emerging topics) **plus a short narrative**.
4. On demand, fetch **example quotes** for any flagged slice and run a tiny follow-up prompt to enrich the story.

If you want, I can generate a small Python script that builds the JSON “metrics pack” from your exact columns and a matching prompt file you can post to LM Studio/OpenWebUI.
