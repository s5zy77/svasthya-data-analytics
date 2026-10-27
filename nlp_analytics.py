"""
nlp_analytics.py
================
NLP + data-mining pipeline for Svasthya community posts.

Sections
--------
1. Text preprocessing & medical entity extraction (lexicon-based, negation-aware)
2. Hashtag co-occurrence mining (Jaccard / phi / lift)
3. Urgency & sentiment scoring (continuous 0-100 score)
4. Outbreak detection (rolling 7-day window, robust z-score vs. historical baseline)
5. Unsupervised clustering (TF-IDF -> LSA -> K-Means)
6. Supervised triage model + rule-based red-flag safety net

Everything is deterministic given a seed and degrades gracefully if NLTK corpora
cannot be downloaded (falls back to scikit-learn stop-words and a rule lemmatizer).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix, f1_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

SEVERITY_ORDER: list[str] = ["Mild", "Moderate", "Severe", "Emergency"]
SEVERITY_TO_NUM: dict[str, int] = {s: i + 1 for i, s in enumerate(SEVERITY_ORDER)}

# =====================================================================================
# 1. TEXT PREPROCESSING & MEDICAL ENTITY EXTRACTION
# =====================================================================================

# Words that are frequent in forum posts but carry no clinical/topic information.
DOMAIN_STOPWORDS = {"hi", "everyone", "please", "help", "advice", "need", "sure", "anyone", "experienced",
                    "concerned", "should", "know", "like", "really", "also", "feeling", "feel", "right", "now",
                    "since", "past", "ive", "been", "dealing", "having", "experiencing", "week", "day", "days"}
_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


@lru_cache(maxsize=1)
def _stopwords() -> frozenset[str]:
    """NLTK English stop-words if available, otherwise scikit-learn's list (minus a few clinical words)."""
    words: set[str]
    try:
        import nltk
        from nltk.corpus import stopwords
        try:
            words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            words = set(stopwords.words("english"))
    except Exception:
        words = set(ENGLISH_STOP_WORDS) - {"back", "side", "front", "top", "bottom", "cry", "fill"}
    return frozenset(words | DOMAIN_STOPWORDS)


@lru_cache(maxsize=1)
def _lemmatizer():
    """WordNet lemmatizer, or None if the corpus is unavailable (rule-based fallback is used)."""
    try:
        from nltk.stem import WordNetLemmatizer
        wnl = WordNetLemmatizer()
        try:
            wnl.lemmatize("tests")
        except LookupError:
            import nltk
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            wnl.lemmatize("tests")
        return wnl
    except Exception:
        return None


def _rule_lemma(tok: str) -> str:
    """Tiny suffix-stripping fallback (bumps->bump, vomiting->vomit, allergies->allergy)."""
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 5 and tok.endswith("ing"):
        return tok[:-3]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


@lru_cache(maxsize=20000)
def _lemma(tok: str) -> str:
    wnl = _lemmatizer()
    if wnl is None:
        return _rule_lemma(tok)
    # noun pass then verb pass: "bumps"->"bump", "vomiting"->"vomit"
    return wnl.lemmatize(wnl.lemmatize(tok, "n"), "v")


def normalize_text(text: str) -> str:
    """
    Light normalisation used by the *supervised* triage model: lowercase, drop URLs and hashtags,
    keep stop-words (n-grams like "not helping" matter for severity).
    Hashtags are removed on purpose so the model learns from symptom language and still works
    when a new user pastes a post without any hashtags.
    """
    t = re.sub(r"https?://\S+", " ", str(text).lower())
    t = re.sub(r"#\w+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def clean_text(text: str) -> str:
    """
    Full cleaning used for *unsupervised* mining: lowercase -> tokenise -> stop-word removal ->
    lemmatisation. Hashtag words are kept (without '#') because they are strong topical signals.
    """
    t = re.sub(r"https?://\S+", " ", str(text).lower()).replace("#", " ")
    sw = _stopwords()
    toks = [m for m in _TOKEN_RE.findall(t) if len(m) >= 3 and m not in sw]
    return " ".join(_lemma(t) for t in toks)


def parse_hashtags(value) -> list[str]:
    """Parse '#fever #dengue' (or a list) into a normalised list of unique hashtags."""
    if isinstance(value, (list, tuple, set)):
        items = [str(v).lower() for v in value]
    elif isinstance(value, str):
        items = re.findall(r"#\w+", value.lower())
    else:
        items = []
    return list(dict.fromkeys(items))


@dataclass(frozen=True)
class SymptomSpec:
    name: str
    pattern: str
    department: str
    hashtags: tuple[str, ...] = ()


# Lexicon-based medical NER. A curated lexicon is transparent, auditable and dependency-free,
# which matters in healthcare; swap in scispaCy/MedCAT later if you need broader coverage.
SYMPTOMS: list[SymptomSpec] = [
    SymptomSpec("fever", r"\b(fever|febrile|high temperature|temperature of 10\d\w*|chills)\b", "General Medicine", ("#fever",)),
    SymptomSpec("headache", r"\bhead\s?aches?\b", "Neurology", ("#headache",)),
    SymptomSpec("migraine", r"\bmigraines?\b", "Neurology", ("#migraine", "#headache")),
    SymptomSpec("light sensitivity", r"\b(sensitiv\w*\s+to\s+(bright\s+)?light|photophobia)\b", "Neurology", ("#migraine",)),
    SymptomSpec("dizziness", r"\b(dizz\w+|light-?headed\w*|vertigo)\b", "Neurology", ("#dizziness",)),
    SymptomSpec("fainting", r"\b(faint\w*|passed out|unconscious|blackouts?|collapsed)\b", "Emergency Medicine", ()),
    SymptomSpec("nausea", r"\bnause\w+", "Gastroenterology", ("#gastro",)),
    SymptomSpec("vomiting", r"\b(vomit\w*|throwing up)\b", "Gastroenterology", ("#gastro",)),
    SymptomSpec("diarrhea", r"\b(loose motions?|diarrh\w+)\b", "Gastroenterology", ("#gastro",)),
    SymptomSpec("abdominal pain", r"\b(stomach (pain|ache)|abdominal pain|belly pain|cramps)\b", "Gastroenterology", ("#gastro", "#stomach")),
    SymptomSpec("acidity / bloating", r"\b(acidity|bloating|heartburn|acid reflux)\b", "Gastroenterology", ("#stomach",)),
    SymptomSpec("rash", r"\b(rash\w*|red (itchy )?bumps|hives|welts)\b", "Dermatology / Allergy", ("#allergy", "#skinproblem")),
    SymptomSpec("itching", r"\bitch\w+", "Dermatology / Allergy", ("#skinproblem",)),
    SymptomSpec("swelling", r"\b(swell\w*|swollen)\b", "Dermatology / Allergy", ("#allergy",)),
    SymptomSpec("joint pain", r"\b(joint pains?|arthralgia)\b", "General Medicine", ()),
    SymptomSpec("body ache", r"\b(body ache|body pain|myalgia|muscle pain)\b", "General Medicine", ()),
    SymptomSpec("eye pain (retro-orbital)", r"\bpain behind (my |the |one |both )?(eyes?|eye)\b", "Infectious Disease", ()),
    SymptomSpec("low platelets", r"\bplatelet\w*", "Infectious Disease", ("#dengue",)),
    SymptomSpec("fatigue", r"\b(fatigue|tired\w*|weakness|exhaust\w+)\b", "General Medicine", ()),
    SymptomSpec("cough", r"\bcough\w*", "Pulmonology", ("#cough",)),
    SymptomSpec("sore throat", r"\b(sore throat|throat pain)\b", "ENT", ("#cold",)),
    SymptomSpec("runny / blocked nose", r"\b(runny nose|blocked nose|sneez\w+|congest\w+)\b", "ENT", ("#cold", "#flu")),
    SymptomSpec("shortness of breath", r"\b(short(ness)? of breath|breathless\w*|difficulty breathing|struggling to breathe|can'?t breathe|cannot breathe)\b", "Pulmonology", ("#asthma",)),
    SymptomSpec("wheezing", r"\bwheez\w+", "Pulmonology", ("#asthma",)),
    SymptomSpec("chest pain", r"\bchest (pain|tightness|pressure)|crushing chest", "Cardiology", ("#hearthealth",)),
    SymptomSpec("palpitations", r"\b(palpitations?|racing heart|heart racing)\b", "Cardiology", ("#hearthealth",)),
    SymptomSpec("high blood pressure", r"\b(high (blood pressure|bp)|hypertension|blood pressure readings?)\b", "Cardiology", ("#bp", "#hearthealth")),
    SymptomSpec("high blood sugar", r"\b(blood sugar|glucose|hyperglyc\w+)\b", "Endocrinology", ("#diabetes",)),
    SymptomSpec("excessive thirst", r"\b(increased|excessive|extreme)\s+thirst\w*|\bthirst\w*\b", "Endocrinology", ("#diabetes",)),
    SymptomSpec("frequent urination", r"\bfrequent urination\b", "Endocrinology", ("#diabetes",)),
    SymptomSpec("blurred vision", r"\bblurr?(ed|y) vision\b", "Ophthalmology", ("#diabetes",)),
    SymptomSpec("tingling / numbness", r"\b(tingling|numbness)\b", "Neurology", ()),
    SymptomSpec("anxiety / panic", r"\b(anxi\w+|panic( attacks?)?|racing thoughts)\b", "Psychiatry", ("#mentalhealth", "#anxiety")),
    SymptomSpec("insomnia", r"\b(can'?t sleep|insomnia|sleepless\w*|unable to sleep|trouble sleeping|no sleep)\b", "Psychiatry", ("#mentalhealth",)),
    SymptomSpec("low mood", r"\b(hopeless\w*|feeling (very )?low|depress\w+)\b", "Psychiatry", ("#mentalhealth",)),
    SymptomSpec("bleeding", r"\b(bleed\w+|black stools?|blood in (stool|urine))\b", "General Medicine", ()),
]
_SYMPTOM_RES = [(s, re.compile(s.pattern, re.I)) for s in SYMPTOMS]

NEGATIONS = {"no", "not", "without", "never", "denies", "denied", "nor", "neither", "nothing", "none"}


def _is_negated(text: str, start: int) -> bool:
    """True if one of the 3 words before `start` is a negation ('no chest pain', 'without fever')."""
    prev = re.findall(r"[a-z']+", text[:start].lower())[-3:]
    return any(w in NEGATIONS for w in prev)


def extract_medical_entities(text: str) -> list[str]:
    """Return canonical symptom names found in `text` (negated mentions are ignored), in order of appearance."""
    found: list[tuple[int, str]] = []
    for spec, rx in _SYMPTOM_RES:
        m = next((m for m in rx.finditer(text) if not _is_negated(text, m.start())), None)
        if m:
            found.append((m.start(), spec.name))
    return [name for _, name in sorted(found)]


_SPEC_BY_NAME = {s.name: s for s in SYMPTOMS}
_DENGUE_SIGNALS = {"joint pain", "eye pain (retro-orbital)", "low platelets", "rash"}

_DUR_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(hour|hr|day|week|month|year)s?")
_DUR_UNITS = {"hour": 1 / 24, "hr": 1 / 24, "day": 1, "week": 7, "month": 30, "year": 365}
_DUR_WORDS = {"yesterday": 1.0, "this morning": 0.3, "since morning": 0.3, "last night": 0.5,
              "past week": 7.0, "last week": 7.0, "few days": 3.0, "couple of days": 2.0}


def extract_duration_days(text: str) -> float | None:
    """Longest symptom duration mentioned, in days ('for 3 days' -> 3.0, 'since yesterday' -> 1.0)."""
    t = text.lower()
    vals = [float(n) * _DUR_UNITS[u] for n, u in _DUR_RE.findall(t)]
    vals += [v for k, v in _DUR_WORDS.items() if k in t]
    return max(vals) if vals else None


# =====================================================================================
# 3. URGENCY & SENTIMENT SCORING
# =====================================================================================
# Weighted keyword lexicon. Weights are clinical-intuition priors (red-flag terms >> mild terms);
# a negative weight de-escalates. Weights are additive, each term counted once per post.
URGENCY_TERMS: list[tuple[str, str, float]] = [
    ("throbbing", r"\bthrobbing\b", 1.0),
    ("dizziness", r"\b(dizz\w+|light-?headed\w*)\b", 1.2),
    ("itching", r"\bitch\w+", 0.5),
    ("rash / hives", r"\b(hives|rash\w*)\b", 0.8),
    ("bleeding", r"\bbleed\w+", 3.0),
    ("severe", r"\bsevere(ly)?\b", 2.0),
    ("unbearable", r"\bunbearable\b", 2.5),
    ("worsening", r"\b(worsen\w*|getting worse|worse every)\b", 1.5),
    ("vomiting", r"\bvomit\w+", 1.5),
    ("high fever", r"\bhigh fever\b|temperature of 10[3-9]", 1.5),
    ("low platelets", r"\bplatelet\w*", 2.0),
    ("joint pain", r"\bjoint pains?\b", 1.0),
    ("breathing difficulty", r"\b(short(ness)? of breath|breathless\w*|wheez\w+|struggling to breathe|can'?t breathe|cannot breathe)\b", 3.0),
    ("chest pain", r"\bchest (pain|tightness|pressure)\b", 3.5),
    ("palpitations", r"\bpalpitations?\b", 1.8),
    ("fainting", r"\b(faint\w*|passed out|unconscious|collapsed)\b", 4.0),
    ("swelling", r"\b(swollen|swell\w+)\b", 1.5),
    ("no relief", r"\b(not (helping|working)|no relief|only (helps|helped))\b", 1.2),
    ("hopelessness / panic", r"\b(hopeless\w*|panic)\b", 1.8),
    ("confusion", r"\bconfus\w+", 2.0),
    ("blurred vision", r"\bblurr?(ed|y) vision\b", 1.2),
    ("mild wording", r"\b(mild|slight\w*|bearable|manageable)\b", -1.5),
]
_URGENCY_RES = [(label, re.compile(p, re.I), w) for label, p, w in URGENCY_TERMS]
_POS_WORDS = {"better", "improved", "improving", "relieved", "fine", "okay", "resolved", "manageable", "bearable"}
_NEG_WORDS = {"worried", "scared", "afraid", "anxious", "terrified", "worse", "worsening", "unbearable",
              "hopeless", "panic", "helpless", "desperate", "confused", "struggling", "worst"}


def urgency_score(text: str) -> tuple[float, list[str]]:
    """
    Continuous urgency score in [0, 100] plus the matched keywords.

    score = 100 * (1 - exp(-raw * persistence / 8))

    * raw = sum of keyword weights (negation-aware).
    * persistence = 1 + 0.12 * ln(1 + duration_days): symptoms lasting longer are more concerning.
    * The saturating exponential maps an unbounded additive score onto (0, 100) with diminishing
      returns, so one extra keyword matters more in a mild post than in an already-alarming one.
    """
    raw, hits = 0.0, []
    for label, rx, w in _URGENCY_RES:
        if any(not _is_negated(text, m.start()) for m in rx.finditer(text)):
            raw += w
            hits.append(label)
    days = extract_duration_days(text)
    persistence = 1.0 + 0.12 * math.log1p(days) if days else 1.0
    return float(100 * (1 - math.exp(-max(raw, 0.0) * persistence / 8))), hits


def sentiment_score(text: str) -> float:
    """Lexicon polarity in (-1, 1): (pos - neg) / (pos + neg + 1). The +1 smooths posts with no sentiment words to 0."""
    toks = set(_TOKEN_RE.findall(text.lower()))
    pos, neg = len(toks & _POS_WORDS), len(toks & _NEG_WORDS)
    return (pos - neg) / (pos + neg + 1)


def score_to_band(score: float) -> str:
    """Map a 0-100 urgency score to a severity band (used as a sanity check against the ML prediction)."""
    return SEVERITY_ORDER[0 if score < 25 else 1 if score < 50 else 2 if score < 75 else 3]


RED_FLAGS: dict[str, str] = {
    "Chest pain with radiation / sweating": r"crushing chest|chest (pain|pressure) (spreading|radiating)|chest pain.{0,40}(left arm|jaw|cold sweat)",
    "Severe breathing difficulty": r"can'?t breathe|cannot breathe|struggling to breathe|gasping|lips (are |look )?(turning )?blue",
    "Loss of consciousness": r"faint(ed|ing)|passed out|unconscious|unresponsive|collapsed",
    "Severe bleeding": r"vomiting blood|coughing (up )?blood|heavy bleeding|bleeding (that )?(won'?t|will not|does not) stop|black stools?|bleeding from my gums",
    "Airway swelling / anaphylaxis": r"(lips|tongue|throat|face).{0,25}swell|swell\w*.{0,25}(lips|tongue|throat)",
    "Stroke warning signs": r"slurred speech|face (is )?droop\w*|sudden (weakness|numbness) (on|in) one side",
    "Self-harm risk": r"suicid\w+|want to die|end my life|hurt(ing)? myself|kill myself",
    "Seizure": r"seizure|convuls\w+",
}
_RED_FLAG_RES = [(k, re.compile(v, re.I)) for k, v in RED_FLAGS.items()]


def detect_red_flags(text: str) -> list[str]:
    """Rule-based safety net: phrases that should always be escalated regardless of the ML model."""
    return [label for label, rx in _RED_FLAG_RES if any(not _is_negated(text, m.start()) for m in rx.finditer(text))]


def enrich_posts(df: pd.DataFrame) -> pd.DataFrame:
    """Add all derived NLP features (clean_text, hashtag_list, entities, urgency_score, sentiment, severity_num)."""
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["hashtag_list"] = out["hashtags"].map(parse_hashtags)
    out["clean_text"] = out["raw_text"].map(clean_text)
    out["entities"] = out["raw_text"].map(extract_medical_entities)
    scores = out["raw_text"].map(urgency_score)
    out["urgency_score"] = scores.map(lambda s: s[0])
    out["sentiment"] = out["raw_text"].map(sentiment_score)
    out["severity_num"] = out["triage_severity"].map(SEVERITY_TO_NUM)
    return out


# =====================================================================================
# 2. HASHTAG CO-OCCURRENCE MINING
# =====================================================================================
def hashtag_matrix(df: pd.DataFrame, min_count: int = 10) -> pd.DataFrame:
    """Binary post x hashtag indicator matrix, keeping tags used in >= `min_count` posts."""
    tags = df["hashtag_list"].explode().dropna()
    keep = tags.value_counts()
    keep = keep[keep >= min_count].index
    exploded = tags[tags.isin(keep)]
    mat = pd.crosstab(exploded.index, exploded).clip(upper=1)
    return mat.reindex(df.index, fill_value=0).astype(int)


def cooccurrence_metrics(binary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Pairwise association metrics between hashtags. Each answers a different question:

    * counts  - raw co-occurrence count (X'X).
    * jaccard - |A and B| / |A or B|. Intuitive overlap in [0,1]; ignores posts with neither tag.
    * phi     - Pearson correlation of two binary variables. Symmetric, corrects for base rates,
                can be negative (tags that avoid each other).
    * lift    - P(A,B) / (P(A) P(B)). >1 means the pair occurs more than chance: the standard
                market-basket / association-rule metric and good at surfacing rare-but-meaningful pairs.
    """
    X = binary.to_numpy(dtype=float)
    n = X.shape[0]
    counts = X.T @ X
    freq = np.diag(counts)
    union = freq[:, None] + freq[None, :] - counts
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.where(union > 0, counts / union, 0.0)
        lift = np.where(np.outer(freq, freq) > 0, n * counts / np.outer(freq, freq), 0.0)
    cols = binary.columns
    phi = binary.corr().fillna(0.0)
    mk = lambda a: pd.DataFrame(a, index=cols, columns=cols)
    res = {"counts": mk(counts), "jaccard": mk(jaccard), "phi": phi, "lift": mk(lift)}
    for k in ("jaccard", "phi", "lift"):          # diagonal (self-pairs) is uninformative
        res[k] = res[k].mask(np.eye(len(cols), dtype=bool))
    return res


def top_pairs(metrics: dict[str, pd.DataFrame], by: str = "lift", k: int = 15, min_pair_count: int = 5) -> pd.DataFrame:
    """Rank hashtag pairs by a chosen metric (upper triangle only, minimum support enforced)."""
    m, counts = metrics[by], metrics["counts"]
    rows = []
    cols = list(m.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if counts.loc[a, b] >= min_pair_count:
                rows.append({"tag_a": a, "tag_b": b, "co_count": int(counts.loc[a, b]),
                             "jaccard": metrics["jaccard"].loc[a, b], "phi": metrics["phi"].loc[a, b],
                             "lift": metrics["lift"].loc[a, b]})
    return pd.DataFrame(rows).sort_values(by, ascending=False).head(k).reset_index(drop=True)


# =====================================================================================
# 4. OUTBREAK & TREND DETECTION
# =====================================================================================
def build_daily_counts(df: pd.DataFrame, geo_col: str = "city", min_tag_posts: int = 15) -> pd.DataFrame:
    """Wide daily count matrix: index = every calendar day, columns = (region, hashtag)."""
    ex = df[["timestamp", geo_col, "hashtag_list"]].explode("hashtag_list").dropna(subset=["hashtag_list"])
    ex["date"] = ex["timestamp"].dt.normalize()
    vc = ex["hashtag_list"].value_counts()
    ex = ex[ex["hashtag_list"].isin(vc[vc >= min_tag_posts].index)]
    wide = ex.groupby(["date", geo_col, "hashtag_list"]).size().unstack([geo_col, "hashtag_list"]).fillna(0.0)
    full = pd.date_range(df["timestamp"].min().normalize(), df["timestamp"].max().normalize(), freq="D")
    return wide.reindex(full, fill_value=0.0)


def detect_outbreaks(df: pd.DataFrame, geo_col: str = "city", window: int = 7, baseline_days: int = 56,
                     z_threshold: float = 2.5, min_count: int = 6, min_tag_posts: int = 15) -> pd.DataFrame:
    """
    Flag abnormal spikes in hashtag volume per region over a rolling window.

    Method (a simplified version of the EARS/CDC aberration-detection family)
    ------------------------------------------------------------------------
    1. For each (region, hashtag) build a daily count series.
    2. Smooth with a rolling `window`-day sum  -> current activity.
    3. Baseline = rolling *median* of the previous `baseline_days` of those sums, shifted by
       `window` days so the current window never contaminates its own baseline.
    4. Spread = max(1.4826 * MAD, sqrt(median), 1). MAD is robust to earlier outbreaks polluting the
       baseline; sqrt(median) is the Poisson standard deviation, which stops tiny-count series from
       producing huge z-scores.
    5. z = (current - baseline) / spread. Alert if z >= z_threshold AND current >= min_count
       (the minimum-support rule avoids alerting on 1-vs-0 noise).

    Returns a long DataFrame: [geo_col, tag, date, count_7d, baseline, z_score, is_alert].
    """
    counts = build_daily_counts(df, geo_col, min_tag_posts)
    roll = counts.rolling(window).sum()
    prior = roll.shift(window)
    minp = max(14, baseline_days // 3)
    mu = prior.rolling(baseline_days, min_periods=minp).median()
    mad = prior.rolling(baseline_days, min_periods=minp).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
    sigma = np.maximum.reduce([1.4826 * mad.to_numpy(), np.sqrt(np.maximum(mu.to_numpy(), 0)), np.ones(mu.shape)])
    z = (roll - mu) / pd.DataFrame(sigma, index=mu.index, columns=mu.columns)
    alert = (z >= z_threshold) & (roll >= min_count)

    long = pd.DataFrame({"count_7d": roll.unstack(), "baseline": mu.unstack(),
                         "z_score": z.unstack(), "is_alert": alert.unstack()})
    long.index.names = [geo_col, "tag", "date"]
    return long.reset_index()


def active_alerts(signals: pd.DataFrame, geo_col: str = "city", lookback_days: int = 3) -> pd.DataFrame:
    """Alerts raised in the last `lookback_days` of the data, one row per (region, hashtag)."""
    last = signals["date"].max()
    recent = signals[signals["is_alert"] & (signals["date"] > last - pd.Timedelta(days=lookback_days))]
    if recent.empty:
        return pd.DataFrame(columns=[geo_col, "tag", "first_flagged", "count_7d", "baseline", "peak_z"])
    return (recent.groupby([geo_col, "tag"])
            .agg(first_flagged=("date", "min"), count_7d=("count_7d", "max"),
                 baseline=("baseline", "last"), peak_z=("z_score", "max"))
            .reset_index().sort_values("peak_z", ascending=False).reset_index(drop=True))


def latest_intensity(signals: pd.DataFrame, geo_col: str = "city") -> pd.DataFrame:
    """Region x hashtag matrix of the most recent z-score (negatives clipped to 0) for heatmaps."""
    last = signals[signals["date"] == signals["date"].max()]
    return last.pivot(index=geo_col, columns="tag", values="z_score").fillna(0.0).clip(lower=0.0)


# =====================================================================================
# 5. UNSUPERVISED PATIENT QUERY CLUSTERING
# =====================================================================================
@dataclass
class ClusterResult:
    labels: np.ndarray
    names: dict[int, str]
    top_terms: dict[int, list[str]]
    silhouette: float
    X: np.ndarray                      # L2-normalised LSA embedding (n_posts x n_components)
    k: int
    explained_variance: float


def build_embedding(texts: Sequence[str], n_components: int = 40, seed: int = 42):
    """
    TF-IDF -> Truncated SVD (LSA) -> L2 normalisation.

    * TF-IDF (1-2 grams, sublinear tf) down-weights ubiquitous words and keeps phrases like "joint pain".
    * LSA compresses the sparse matrix into a dense semantic space where distances are meaningful
      and K-Means (which assumes Euclidean geometry) behaves well.
    * Normalising rows makes Euclidean distance equivalent to cosine distance.
    """
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.7, sublinear_tf=True, max_features=4000)
    tfidf = vec.fit_transform(texts)
    n_comp = max(2, min(n_components, tfidf.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    X = Normalizer(copy=False).fit_transform(svd.fit_transform(tfidf))
    return X, vec, svd


def evaluate_k(X: np.ndarray, k_range: Sequence[int] = range(3, 11), seed: int = 42) -> pd.DataFrame:
    """Elbow (inertia) + silhouette sweep to justify the choice of k."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows)


def cluster_posts(df: pd.DataFrame, k: int = 6, n_components: int = 40, seed: int = 42) -> ClusterResult:
    """Group posts into `k` health-inquiry clusters and label each by its top centroid terms."""
    X, vec, svd = build_embedding(df["clean_text"].tolist(), n_components, seed)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    terms = np.array(vec.get_feature_names_out())
    centroids_terms = svd.inverse_transform(km.cluster_centers_)       # back-project to term space
    top = {c: [t for t in terms[np.argsort(centroids_terms[c])[::-1][:12]]] for c in range(k)}
    # prefer unigrams for readable names
    names = {c: f"C{c + 1}: " + ", ".join([t for t in top[c] if " " not in t][:3]) for c in range(k)}
    return ClusterResult(labels=km.labels_, names=names, top_terms=top,
                         silhouette=float(silhouette_score(X, km.labels_)), X=X, k=k,
                         explained_variance=float(svd.explained_variance_ratio_.sum()))


def project_2d(X: np.ndarray, method: str = "pca", seed: int = 42) -> np.ndarray:
    """
    2-D projection for visualisation.
    PCA preserves global variance (fast, deterministic); t-SNE preserves local neighbourhoods and
    separates clusters visually, but distances *between* clusters are not meaningful.
    """
    if method.lower() == "tsne":
        perplexity = float(min(30, max(5, (len(X) - 1) // 3)))
        return TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed).fit_transform(X)
    return PCA(n_components=2, random_state=seed).fit_transform(X)


# =====================================================================================
# 6. SUPERVISED TRIAGE MODEL + RED-FLAG SAFETY NET
# =====================================================================================
class TriageClassifier:
    """
    TF-IDF (1-2 grams) + two lexicon features (urgency, sentiment) -> multinomial logistic regression.

    Why logistic regression? It is fast, calibrated enough for confidence display, and its
    coefficients are inspectable, which matters when a model touches clinical triage.
    `class_weight="balanced"` stops the rare Emergency class from being ignored.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.clf = LogisticRegression(max_iter=2000, C=3.0, class_weight="balanced", random_state=seed)
        self.metrics: dict = {}

    def _features(self, texts: Sequence[str], fit: bool = False) -> sparse.csr_matrix:
        norm = [normalize_text(t) for t in texts]
        X_text = self.vec.fit_transform(norm) if fit else self.vec.transform(norm)
        num = np.array([[urgency_score(t)[0] / 100.0, sentiment_score(t)] for t in texts])
        return sparse.hstack([X_text, sparse.csr_matrix(num)]).tocsr()

    def fit(self, df: pd.DataFrame, test_size: float = 0.2) -> "TriageClassifier":
        """Hold-out evaluation first (stored in `.metrics`), then refit on all data for deployment."""
        texts, y = df["raw_text"].tolist(), df["triage_severity"].to_numpy()
        tr, te = train_test_split(np.arange(len(df)), test_size=test_size, stratify=y, random_state=self.seed)
        self.clf.fit(self._features([texts[i] for i in tr], fit=True), y[tr])
        pred = self.clf.predict(self._features([texts[i] for i in te]))
        order = {s: i for i, s in enumerate(SEVERITY_ORDER)}
        within_one = float(np.mean([abs(order[a] - order[b]) <= 1 for a, b in zip(y[te], pred)]))
        self.metrics = {
            "accuracy": float(np.mean(pred == y[te])),
            "macro_f1": float(f1_score(y[te], pred, average="macro")),
            "within_one_level": within_one,      # ordinal-aware: Moderate vs Severe is a smaller error than Mild vs Emergency
            "report": classification_report(y[te], pred, labels=SEVERITY_ORDER, output_dict=True, zero_division=0),
            "confusion": pd.DataFrame(confusion_matrix(y[te], pred, labels=SEVERITY_ORDER),
                                      index=[f"true {s}" for s in SEVERITY_ORDER],
                                      columns=[f"pred {s}" for s in SEVERITY_ORDER]),
            "n_test": int(len(te)),
        }
        self.clf.fit(self._features(texts, fit=True), y)
        return self

    def predict_proba(self, text: str) -> dict[str, float]:
        p = self.clf.predict_proba(self._features([text]))[0]
        probs = dict(zip(self.clf.classes_, p))
        return {s: float(probs.get(s, 0.0)) for s in SEVERITY_ORDER}


@dataclass
class TriageResult:
    severity: str
    ml_severity: str
    confidence: float
    probabilities: dict[str, float]
    urgency_score: float
    sentiment: float
    entities: list[str]
    department: str
    hashtags: list[str]
    red_flags: list[str]
    urgency_terms: list[str] = field(default_factory=list)
    duration_days: float | None = None
    possible_dengue: bool = False
    assessable: bool = True            # False when the text has no recognisable clinical signal
    crisis: bool = False               # True when self-harm language is detected
    notes: list[str] = field(default_factory=list)   # human-readable explanation of overrides


def recommend_department(entities: Sequence[str], severity: str) -> str:
    """Vote for a department using matched symptoms; emergencies go to the ER regardless."""
    if severity == "Emergency":
        return "Emergency Medicine (nearest ER)"
    votes = Counter(_SPEC_BY_NAME[e].department for e in entities)
    return votes.most_common(1)[0][0] if votes else "General Medicine"


def suggest_hashtags(entities: Sequence[str], text: str, max_tags: int = 5) -> tuple[list[str], bool]:
    """Suggest hashtags from entity->tag mappings; add #dengue when fever co-occurs with dengue-typical signs."""
    counter: Counter = Counter()
    for e in entities:
        counter.update(_SPEC_BY_NAME[e].hashtags)
    dengue = "fever" in entities and bool(_DENGUE_SIGNALS & set(entities))
    if dengue:
        counter["#dengue"] += 2
    tags = [t for t, _ in counter.most_common(max_tags)]
    return (tags or ["#health"]), dengue


def _non_latin_share(text: str) -> float:
    """Share of alphabetic characters outside basic Latin (detects Hindi/Bengali/etc. input)."""
    letters = [c for c in text if c.isalpha()]
    return sum(ord(c) > 0x024F for c in letters) / len(letters) if letters else 0.0


def analyze_post(text: str, model: TriageClassifier) -> TriageResult:
    """
    Full triage of one free-text post.

    Decision logic (each step can only *raise* severity, never lower it):
      1. ML prediction (TF-IDF + lexicon features -> logistic regression).
      2. Explicit-severity floor: the patient's own words "severe" / "unbearable" (not negated)
         set a floor of 'Severe'. Patients rarely exaggerate these words, and the synthetic
         training data under-represents them, so this corrects a known model weakness.
      3. Red-flag floor: life-threatening phrases force 'Emergency'. In triage a false negative
         (missing an emergency) is far costlier than a false positive.

    Inputs with no recognisable clinical signal (gibberish, unsupported scripts) are marked
    `assessable=False` instead of returning a confident-looking 'Mild'.
    """
    probs = model.predict_proba(text)
    ml_label = max(probs, key=probs.get)
    flags = detect_red_flags(text)
    score, terms = urgency_score(text)
    entities = extract_medical_entities(text)
    notes: list[str] = []

    assessable = bool(entities or flags or terms)
    if not assessable:
        notes.append("Unsupported language" if _non_latin_share(text) > 0.3 else
                     "No recognisable symptoms found; the model cannot assess this text.")

    severity = ml_label
    if {"severe", "unbearable"} & set(terms) and SEVERITY_ORDER.index(severity) < 2:
        severity = "Severe"
        notes.append("Raised to Severe: patient describes symptoms as severe/unbearable.")
    if flags:
        severity = "Emergency"
        notes.append("Raised to Emergency by red-flag rule: " + "; ".join(flags))

    tags, dengue = suggest_hashtags(entities, text)
    dept = "Infectious Disease" if dengue and severity != "Emergency" else recommend_department(entities, severity)
    return TriageResult(severity=severity, ml_severity=ml_label, confidence=probs[ml_label], probabilities=probs,
                        urgency_score=score, sentiment=sentiment_score(text), entities=entities,
                        department=dept, hashtags=tags, red_flags=flags, urgency_terms=terms,
                        duration_days=extract_duration_days(text), possible_dengue=dengue,
                        assessable=assessable, crisis="Self-harm risk" in flags, notes=notes)


if __name__ == "__main__":
    from data_generator import generate_dataset

    raw = generate_dataset()
    data = enrich_posts(raw)
    print(f"Posts: {len(data):,}")
    print("\nTop co-occurring hashtag pairs (lift):")
    print(top_pairs(cooccurrence_metrics(hashtag_matrix(data)), by="lift", k=8).round(2).to_string(index=False))
    sig = detect_outbreaks(data)
    print("\nActive outbreak alerts:")
    print(active_alerts(sig).head(8).round({"count_7d": 0, "baseline": 1, "peak_z": 1}).to_string(index=False))
    cl = cluster_posts(data, k=6)
    print(f"\nClusters (silhouette={cl.silhouette:.3f}):", *cl.names.values(), sep="\n  ")
    tm = TriageClassifier().fit(data)
    print(f"\nTriage model: accuracy={tm.metrics['accuracy']:.3f} macro-F1={tm.metrics['macro_f1']:.3f} "
          f"within-one-level={tm.metrics['within_one_level']:.3f}")
    print(analyze_post("High fever and joint pain for 2 days, also pain behind my eyes", tm))
