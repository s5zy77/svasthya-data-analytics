"""
data_generator.py
=================
Synthetic dataset generator for the Svasthya community-health analytics module.

Design goals (worth explaining in a portfolio write-up)
-------------------------------------------------------
1. **Realistic language** - posts are composed from condition-specific symptom
   phrases, duration phrases, severity-tier phrases and question closers, so the
   NLP pipeline has genuine signal to learn from.
2. **Controlled ground truth** - severity is drawn *first* and the text is
   written to match it (plus ~4 % label noise), so classifier metrics are
   meaningful but not trivially perfect.
3. **Planted outbreaks** - three geographic outbreaks are injected on top of the
   baseline traffic so the outbreak-detection algorithm has something real to
   find (Dengue/Kolkata, Flu/Delhi, Gastro/Mumbai).
4. **Reproducible** - everything flows from one seed.

Usage
-----
    python data_generator.py --n-posts 1200 --seed 42 --out data/svasthya_posts.csv
"""
from __future__ import annotations

import argparse
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

SEVERITY_LEVELS: list[str] = ["Mild", "Moderate", "Severe", "Emergency"]
USER_TYPES: list[str] = ["PATIENT", "DOCTOR_VERIFIED", "ANONYMOUS"]

# City metadata: coordinates are used by the dashboard map; weights approximate
# relative user-base size so baseline traffic is not uniform.
CITY_META: dict[str, dict] = {
    "Kolkata":   {"lat": 22.5726, "lon": 88.3639, "pincodes": ["700001", "700019", "700091"], "weight": 0.20},
    "Delhi":     {"lat": 28.6139, "lon": 77.2090, "pincodes": ["110001", "110016", "110092"], "weight": 0.22},
    "Mumbai":    {"lat": 19.0760, "lon": 72.8777, "pincodes": ["400001", "400050", "400092"], "weight": 0.20},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946, "pincodes": ["560001", "560034", "560100"], "weight": 0.16},
    "Chennai":   {"lat": 13.0827, "lon": 80.2707, "pincodes": ["600001", "600020"],           "weight": 0.10},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867, "pincodes": ["500001", "500034"],           "weight": 0.08},
    "Pune":      {"lat": 18.5204, "lon": 73.8567, "pincodes": ["411001", "411045"],           "weight": 0.04},
}


@dataclass(frozen=True)
class Condition:
    """A clinical scenario that posts are generated from."""
    key: str
    tags: tuple[str, ...]                      # always-present hashtags
    extra_tags: tuple[tuple[str, float], ...]  # (hashtag, probability) -> creates co-occurrence structure
    symptoms: tuple[str, ...]
    emergency_phrase: str
    severity_probs: tuple[float, float, float, float]  # P(Mild, Moderate, Severe, Emergency)
    weight: float                              # share of baseline traffic


CONDITIONS: list[Condition] = [
    Condition("fever", ("#fever",), (("#headache", 0.15),),
              ("high fever", "chills and shivering", "a temperature of 102F", "fever that spikes at night", "sweating and weakness"),
              "I fainted and I am very confused with this fever",
              (0.35, 0.40, 0.20, 0.05), 0.14),
    Condition("headache", ("#headache",), (("#migraine", 0.60),),
              ("severe throbbing headache on the right side", "sensitivity to light", "nausea with the headache",
               "feeling dizzy when I stand up quickly", "pain behind one eye"),
              "sudden worst headache of my life with slurred speech",
              (0.30, 0.40, 0.25, 0.05), 0.14),
    Condition("dengue", ("#dengue", "#fever"), (("#headache", 0.35),),
              ("high fever with severe joint pain", "pain behind my eyes", "red rashes on my arms and chest",
               "platelet count dropping to 90,000", "extreme fatigue and body ache"),
              "there is bleeding from my gums and black stools",
              (0.10, 0.35, 0.40, 0.15), 0.05),
    Condition("flu", ("#flu",), (("#cold", 0.70), ("#fever", 0.45), ("#cough", 0.30)),
              ("runny nose and sore throat", "dry cough and sneezing", "body ache and tiredness", "a blocked nose and mild fever"),
              "I am struggling to breathe",
              (0.55, 0.35, 0.08, 0.02), 0.10),
    Condition("allergy", ("#allergy", "#skinproblem"), (),
              ("red itchy bumps on my forearm after eating shrimp", "itching and hives all over my back",
               "a rash spreading on my neck", "swollen lips after a new medicine"),
              "my lips and throat are swelling and I cannot breathe",
              (0.40, 0.40, 0.15, 0.05), 0.12),
    Condition("diabetes", ("#diabetes",), (("#hearthealth", 0.12),),
              ("increased thirst and frequent urination", "a blood sugar reading of 280 after meals",
               "blurred vision in the evenings", "tingling in both feet"),
              "I fainted and my sugar reading is above 450",
              (0.30, 0.45, 0.20, 0.05), 0.10),
    Condition("mentalhealth", ("#mentalhealth",), (("#anxiety", 0.50),),
              ("constant anxiety and racing thoughts", "no sleep for days", "feeling hopeless and very low", "frequent panic attacks"),
              "I feel unsafe and cannot stop thinking of hurting myself",
              (0.30, 0.45, 0.20, 0.05), 0.09),
    Condition("hearthealth", ("#hearthealth",), (("#bp", 0.40), ("#diabetes", 0.10)),
              ("chest tightness while climbing stairs", "palpitations at night", "high blood pressure readings",
               "shortness of breath on mild effort"),
              "crushing chest pain spreading to my left arm with cold sweat",
              (0.15, 0.35, 0.30, 0.20), 0.07),
    Condition("gastro", ("#gastro",), (("#stomach", 0.60), ("#fever", 0.10)),
              ("stomach pain and loose motions", "repeated vomiting and nausea", "acidity and bloating after meals"),
              "I am vomiting blood",
              (0.40, 0.38, 0.17, 0.05), 0.09),
    Condition("respiratory", ("#asthma",), (("#cough", 0.60),),
              ("wheezing and shortness of breath at night", "a persistent cough for weeks", "tightness in the chest when I cough"),
              "I am struggling to breathe and my lips look blue",
              (0.25, 0.40, 0.25, 0.10), 0.10),
]
_COND_BY_KEY = {c.key: c for c in CONDITIONS}

# (condition, city, start_days_ago, peak_days_ago, n_extra_posts)
OUTBREAKS: list[dict] = [
    {"condition": "dengue", "city": "Kolkata", "start": 26, "peak": 7, "n": 150},
    {"condition": "flu",    "city": "Delhi",   "start": 20, "peak": 5, "n": 110},
    {"condition": "gastro", "city": "Mumbai",  "start": 14, "peak": 4, "n": 70},
]

OPENERS = ["", "", "Hi everyone, ", "Need advice. ", "Please help. ", "Not sure what to do. "]
CLOSERS = ["Should I be concerned?", "Has anyone experienced this?", "Should I see a doctor?",
           "What could this be?", "Any home remedies that actually work?", "Is it safe to wait it out?"]
DURATIONS = ["for 3 days", "for 2 days", "since yesterday", "for the past week", "for 2 weeks",
             "since this morning", "for 5 hours", "for about 4 days", "for a month"]
SEVERITY_PHRASES: dict[str, list[str]] = {
    "Mild": ["It is mild and manageable.", "Just a slight discomfort.", "Not too bad but lingering.", "Mostly bearable during the day."],
    "Moderate": ["It is affecting my sleep.", "Painkillers give only partial relief.", "It keeps coming and going.",
                 "Hard to focus at work because of it."],
    "Severe": ["It is unbearable and getting worse.", "Medicines are not helping at all.",
               "I am unable to eat or drink properly.", "The symptoms are worsening every day."],
}
DOCTOR_TEMPLATES = [
    "Clinical note: seeing a rise in {sym} cases at my OPD this week. Stay hydrated, rest, and seek care early.",
    "Reminder from a physician: persistent {sym} for more than a few days deserves a proper check-up.",
    "Doctor's advice: if {sym} gets worse or does not settle, please visit a clinic.",
]


def _pick(rng: np.random.Generator, seq: Sequence):
    """Pick one element from a sequence (returns native Python type, not numpy scalar)."""
    return seq[int(rng.integers(len(seq)))]


def _compose_post(rng: np.random.Generator, cond: Condition, severity: str, user_type: str) -> tuple[str, list[str]]:
    """Build (text, hashtags) for one post."""
    tags = list(cond.tags) + [t for t, p in cond.extra_tags if rng.random() < p]
    tag_str = " ".join(tags)
    k = int(rng.integers(2, 4))
    syms = [str(s) for s in rng.choice(list(cond.symptoms), size=min(k, len(cond.symptoms)), replace=False)]

    if user_type == "DOCTOR_VERIFIED":
        text = _pick(rng, DOCTOR_TEMPLATES).format(sym=" and ".join(syms[:2]))
        return f"{text} {tag_str}", tags

    sym_text = ", ".join(syms[:-1]) + (" and " if len(syms) > 1 else "") + syms[-1]
    lead = _pick(rng, ["I have", "Having", "Experiencing", "Been dealing with"])
    parts = [f"{_pick(rng, OPENERS)}{lead} {sym_text} {_pick(rng, DURATIONS)}."]
    if severity == "Emergency":
        parts.append(f"Right now {cond.emergency_phrase}.")
    else:
        parts.append(_pick(rng, SEVERITY_PHRASES[severity]))
    parts.append(_pick(rng, CLOSERS))
    parts.append(tag_str)
    return " ".join(parts), tags


def _sample_hour(rng: np.random.Generator) -> int:
    """Posting hour: bimodal (late morning + late evening) like real health forums."""
    hours = np.arange(24)
    p = 0.3 + np.exp(-0.5 * ((hours - 10) / 2.5) ** 2) + 1.2 * np.exp(-0.5 * ((hours - 21) / 2.5) ** 2)
    return int(rng.choice(hours, p=p / p.sum()))


def generate_dataset(n_posts: int = 1200, seed: int = 42, end_date: datetime | None = None,
                     include_outbreaks: bool = True) -> pd.DataFrame:
    """
    Generate the synthetic Svasthya post dataset.

    Parameters
    ----------
    n_posts : baseline posts (outbreak posts are added on top, so total > n_posts).
    seed : RNG seed for reproducibility.
    end_date : last day of the 6-month window (default: now).
    include_outbreaks : inject the planted regional outbreaks.

    Returns
    -------
    DataFrame with columns: post_id, timestamp, user_type, raw_text, hashtags, likes_count,
    comments_count, doctor_replied, location_pincode, city, triage_severity.
    """
    rng = np.random.default_rng(seed)
    end = end_date or datetime.now()
    end_day = pd.Timestamp(end).normalize()
    cities = list(CITY_META)
    city_p = np.array([CITY_META[c]["weight"] for c in cities])
    cond_p = np.array([c.weight for c in CONDITIONS])

    # Plan every post as (condition, city, timestamp) first ...............................
    plan: list[tuple[Condition, str, pd.Timestamp]] = []

    def _stamp(days_ago: float) -> pd.Timestamp:
        ts = end_day - pd.Timedelta(days=int(days_ago)) + pd.Timedelta(hours=_sample_hour(rng), minutes=int(rng.integers(60)))
        return min(ts, pd.Timestamp(end))

    for _ in range(n_posts):
        cond = CONDITIONS[int(rng.choice(len(CONDITIONS), p=cond_p / cond_p.sum()))]
        city = str(rng.choice(cities, p=city_p / city_p.sum()))
        plan.append((cond, city, _stamp(rng.uniform(0, 180))))

    if include_outbreaks:
        for ob in OUTBREAKS:
            # Triangular distribution in "days ago": ramps up, peaks, then decays toward today.
            for d in rng.triangular(0, ob["peak"], ob["start"], size=ob["n"]):
                plan.append((_COND_BY_KEY[ob["condition"]], ob["city"], _stamp(d)))

    # ... then realise each planned post ...................................................
    rows = []
    user_p = [0.70, 0.08, 0.22]
    for cond, city, ts in plan:
        user_type = str(rng.choice(USER_TYPES, p=user_p))
        probs = np.array(cond.severity_probs)
        if user_type == "DOCTOR_VERIFIED":      # doctors write advisories, not emergencies
            probs = probs * np.array([1, 1, 0, 0])
        sev_idx = int(rng.choice(4, p=probs / probs.sum()))
        text, tags = _compose_post(rng, cond, SEVERITY_LEVELS[sev_idx], user_type)

        label_idx = sev_idx
        if rng.random() < 0.04:                 # label noise: human triagers disagree
            label_idx = int(np.clip(sev_idx + rng.choice([-1, 1]), 0, 3))

        # Engagement: negative-binomial likes (over-dispersed), Poisson comments.
        lam_likes = (4 + 5 * sev_idx) * (2 if user_type == "DOCTOR_VERIFIED" else 1)
        likes = int(rng.negative_binomial(2, 2 / (2 + lam_likes)))
        replied = user_type != "DOCTOR_VERIFIED" and rng.random() < [0.35, 0.50, 0.70, 0.85][sev_idx]
        comments = int(rng.poisson(1 + 0.8 * sev_idx))
        if replied:
            comments = max(comments, 1)

        rows.append({
            "post_id": str(uuid.UUID(bytes=rng.bytes(16), version=4)),
            "timestamp": ts,
            "user_type": user_type,
            "raw_text": text,
            "hashtags": " ".join(tags),
            "likes_count": likes,
            "comments_count": comments,
            "doctor_replied": bool(replied),
            "location_pincode": _pick(rng, CITY_META[city]["pincodes"]),
            "city": city,
            "triage_severity": SEVERITY_LEVELS[label_idx],
        })

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-posts", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data/svasthya_posts.csv"))
    ap.add_argument("--end-date", type=str, default=None,
                    help="ISO end of the 6-month window (e.g. 2026-10-27T12:00); default = now. Fix it for reproducible data.")
    args = ap.parse_args()
    end = datetime.fromisoformat(args.end_date) if args.end_date else None
    data = generate_dataset(args.n_posts, args.seed, end_date=end)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out, index=False)
    print(f"Wrote {len(data):,} posts -> {args.out}")
    print(data.triage_severity.value_counts().to_string())
