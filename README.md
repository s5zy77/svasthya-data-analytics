# 🩺 Svasthya: Community Health Analytics

Data analytics, NLP and data-mining module for **Svasthya**, an anonymous patient community and doctor-triage platform.
It analyses community health posts to surface **outbreak signals**, **symptom patterns** and **triage urgency**, and exposes everything in an interactive Streamlit dashboard.

> ⚠️ **Educational project on 100% synthetic data. Not a medical device and not medical advice.**

| Overview & Trends | Symptom Mining | Triage Simulator |
|---|---|---|
| ![overview](docs/01_overview.png) | ![mining](docs/02_mining.png) | ![triage](docs/03_triage.png) |

## Features

| Area | What it does |
|---|---|
| **Synthetic data** (`data_generator.py`) | 1,500+ posts over 6 months across 7 Indian cities, with planted outbreaks (Dengue/Kolkata, Flu/Delhi, Gastro/Mumbai), realistic engagement and ~4% label noise. Fully seeded. |
| **NLP preprocessing** | Lowercasing, stop-word removal, WordNet lemmatisation, lexicon-based medical entity extraction with **negation handling** ("no fever" is not fever). |
| **Co-occurrence mining** | Hashtag association via counts, Jaccard, phi correlation and lift. |
| **Urgency scoring** | Continuous 0-100 score from a weighted keyword lexicon, saturating curve, duration-aware, plus lexicon sentiment. |
| **Outbreak detection** | Rolling 7-day counts vs. an 8-week rolling-median baseline, robust z-score (MAD, Poisson floor, minimum support). Simplified EARS/CDC-style aberration detection. |
| **Clustering** | TF-IDF (1-2 grams) → LSA (SVD) → K-Means, with elbow and silhouette sweep; PCA and t-SNE projections. |
| **Triage model** | TF-IDF + lexicon features → logistic regression, plus a rule-based safety net (can only *raise* severity). |
| **Dashboard** | Three-page light-theme Streamlit app with Plotly charts. |

## Quick start

```bash
git clone <your-repo-url> && cd svasthya
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run dashboard.py
```

The repo ships with `data/svasthya_posts.csv`. To regenerate data (optional):

```bash
python data_generator.py --n-posts 1200 --seed 42 --end-date 2026-10-27T12:00
```

Print the pipeline results in the terminal: `python nlp_analytics.py`.
Run the tests: `pip install -r requirements-dev.txt && pytest -q`.
Docker: `docker build -t svasthya . && docker run -p 8501:8501 svasthya`.

## Project structure

```
data_generator.py   synthetic dataset (schema + planted outbreaks)
nlp_analytics.py    preprocessing, mining, outbreak detection, clustering, triage model
dashboard.py        Streamlit app (3 pages)
tests/              pytest unit + dashboard smoke tests
data/               pre-generated CSV
docs/               screenshots
```

## Methods and design choices

* **Why a lexicon for medical NER?** It is transparent, auditable and dependency-free, which matters in healthcare. Swap in scispaCy or MedCAT for wider coverage.
* **Why robust z-scores for outbreaks?** Earlier outbreaks pollute a mean/std baseline. The median/MAD baseline resists that, the Poisson floor √median stops tiny counts producing huge z-scores, and a minimum-support rule avoids 1-vs-0 alerts. The baseline is shifted by the window length so the current window never contaminates it.
* **Why four co-occurrence metrics?** Jaccard is intuitive overlap, phi corrects for popularity and can be negative, lift surfaces rare but meaningful pairs, and counts give raw support.
* **Why LSA before K-Means?** K-Means assumes Euclidean geometry. SVD gives a dense semantic space, and L2-normalisation makes Euclidean distance equivalent to cosine distance.
* **Why logistic regression?** Fast, inspectable coefficients, `class_weight="balanced"` for the rare Emergency class. Ordinal-aware "within one level" accuracy is reported alongside accuracy and macro-F1.
* **Safety net:** an explicit "severe/unbearable" floor and red-flag phrases (chest pain with radiation, breathing difficulty, loss of consciousness, severe bleeding, anaphylaxis, stroke signs, self-harm, seizure) can only raise severity, because a missed emergency costs far more than a false alarm. Self-harm language additionally shows a crisis-helpline message (Tele-MANAS 14416, India).

## Limitations (please read)

* All data is synthetic and template-generated. Reported metrics (~94% accuracy) are **optimistic** and do not predict real-world performance.
* English only. Non-English or unrecognised text is reported as "cannot assess" rather than guessed.
* The lexicon covers ~40 symptoms; slang, misspellings and multi-symptom nuance are out of scope.
* Outbreak detection runs on simulated traffic; real surveillance needs de-duplication, population normalisation and epidemiological validation.
* Never use this for real clinical decisions.

## Roadmap

Transformer embeddings (e.g. sentence-transformers) for clustering, multilingual support (Hindi/Bengali), calibrated probabilities, pincode-level geospatial detection, and evaluation on a clinician-labelled dataset.

## License

MIT
