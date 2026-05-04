# Fair-Ranking-Model-for-Skill-Based-Job-Matching


A final-year project implementing a fairness-aware job-resume matching system that ranks candidates based on semantic relevance while improving representation across experience groups.

The system combines Sentence-BERT embeddings, multi-factor scoring, and fairness-aware post-processing using group normalization, UCB exploration, and MMR re-ranking.

---

## Project Overview

Traditional job-resume matching systems often rank candidates mainly using relevance signals such as semantic similarity, skill overlap, experience, or profile strength. While these methods improve matching efficiency, they may also reinforce visibility bias by repeatedly placing already advantaged candidates at the top.

This project proposes a fairness-aware matching pipeline that:

- Maintains relevance using semantic similarity and skill matching
- Balances representation across experience groups
- Uses Upper Confidence Bound (UCB) for exploration
- Uses Maximal Marginal Relevance (MMR) for diversity-aware re-ranking
- Provides CLI, REST API, and web dashboard interfaces

---

## Key Features

- Semantic job-resume matching using Sentence-BERT
- Multi-factor relevance scoring
- Fairness-aware post-processing
- Experience-group based exposure balancing
- Baseline vs fair ranking comparison
- Evaluation using NDCG@10, Precision@10, fresher share, and exposure disparity
- FastAPI backend with dashboard support
- CSV, JSON, and PNG output generation
- Smoke test for validation

---

## System Architecture

```text
User Interface Layer
(CLI, Web Dashboard, REST API)
        ↓
FastAPI Backend (app.py)
        ↓
Pipeline Runner
(backend/pipeline_runner.py)
        ↓
Core Pipeline
(src/data, src/nlp, src/features, src/ranking, src/evaluation)
        ↓
Persistent Outputs
(CSV, JSON, PNG files)
```

---

## Pipeline Stages

The system follows an eight-stage execution pipeline:

1. Load data
2. Encode embeddings
3. Build ranking features
4. Generate baseline ranking
5. Apply fairness-aware re-ranking
6. Compute evaluation metrics
7. Generate visual plots
8. Save final outputs

---

## Feature Engineering

The baseline relevance score is calculated using five features:

| Feature | Weight | Description |
|---|---:|---|
| Semantic Similarity | 0.45 | Cosine similarity between job and resume embeddings |
| Skill Coverage | 0.25 | Fraction of required skills matched |
| Skill Rarity | 0.12 | Importance of less common skills |
| Experience Score | 0.10 | Normalized years of experience |
| Skill Depth | 0.08 | Breadth and sophistication of skills |

Baseline scoring formula:

```text
score = 0.45 * semantic_similarity
      + 0.25 * skill_coverage
      + 0.12 * skill_rarity
      + 0.10 * experience_score
      + 0.08 * skill_depth
```

---

## Fairness-Aware Re-Ranking

The fairness module applies three post-processing stages:

### 1. Group Normalization

Scores are normalized within each experience group:

```text
normalized_score = (score - group_mean) / group_std
```

This reduces score advantages caused by profile length or experience group imbalance.

### 2. UCB Exploration

The system applies an exploration bonus to improve visibility for under-exposed groups:

```text
ucb_score = alpha * normalized_score
          + (1 - alpha) * c * sqrt(log(N) / (n_g + epsilon))
```

### 3. MMR Re-Ranking

MMR is used to balance relevance and diversity:

```text
mmr_score = lambda * relevance
          - (1 - lambda) * similarity_to_selected
```

---

## Configuration

Key parameters are stored in `config.py`.

```python
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

FAIRNESS_ALPHA = 0.75
UCB_C = 1.0
MMR_LAMBDA = 0.78
TOP_K = 10

FEATURE_WEIGHTS = {
    "semantic_similarity": 0.45,
    "skill_coverage": 0.25,
    "skill_rarity": 0.12,
    "experience_score": 0.10,
    "skill_depth": 0.08,
}
```

Experience groups:

```text
Fresher: <= 1.5 years
Early-career: <= 3 years
Mid-level: <= 6 years
Senior: > 6 years
```

---

## Dataset Format

### Freelancer Data

Expected fields:

```text
id
name
title
bio
skills
experience_text
```

### Job Data

Expected fields:

```text
job_id
company
title
description
required_skills
```

The system supports fallback loading from refined JSON, CSV files, or synthetic data generation.

---

## Installation

```bash
pip install -r requirements.txt
```

Main dependencies:

```text
pandas
numpy
scipy
sentence-transformers
scikit-learn
matplotlib
seaborn
fastapi
uvicorn
python-multipart
```

---

## Running the Project

### CLI Mode

Run the full pipeline:

```bash
python main.py
```

Run baseline ranking only:

```bash
python main.py --baseline
```

Run fairness-aware ranking:

```bash
python main.py --fairness
```

Run evaluation:

```bash
python main.py --eval
```

Run setup validation:

```bash
python smoke_test.py
```

---

## Running the Web Dashboard

Start the FastAPI backend:

```bash
uvicorn app:app --reload --port 8001
```

Open the dashboard:

```text
http://127.0.0.1:8001
```

For external access:

```bash
uvicorn app:app --host 0.0.0.0 --port 8001
```

---

## REST API Endpoints

### Health and Status

```text
GET /health
GET /status
```

### Run Pipeline

```text
POST /run/baseline
POST /run/fairness
POST /run/eval
POST /run/full
```

### Results

```text
GET /results/evaluation
GET /results/bias
GET /results/rankings/baseline
GET /results/rankings/fair
```

### Plots

```text
GET /plots/fairness-vs-accuracy
GET /plots/fresher-distribution
GET /plots/score-distribution
```

---

## Output Files

Generated files are saved in the `outputs/` directory.

```text
outputs/
├── baseline_ranking.csv
├── fair_ranking.csv
├── evaluation.json
├── bias.json
└── plots/
    ├── fairness_vs_accuracy.png
    ├── fresher_distribution.png
    └── score_distribution.png
```

### `baseline_ranking.csv`

Contains the baseline relevance-only ranking.

```text
job_id, freelancer_id, freelancer_name, experience_group, score, rank
```

### `fair_ranking.csv`

Contains the fairness-adjusted ranking after group normalization, UCB exploration, and MMR re-ranking.

### `evaluation.json`

Stores ranking quality metrics.

```json
{
  "baseline": {
    "ndcg_at_10": 0.7842,
    "precision_at_10": 0.8500
  },
  "fair": {
    "ndcg_at_10": 0.7231,
    "precision_at_10": 0.8000
  }
}
```

### `bias.json`

Stores fairness and exposure metrics.

```json
{
  "baseline": {
    "fresher_share_pct": 10.0,
    "exposure_disparity": 4.2
  },
  "fair": {
    "fresher_share_pct": 28.0,
    "exposure_disparity": 1.8
  }
}
```

---

## Evaluation Metrics

### Ranking Quality

- **NDCG@10**: Measures ranking order quality
- **Precision@10**: Measures proportion of relevant candidates in the top 10

### Fairness

- **Fresher Share %**: Measures fresher representation in top-K rankings
- **Exposure Disparity**: Measures imbalance across experience groups
- **Group Distribution**: Shows representation of fresher, early-career, mid-level, and senior candidates

---

## Typical Results

| Metric | Baseline | Fair Model | Change |
|---|---:|---:|---:|
| NDCG@10 | 0.78 | 0.72 | -7.7% |
| Precision@10 | 0.85 | 0.80 | -5.9% |
| Fresher Share | 10% | 28% | +180% |
| Exposure Disparity | 4.2 | 1.8 | -57% |

The results show a small relevance trade-off for a significant improvement in fairness and representation.

---

## Dashboard

The dashboard provides four main tabs:

### Home

- Run Baseline
- Run Fairness
- Run Evaluation
- Run Full Pipeline
- Real-time progress bar
- Current pipeline stage

### Metrics

- NDCG@10
- Precision@10
- Fresher share
- Exposure disparity
- Group distribution

### Rankings

- Baseline top-20 candidates
- Fair top-20 candidates
- Side-by-side ranking comparison

### Plots

- Fairness vs accuracy plot
- Fresher distribution plot
- Score distribution plot

---

## Testing

Run the smoke test:

```bash
python smoke_test.py
```

The smoke test validates:

- Required files and folders
- Package imports
- FastAPI app import
- Endpoint registration
- Data loading
- Feature computation
- Ranking generation
- Output writing

---

## Demo Steps

```bash
python smoke_test.py
uvicorn app:app --reload --port 8001
```

Then open:

```text
http://127.0.0.1:8001
```

Click **Run Full**, wait for completion, and view:

- Metrics
- Rankings
- Plots

---

## Project Structure

```text
.
├── app.py
├── main.py
├── smoke_test.py
├── config.py
├── requirements.txt
├── backend/
│   ├── pipeline_runner.py
│   └── state.py
├── src/
│   ├── data/
│   ├── nlp/
│   ├── features/
│   ├── ranking/
│   └── evaluation/
├── outputs/
│   ├── baseline_ranking.csv
│   ├── fair_ranking.csv
│   ├── evaluation.json
│   ├── bias.json
│   └── plots/
└── static/
    ├── index.html
    ├── app.js
    └── style.css
```

---

## Conclusion

This project demonstrates that job-resume matching systems can be improved using fairness-aware ranking without completely sacrificing relevance. By combining semantic embeddings, multi-factor scoring, and post-processing fairness techniques, the system produces rankings that are more explainable, inclusive, and suitable for responsible AI-driven recruitment workflows.
```
