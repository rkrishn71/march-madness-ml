# NCAA March Madness Prediction

🥈 **Silver Medal — Top 5%** | [March Machine Learning Mania (Kaggle)](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)

Predicts win probabilities for every possible NCAA tournament matchup (men's + women's). Evaluated on Brier score — a proper scoring rule that rewards well-calibrated probabilities, not just picking the right winner.

---

## Methodology

### Feature Engineering — 22 Features Across 8 Signal Groups

All features are computed per team-season, then expressed as **Team A − Team B differences** so the model sees a symmetric, sign-consistent input regardless of which team is listed first.

| # | Group | Features | Key Design Choices |
|---|---|---|---|
| 1 | **Elo Ratings** | Prior-season Elo, current-season Elo | Inter-season mean reversion (75% carry-over); home-court adjustment (+65 pts) |
| 2 | **SOS-Adjusted Margins** | Full-season adj. margin, last-10-game momentum | Opponent Elo used to adjust raw margin: `adj = margin + (opp_elo − 1500) / 10` |
| 3 | **Win Rates & SOS** | Overall, home, away, neutral win%; strength of schedule | SOS = average opponent win% (KenPom-style) |
| 4 | **Efficiency** | Offensive rating, defensive rating, net rating, margin std dev, clutch rate | Clutch rate = win% in games decided by ≤5 pts |
| 5 | **Late-Season Form** | Win% in early/mid/late thirds, trajectory | Trajectory = late win% − early win% |
| 6 | **Box-Score Stats** | Offensive rebound rate, defensive rebound rate, FT rate, FT% | From detailed game logs: OR rate = team OR / (team OR + opp DR) |
| 7 | **Conference Strength** | Historical avg tournament seed, avg tourney team count | Exponential decay (0.7^years_ago) weights recent seasons more heavily |
| 8 | **Massey Rankings** | Consensus avg ordinal rank across all systems | Uses final pre-tournament ranking day; falls back to seed-derived rank for women's teams |

### Model Architecture

```
22 difference features
        │
        ├── StandardScaler → LogisticRegression (C=1.0)   # linear signals
        │
        └── XGBClassifier (depth=2, lr=0.02, n_estimators=10)  # nonlinear interactions
                    │
            Soft-voting ensemble (equal weights)
                    │
            CalibratedClassifierCV — Platt sigmoid, 5-fold CV
```

- **Logistic regression** captures the dominant linear signals (Elo gap, seed difference).
- **Shallow XGBoost** captures nonlinear interactions (e.g. high-offense team vs. weak defensive opponent).
- **Platt scaling** via `CalibratedClassifierCV` corrects probability overconfidence, directly optimising the Brier score metric.

### Evaluation

| Metric | Value |
|---|---|
| CV Brier Score (5-fold) | Reported in notebook output |
| 2025 Holdout Brier Score | Reported in notebook output |
| Competition result | 🥈 Silver Medal — Top 5% |

The notebook includes a **strict holdout backtest**: the model is re-trained on all seasons before 2025 and evaluated on 2025 tournament games only, simulating real competition conditions.

---

## Repository Structure

```
├── march_madness_prediction.ipynb   # Full pipeline: features → model → submission
├── README.md
└── (data files not included — see below)
```

## Data

Download the competition data from [Kaggle](https://www.kaggle.com/competitions/march-machine-learning-mania-2025/data) and place the CSV files in the same directory as the notebook (or update `DATA_DIR`).

Key files used:
- `MTeams.csv`, `WTeams.csv`
- `MRegularSeasonCompactResults.csv`, `WRegularSeasonCompactResults.csv`
- `MRegularSeasonDetailedResults.csv`, `WRegularSeasonDetailedResults.csv`
- `MNCAATourneyCompactResults.csv`, `WNCAATourneyCompactResults.csv`
- `MNCAATourneySeeds.csv`, `WNCAATourneySeeds.csv`
- `MTeamConferences.csv`, `WTeamConferences.csv`
- `MMasseyOrdinals.csv`
- `SampleSubmissionStage1.csv`, `SampleSubmissionStage2.csv`
- `MNCAATourneySeedRoundSlots.csv`

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost matplotlib
```

Tested on Python 3.10+.

---

## Notebook Walkthrough

| Section | Description |
|---|---|
| §2 Data Loading | Load and validate all CSVs |
| §3 Feature Engineering | Compute all 8 feature groups |
| §4 Feature Vector | Assemble 22-element matchup difference vectors |
| §5 Model Training | Train calibrated LR + XGBoost ensemble |
| §6 Run Pipeline | Execute end-to-end |
| §7 Feature Importance | XGBoost importances + LR coefficients (plotted) |
| §8 2025 Backtest | Holdout evaluation on 2025 tournament |
| §9 Submission | Generate Stage 2 submission CSV |
| §10 Prediction Distribution | Histogram of output probabilities |
| §11 Tournament Analysis | Most confident and closest 2026 matchup predictions |
