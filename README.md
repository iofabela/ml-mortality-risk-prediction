# ml-mortality-risk-prediction

Machine-learning model that predicts the **risk of in-hospital death** for critically ill adult patients, using clinical data collected around **day 3 of hospitalization** (SUPPORT2 dataset).

---

## 1. Problem description

### What we predict
A binary outcome, **`hospdead`**: did the patient die during the hospital stay?

| Value | Meaning | Count | Share |
|---|---|---|---|
| `0` | Discharged alive | 6,745 | 74.1% |
| `1` | Died in hospital | 2,360 | 25.9% |

The model outputs a **probability of death** (0–1), not just a yes/no label, so the score can be used to rank patients by risk.

### Who benefits and how the model is used
- **ICU / ward clinicians**: an early, objective risk score that helps flag patients who are deteriorating and need closer monitoring or an escalation of care.
- **Palliative-care teams**: a prompt to start goals-of-care and advance-directive conversations early, rather than in the last hours.
- **Hospital managers**: help with planning beds, staff and ICU capacity.

Intended use: the model is a **decision-support tool**. A clinician sends a patient's day-3 variables to a prediction service and receives a risk probability. The model does **not** replace clinical judgement.

### Why this problem matters
Doctors often misjudge how likely a seriously ill patient is to survive. The original SUPPORT study was designed to improve end-of-life care because doctors' estimates and patients' wishes often did not match the care the patients received. An accurate, calibrated mortality score helps clinicians and families make better-informed decisions, put scarce ICU resources where they are needed, and avoid aggressive treatment that the patient does not want.

### Evaluation metric
The target is **imbalanced** (~26% positive), so accuracy is misleading: a model that always predicts "survives" is 74% accurate and useless.

- **Primary metric: PR-AUC (Average Precision).** This measures how well the model ranks the minority class (deaths).
- **Secondary metrics:**
  - **Recall** for deaths at the chosen threshold, because missing a high-risk patient (a false negative) is the costlier error.
  - **Precision** at that threshold, to keep the false-alarm rate acceptable.
  - **ROC-AUC**, which makes results easy to compare with the medical literature.
  - **Brier score / calibration curve**, because the output is used as a probability.

---

## 2. Dataset

**SUPPORT2**: *Study to Understand Prognoses and Preferences for Outcomes and Risks of Treatments*. It covers 9,105 critically ill adults from 5 US medical centers (1989–1991 and 1992–1994).

- UCI Machine Learning Repository: <https://archive.ics.uci.edu/dataset/880/support2>
- Kaggle (CSV): <https://www.kaggle.com/datasets/joebeachcapital/support2>

The dataset is committed at `data/support2.csv` (~3 MB, 9,105 rows × 47 variables plus a row index).

### Download (if `data/support2.csv` is missing)
```bash
# Option A: Kaggle CLI (needs ~/.kaggle/kaggle.json)
kaggle datasets download -d joebeachcapital/support2 -p data/ --unzip

# Option B: UCI via Python
pip install ucimlrepo
python -c "from ucimlrepo import fetch_ucirepo; d=fetch_ucirepo(id=880); \
d.data.original.to_csv('data/support2.csv', index=False)"
```

### Feature groups
| Group | Columns |
|---|---|
| Demographics | `age`, `sex`, `race`, `edu`, `income` |
| Diagnosis / comorbidity | `dzgroup`, `dzclass`, `num.co`, `diabetes`, `dementia`, `ca` |
| Day-3 physiology | `meanbp`, `wblc`, `hrt`, `resp`, `temp`, `pafi`, `alb`, `bili`, `crea`, `sod`, `ph`, `glucose`, `bun`, `urine`, `scoma` |
| Functional status | `adlp`, `adls`, `adlsc` |
| Care context | `hday`, `avtisst`, `dnr`, `dnrday` |

### Missing data (handled in preprocessing)
| Column | Missing | Column | Missing |
|---|---|---|---|
| `adlp` | 62% | `adls` | 32% |
| `urine` | 53% | `bili` | 29% |
| `glucose` | 49% | `pafi` | 26% |
| `bun` | 48% | `ph` | 25% |
| `alb` | 37% | `edu` | 18% |
| `income` | 33% | `wblc`, `crea`, `race`, `dnr` | < 3% |

---

## 3. Project structure
```
.
├── data/
│   └── support2.csv
├── notebooks/         # EDA and model experiments
├── src/               # train.py, predict.py
└── README.md
```

## 4. How to run
> Work in progress. The steps below will be completed as the project develops.

```bash
git clone https://github.com/iofabela/ml-mortality-risk-prediction.git
cd ml-mortality-risk-prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

jupyter notebook notebooks/      # EDA + experiments
python src/train.py              # train and save the final model
python src/predict.py            # serve predictions
```
