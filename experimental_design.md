# PhysioNet 2019 — Final Pipeline Design
### Comparing class imbalance techniques · One Colab notebook

---

## What we are doing

Train the same Random Forest eleven times. Each time, handle the class imbalance a different way. Compare the results, grouped into families.

```
Load → Build features → Split by patient → Run 11 experiments → Compare families
```

That is the whole project. One notebook, about 9 cells.

**The question we are answering:**

> Which family of class imbalance techniques works best for early sepsis detection — and does the choice of family matter more than the choice of method inside a family?

---

## The four families

This is the structure your mentor asked for, and it is the reason the project says something general instead of just ranking eleven names.

| Family | What it does | Risk |
|---|---|---|
| **Baseline** | Nothing, or changes the model instead of the data | — |
| **Oversampling** | Adds more of the rare class | Invents data that may not be realistic |
| **Undersampling** | Removes some of the common class | Throws away real information |
| **Hybrid** | Adds rare, then cleans up | Can inherit both problems |

Grouping matters because each family fails for a *different reason*. When you group them, a pattern in your numbers has an explanation attached. Without grouping, you just have a leaderboard.

---

## The eleven experiments

```python
EXPERIMENTS = {
  # ---- Baselines (not a sampling family) ----
  'B1' : ('No treatment',          None,                  None),
  'B2' : ('Balanced RF',           None,                  'balanced'),

  # ---- Oversampling ----
  'O1' : ('Random Oversampling',   RandomOverSampler(),   None),
  'O2' : ('SMOTE',                 SMOTE(),               None),
  'O3' : ('Borderline-SMOTE',      BorderlineSMOTE(),     None),
  'O4' : ('ADASYN',                ADASYN(),              None),

  # ---- Undersampling ----
  'U1' : ('Random Undersampling',  RandomUnderSampler(),  None),
  'U2' : ('Tomek Links',           TomekLinks(),          None),
  'U3' : ('Edited Nearest Neigh.', EditedNearestNeighbours(), None),

  # ---- Hybrid ----
  'H1' : ('SMOTEENN',              SMOTEENN(),            None),
  'H2' : ('SMOTETomek',            SMOTETomek(),          None),
}
# (display name, sampler, class_weight)
```

### Why U2 and U3 were added

Your original plan had **four** oversampling methods and only **one** undersampling method. You cannot fairly compare families like that — if oversampling wins, you will not know whether it is genuinely better or whether you simply gave it four tries and undersampling one.

Tomek Links and ENN are also not random picks. SMOTEENN is literally SMOTE + ENN. SMOTETomek is SMOTE + Tomek. So now you have the combinations *and* their separate parts, which lets you answer a real question:

> Does combining oversampling with cleaning actually beat either piece on its own?

Two extra experiments, and your hybrid row goes from "two more methods" to a proper analysis.

### Two kinds of undersampling — say this in your report

The undersampling family is not one thing:

- **U1 (balancing)** deletes a huge amount of data to force a 50/50 split. On our data it keeps about 13,000 rows out of 380,000.
- **U2, U3 (cleaning)** only remove common-class rows sitting in confusing places. They might delete a few percent. The data stays almost as imbalanced as it started.

These are very different actions. Expect their results to look nothing alike, and do not treat them as the same kind of intervention.

### Why B2 is not in a family

B2 is a Random Forest with `class_weight='balanced'`. It does not touch the data at all — it changes how the model is *punished* for mistakes. That makes it an **algorithm-level** fix, not a sampling one.

Keeping it means your report compares data-level fixes against an algorithm-level fix. It costs one line and it is often the winner, so if you leave it out and conclude "SMOTE helps," the first viva question will be "compared to what?"

---

## The three rules you cannot break

Everything else here is a suggestion. These three are not.

**Rule 1 — Split by patient, never by row.**
Each patient has about 38 hourly rows, and hour 14 looks almost identical to hour 15. Put one in training and one in testing and the model has effectively seen the answer. Scores go up; they mean nothing. Use `StratifiedGroupKFold` with `patient_id` as the group.

**Rule 2 — Resample inside the fold, never before splitting.**
This is your supervisor's point. The good news: doing it correctly is *fewer* lines than doing it wrong, because `imblearn`'s Pipeline handles it automatically.

**Rule 3 — Forward-fill only, never backward-fill.**
Backward-filling copies a future measurement into the past, giving the model information the doctor did not have. It throws no error and quietly improves your scores. That is what makes it dangerous.

---

## Step 0 — Colab setup

```python
from google.colab import drive
drive.mount('/content/drive')

ROOT = '/content/drive/MyDrive/sepsis_project'
!pip install -q -U imbalanced-learn
```

Save everything to Drive. When the Colab session ends, anything not on Drive is gone.

---

## Step 1 — Load the data

40,336 small `.psv` files (pipe-separated), one per patient. Set A has 20,336, set B has 20,000.

```python
# for each file:
#   pd.read_csv(path, sep='|')
#   add patient_id from the filename
# concatenate into one DataFrame
```

**Use a random subsample of 10,000 patients.** The full set is 1.5 million rows and will be slow or crash on Colab. 10,000 patients gives about 380,000 rows, which is plenty. Fix the random seed and mention the subsampling in your report.

**Save to Drive as Parquet.** Reading 10,000 files is slow. Do it once, then load the Parquet forever after.

### Check before continuing

- About 380,000 rows
- Positive rate about **1.8%** — write this number down
- `ICULOS` increases within each patient

---

## Step 2 — Build features

Four operations, **in this order**.

### 2a. Record what was missing (26 columns)

```python
for col in LAB_COLS:
    df[col + '_missing'] = df[col].isna().astype(int)
```

Do this **first**, before any filling, or these will all be zeros.

Why: the labs are 85–95% missing, and whether a doctor *ordered* a test is itself a clue that they were worried. Real signal, one line of code.

### 2b. Forward-fill inside each patient

```python
df = df.sort_values(['patient_id', 'ICULOS'])
df[CLINICAL_COLS] = df.groupby('patient_id')[CLINICAL_COLS].ffill()
```

This carries the last known reading forward, which is what a doctor actually has. See Rule 3 — no `bfill`.

### 2c. Add trend features (8 columns)

```python
for v in VITALS:                       # HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2
    df[v + '_delta6'] = df.groupby('patient_id')[v].diff(6)
```

This is "how much has this changed in the last 6 hours."

Why it matters: a heart rate going 70 → 85 → 100 is a patient getting worse. One sitting flat at 100 might just be normal for them. Without this, your model sees both as simply "HR = 100" and cannot tell them apart.

The first 6 hours of each patient will be blank here. That is expected — the imputer in Step 4 handles it.

### 2d. Leave the remaining blanks alone

Do **not** fill them now. The pipeline fills them inside each fold. If you fill them here using an average over the whole dataset, that average includes your test rows.

### Result: 74 features

| Block | Count |
|---|---|
| Vitals | 8 |
| Labs | 26 |
| Missing indicators | 26 |
| Trend (delta) | 8 |
| Demographics | 6 |

```python
X      = df[FEATURES]          # everything except patient_id and SepsisLabel
y      = df['SepsisLabel']
groups = df['patient_id']
```

### Why we stopped here

We deliberately did **not** add rolling averages, hours-since-last-test, or clinical scores.

The reason: SMOTE works by measuring distances between points, and distances get less meaningful as you add more columns. So piling on features would hurt the oversampling family specifically, while leaving undersampling untouched. Then you could not tell whether oversampling lost because it is worse or because your feature set was unfair to it.

**A feature that affects your eleven methods differently is a confound, not an improvement.** Keeping features thin is what keeps the comparison honest.

---

## Step 3 — Split

```python
from sklearn.model_selection import StratifiedGroupKFold
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
```

Five folds, and **the same folds for all eleven experiments**. Do not generate new ones per experiment or the comparison is not fair.

### The one assertion you must actually write

```python
for train_idx, test_idx in cv.split(X, y, groups=groups):
    assert len(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])) == 0
```

A real `assert` that stops the program. Not a comment you plan to check later. This is the most important line of code in the project.

---

## Step 4 — Run the experiments

### The pipeline

```python
from imblearn.pipeline import Pipeline      # imblearn, NOT sklearn

pipe = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('sample', sampler if sampler is not None else 'passthrough'),
    ('clf',    RandomForestClassifier(
                   n_estimators=200, max_depth=20, min_samples_leaf=5,
                   class_weight=cw, random_state=0, n_jobs=4)),
])
```

Four things you will be asked about:

**Why imblearn's Pipeline?** It applies the sampler when training and automatically skips it when predicting. That makes Rule 2 impossible to break by accident. sklearn's version does not understand samplers at all.

**Why is the imputer first?** SMOTE measures distances between points and cannot do that with blanks.

**Why is everything inside the pipeline?** So the median comes from training rows only, recalculated fresh every fold.

**Why `n_jobs=4` and not `-1`?** Every parallel worker copies the data. On Colab's ~12 GB, `-1` can run you out of memory faster than it saves time.

### Forest settings stay frozen

Identical for all eleven. Say so in your report, and say why: if you tuned each method separately, you could no longer tell whether a method won because it is better or because it got a luckier search.

### The loop

```python
for exp_id, (name, sampler, cw) in EXPERIMENTS.items():
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        out = f'{ROOT}/results/{exp_id}_fold{fold}.csv'
        if os.path.exists(out):
            continue                  # already done — survives a dead session
        ...
```

**55 runs total** (11 experiments × 5 folds).

That skip-if-exists line matters more than it looks. Colab sessions die without warning. With it, a lost session costs you two minutes instead of an afternoon. Add it from the very first run.

### Check inside every fold

```python
assert abs(y_test.mean() - 0.018) < 0.005     # test data was NOT resampled
assert y_train_after.mean() >= y_train_before.mean()   # sampler did something
```

---

## Step 5 — Measure

Per fold, at a threshold of 0.5:

| Metric | How |
|---|---|
| Accuracy | `accuracy_score` |
| Precision | `precision_score` |
| Recall | `recall_score` |
| F1 | `f1_score` |
| Specificity | `tn / (tn + fp)` from `confusion_matrix` |
| ROC-AUC | `roc_auc_score(y_test, y_prob)` |
| PR-AUC | `average_precision_score(y_test, y_prob)` |

Two notes:
- Specificity has no sklearn function: `tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()`
- Use `average_precision_score` for PR-AUC. The `auc(recall, precision)` version is optimistically biased on imbalanced data.

**Also save the raw predictions** (`patient_id`, `y_true`, `y_prob`) per fold. One extra line, and it means any metric you forgot can be recalculated later without retraining anything.

### The results table

Average across the 5 folds. Every cell is **mean ± standard deviation**. Group by family, with an average row per family.

| Method | Acc | Prec | Recall | F1 | Spec | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| **BASELINE** | | | | | | | |
| No treatment | | | | | | | |
| Balanced RF | | | | | | | |
| **OVERSAMPLING** | | | | | | | |
| Random Oversampling | | | | | | | |
| SMOTE | | | | | | | |
| Borderline-SMOTE | | | | | | | |
| ADASYN | | | | | | | |
| *Family average* | | | | | | | |
| **UNDERSAMPLING** | | | | | | | |
| Random Undersampling | | | | | | | |
| Tomek Links | | | | | | | |
| Edited Nearest Neighbours | | | | | | | |
| *Family average* | | | | | | | |
| **HYBRID** | | | | | | | |
| SMOTEENN | | | | | | | |
| SMOTETomek | | | | | | | |
| *Family average* | | | | | | | |

**Put the PR-AUC baseline (0.018) in the caption.** PR-AUC means nothing without it — 0.15 sounds low until you know random guessing scores 0.018.

---

## How to read your own results

Three things will happen. Expect all three.

**Accuracy will be useless.** B1 will score around 98% while catching almost no sepsis. Put this sentence in your report: *a model predicting "no sepsis" for every row scores 98.2% accuracy on this dataset.* That single line justifies your whole metric choice better than any argument.

**ROC-AUC will be about 0.80 for everything.** It will not separate your methods. PR-AUC will. Say explicitly that PR-AUC is your main metric and why.

**Resampling will raise recall and lower precision.** That trade-off *is* the finding. The question is which method trades best, not which has the single highest number.

### The comparison that makes this a real project

Compare the **spread inside a family** against the **gap between families**.

- If methods inside oversampling differ from each other more than oversampling differs from undersampling, then "use oversampling" is useless advice and the specific method is what counts.
- If families separate cleanly, then family choice is the real decision.

Either answer is worth reporting. That question is what the family structure exists to answer.

---

## Things that will break

**H1 (SMOTEENN) may run out of memory.** Its cleaning step compares every point to its neighbours across the whole resampled set. If it dies, run that one experiment on 5,000 patients and say so in the report.

**U3 (ENN) and U2 (Tomek) carry the same risk** — they also do neighbour searches across your full common class. Test them on 500 patients first.

**O4 (ADASYN) sometimes crashes** with an error about neighbours. Wrap the loop body in `try/except`, print it, record N/A, and let the rest finish.

**U1 throws away almost everything** — about 13,000 rows out of 380,000. Not a bug, that is the method. Expect its results to jump around between folds, and mention that.

---

## Build order

Do not write everything and then run it.

1. Step 1 on **500 patients**. Confirm the 1.8% rate.
2. Step 2. Confirm the missing-indicators are not all zeros and deltas are blank for the first 6 hours.
3. Step 3. Confirm the patient-overlap assertion passes.
4. **B1 only, one fold.** Confirm it looks sane: high accuracy, low recall.
5. Add B2, O1, U1 — the fast ones.
6. Add O2, O3, O4.
7. Add U2, U3.
8. Add H1, H2 last — the ones that break.
9. Scale to 10,000 patients, run all 55.

Every bug you will hit shows up at 500 patients. Finding it there costs a minute. At full scale it costs an hour.

---

## Limitations for your report

Naming these yourself is a strength. Examiners look for exactly this.

1. **Subsampling.** 10,000 of 40,336 patients were used due to Colab compute limits.

2. **Synthetic values for binary features.** 29 of our 74 features are strictly 0 or 1 (26 missingness indicators plus Gender, Unit1, Unit2). SMOTE creates new points by interpolating, so it produces values like 0.37 where only 0 or 1 is valid — roughly 39% of every synthetic row is not physically meaningful. Undersampling and cleaning methods are unaffected, since they only delete real rows. **This may disadvantage the oversampling family and should be considered when comparing families.**

3. **Fixed threshold of 0.5.** Part of what resampling does is shift the decision threshold, so some improvement seen here may be recalibration rather than better learning. Tuning the threshold per method would separate these.

4. **No hyperparameter tuning.** Forest settings were held constant so the sampling method is the only varying factor. No method runs at its personal best.

5. **No separate held-out test set.** Results are 5-fold cross-validation averages. Because eleven methods were compared on the same folds, the best score is likely slightly optimistic.

6. **Synthetic samples may not be clinically realistic.** SMOTE interpolates between patients who may be at different stages of illness, so the resulting "patient-hour" may not describe a person who could exist.

### Optional robustness check (cheap, and impressive)

Re-run the oversampling family with the 26 missingness indicators removed. If the ranking holds, you have shown that limitation #2 did not drive your conclusion. That is a small experiment with a strong payoff in a viva.

---

## Two paragraphs for Chapter 3

**On features:**

> Feature engineering was kept deliberately minimal — missingness indicators, within-patient forward-fill, and 6-hour delta features for vital signs — so that the class imbalance handling technique remains the only substantially varying factor across experiments. Richer temporal features were considered but excluded, as distance-based oversampling methods are sensitive to feature dimensionality in ways that undersampling methods are not, which would confound between-family comparisons.

**On leakage:**

> Resampling was applied strictly within each training fold using an imbalanced-learn pipeline, which applies samplers during fitting and bypasses them at prediction time. Applying resampling before splitting would allow synthetic instances to be interpolated between samples that subsequently fall on opposite sides of the split, and would additionally leave the test set at an artificial class ratio rather than the true 1.8% prevalence. All splits were grouped by patient identifier, as the dataset contains approximately 38 hourly observations per patient and row-level splitting would place near-identical records in both training and test sets.
