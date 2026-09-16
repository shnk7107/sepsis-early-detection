"""
PS-1 IMPUTATION-REDUNDANCY DIAGNOSTIC (V2)
Refactored Part 1
"""

import os
import time
import warnings
import logging

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa

from sklearn.impute import (
    SimpleImputer,
    IterativeImputer,
)

from sklearn.linear_model import BayesianRidge

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import (
    RepeatedStratifiedKFold,
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline as SkPipeline

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import average_precision_score

from scipy import stats


# ============================================================
# CONFIG
# ============================================================

SEED = 42

N_TREES = 150

N_SPLITS = 5

N_REPEATS = 5

TOTAL_FOLDS = N_SPLITS * N_REPEATS

K = N_SPLITS

RHO = 1.0 / (K - 1)

DATA_DIR = os.environ.get("DATA_DIR", "/content")

FEATURES_FILE = os.path.join(DATA_DIR, "features_native.parquet")

TAGS_FILE = os.path.join(DATA_DIR, "feature_tags.csv")

OUT_CSV = os.path.join(
    DATA_DIR,
    "psA_imputation_diagnostic.csv",
)

LOG_FILE = os.path.join(
    DATA_DIR,
    "psA_imputation_diagnostic.log",
)

ARMS = [
    "full",
    "lab_indicators_only",
    "lab_values_dropped",
]

IMPUTERS = [
    "median",
    "mice",
]

SITES = [
    "A",
    "B",
]

COND = "1_none"

MICE_MAX_ITER = int(
    os.environ.get(
        "MICE_MAX_ITER",
        "5",
    )
)

# only use nearest predictors
# dramatically speeds MICE
MICE_NEAREST = int(
    os.environ.get(
        "MICE_NEAREST",
        "25",
    )
)

# minimum observed values required
# before a feature can be used
# as sanity probe

MIN_OBS = 20

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)

console = logging.StreamHandler()

console.setLevel(logging.INFO)

logging.getLogger("").addHandler(console)

# ============================================================
# LOAD
# ============================================================

df = pd.read_parquet(
    FEATURES_FILE
)

tags = pd.read_csv(
    TAGS_FILE
)

src_of = dict(
    zip(
        tags.feature,
        tags.source_var,
    )
)

kind_of = dict(
    zip(
        tags.feature,
        tags.kind,
    )
)

LAB_VARS = set(
    tags[
        tags.group == "lab"
    ].source_var.unique()
)

BINARY_STATIC = [
    "Gender",
    "Unit1",
    "Unit2",
]

feature_cols = [
    c
    for c in df.columns
    if c in set(tags.feature)
]

# ============================================================
# FEATURE HELPERS
# ============================================================

def is_cat(col):

    return (
        kind_of.get(col) == "indicator"
        or col in BINARY_STATIC
    )


DROP = {

    "full":
        lambda c: False,

    "lab_indicators_only":
        lambda c:
        src_of.get(c, c) in LAB_VARS
        and kind_of.get(c) == "indicator",

    "lab_values_dropped":
        lambda c:
        src_of.get(c, c) in LAB_VARS
        and kind_of.get(c) == "value",

}


def arm_columns(arm):

    keep = [
        c
        for c in feature_cols
        if not DROP[arm](c)
    ]

    num = [
        c
        for c in keep
        if not is_cat(c)
    ]

    cat = [
        c
        for c in keep
        if is_cat(c)
    ]

    return num, cat


# ============================================================
# PREPROCESSOR
# ============================================================

class SafeMICEImputer(BaseEstimator, TransformerMixin):

    """
    IterativeImputer can fail when BayesianRidge receives columns that
    remain NaN because a fold has sparse or all-missing numeric features.
    This wrapper runs MICE only on columns with enough observed values and
    uses a deterministic median fallback for the rest.
    """

    def __init__(
        self,
        max_iter=5,
        n_nearest_features=25,
        min_obs=20,
        random_state=42,
    ):

        self.max_iter = max_iter
        self.n_nearest_features = n_nearest_features
        self.min_obs = min_obs
        self.random_state = random_state

    def _as_float_array(self, X):

        if isinstance(X, pd.DataFrame):
            X = X.apply(pd.to_numeric, errors="coerce").to_numpy()
        else:
            X = np.asarray(X, dtype=float)

        X = X.astype(float, copy=False)
        X[~np.isfinite(X)] = np.nan

        return X

    def fit(self, X, y=None):

        X = self._as_float_array(X)

        self.n_features_in_ = X.shape[1]

        observed = np.sum(~np.isnan(X), axis=0)
        self.mice_cols_ = observed >= self.min_obs

        if np.sum(self.mice_cols_) >= 2:

            n_nearest = min(
                self.n_nearest_features,
                max(1, int(np.sum(self.mice_cols_)) - 1),
            )

            self.mice_imputer_ = IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=self.max_iter,
                random_state=self.random_state,
                n_nearest_features=n_nearest,
                initial_strategy="median",
                skip_complete=True,
                sample_posterior=False,
                keep_empty_features=True,
            )

            self.mice_imputer_.fit(
                X[:, self.mice_cols_]
            )

        else:

            self.mice_imputer_ = None

        self.fallback_imputer_ = SimpleImputer(
            strategy="median",
            keep_empty_features=True,
        )

        self.fallback_imputer_.fit(X)

        return self

    def transform(self, X):

        X = self._as_float_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "SafeMICEImputer received a different number of columns "
                "during transform."
            )

        Xt = self.fallback_imputer_.transform(X)

        if self.mice_imputer_ is not None:
            Xt[:, self.mice_cols_] = self.mice_imputer_.transform(
                X[:, self.mice_cols_]
            )

        if np.isnan(Xt).any() or not np.isfinite(Xt).all():
            raise ValueError(
                "SafeMICEImputer could not remove all NaN/inf values."
            )

        return Xt


def make_pre(
    num,
    cat,
    imputer,
):

    if imputer == "median":

        num_imp = SimpleImputer(
            strategy="median",
            keep_empty_features=True,
        )

    elif imputer == "mice":

        num_imp = SafeMICEImputer(
            max_iter=MICE_MAX_ITER,
            n_nearest_features=MICE_NEAREST,
            min_obs=MIN_OBS,
            random_state=SEED,
        )

    else:

        raise ValueError(
            f"Unknown imputer: {imputer}"
        )

    numeric = SkPipeline(

        [
            (
                "imputer",
                num_imp,
            ),

            (
                "scale",
                StandardScaler(),
            ),
        ]
    )

    categorical = SimpleImputer(

        strategy="most_frequent",

        keep_empty_features=True,
    )

    return ColumnTransformer(

        [
            (
                "num",
                numeric,
                num,
            ),

            (
                "cat",
                categorical,
                cat,
            ),
        ]
    )
    
    
# ============================================================
# CELL EVALUATION
# ============================================================

def eval_cell(site, arm, imputer):

    """
    Returns:
        np.ndarray of length TOTAL_FOLDS
        containing AUPRC for each split.
    """

    if "set" in df.columns:
        d = df[df.set == site].copy()
    else:
        d = df[df.site == site].copy()

    X_all = d[feature_cols]
    y = d.label.to_numpy()

    num, cat = arm_columns(arm)

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=SEED,
    )

    aps = []

    sanity_checked = False

    overall_start = time.time()

    logging.info(
        f"\n[{site}] {arm} | {imputer} started"
    )

    for fold_idx, (tr, te) in enumerate(cv.split(X_all, y), start=1):

        fold_start = time.time()

        pre = make_pre(
            num,
            cat,
            imputer,
        )

        Xtr = pre.fit_transform(
            X_all.iloc[tr][num + cat],
            y[tr],
        )

        Xte = pre.transform(
            X_all.iloc[te][num + cat]
        )

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        if np.isnan(Xtr).any():
            raise ValueError(
                f"[{site} {arm}] NaN remained after preprocessing."
            )

        if np.isnan(Xte).any():
            raise ValueError(
                f"[{site} {arm}] NaN remained in test data."
            )

        # --------------------------------------------------
        # Improved sanity check
        # --------------------------------------------------

        if (
            imputer == "mice"
            and not sanity_checked
            and len(num) > 0
        ):

            miss_frac = (
                X_all.iloc[tr][num]
                .isna()
                .mean()
            )

            obs_count = (
                X_all.iloc[tr][num]
                .notna()
                .sum()
            )

            candidates = [

                c

                for c in num

                if (
                    obs_count[c] >= MIN_OBS
                    and 0.05 <= miss_frac[c] <= 0.95
                )

            ]

            if len(candidates):

                probe = max(
                    candidates,
                    key=lambda c: miss_frac[c],
                )

                j = num.index(probe)

                miss_mask = (
                    X_all.iloc[tr][probe]
                    .isna()
                    .to_numpy()
                )

                if miss_mask.sum() >= 5:

                    filled = Xtr[
                        miss_mask,
                        j,
                    ]

                    std = np.std(filled)

                    median = np.nanmedian(
                        X_all.iloc[tr][probe]
                    )

                    identical_to_median = np.allclose(
                        filled,
                        median,
                        atol=1e-6,
                    )

                    if std < 1e-8:

                        if identical_to_median:

                            logging.warning(
                                f"[{site}] Probe {probe}: "
                                "MICE ≈ median."
                            )

                        else:

                            logging.warning(
                                f"[{site}] Probe {probe}: "
                                "constant prediction "
                                "(acceptable for sparse labs)."
                            )

                    else:

                        logging.info(
                            f"[{site}] Probe {probe}: "
                            f"variation confirmed "
                            f"(std={std:.4f})"
                        )

            sanity_checked = True

        # --------------------------------------------------
        # Random Forest
        # --------------------------------------------------

        clf = RandomForestClassifier(

            n_estimators=N_TREES,

            random_state=SEED,

            n_jobs=-1,
        )

        clf.fit(
            Xtr,
            y[tr],
        )

        probs = clf.predict_proba(
            Xte
        )[:, 1]

        ap = average_precision_score(
            y[te],
            probs,
        )

        aps.append(ap)

        # --------------------------------------------------
        # Progress
        # --------------------------------------------------

        elapsed = time.time() - overall_start

        mean_fold = elapsed / fold_idx

        eta = mean_fold * (
            TOTAL_FOLDS - fold_idx
        )

        fold_time = time.time() - fold_start

        print(

            f"[{site} | {arm:20s} | {imputer}] "

            f"Fold "

            f"{fold_idx:02d}/{TOTAL_FOLDS} "

            f"AUPRC={ap:.4f} "

            f"({fold_time:.1f}s) "

            f"ETA={eta/60:.1f} min",

            flush=True,

        )

    aps = np.asarray(aps)

    logging.info(

        f"[{site}] {arm} {imputer} "

        f"finished "

        f"(mean={aps.mean():.4f})"

    )

    return aps

# ============================================================
# Nadeau-Bengio corrected t-test
# ============================================================

def nb_test(delta):

    delta = np.asarray(delta)

    mean = delta.mean()

    var = delta.var(ddof=1)

    se = np.sqrt((1.0 / len(delta) + RHO) * var)

    if se == 0:
        return mean, np.nan, 0.0

    t = mean / se

    p = 2 * stats.t.sf(
        abs(t),
        df=len(delta) - 1,
    )

    ci = (
        stats.t.ppf(
            0.975,
            len(delta) - 1,
        )
        * se
    )

    return mean, p, ci


# ============================================================
# MAIN
# ============================================================

overall_start = time.time()

print("=" * 75)
print("PS-1 IMPUTATION-REDUNDANCY DIAGNOSTIC (V2)")
print("=" * 75)
print(f"Sites     : {SITES}")
print(f"Arms      : {ARMS}")
print(f"Imputers  : {IMPUTERS}")
print(f"CV        : {N_SPLITS}x{N_REPEATS}")
print(f"RF Trees  : {N_TREES}")
print(f"MICE Iter : {MICE_MAX_ITER}")
print("=" * 75)

results = {}

total_jobs = len(SITES) * len(IMPUTERS) * len(ARMS)

job_counter = 0

for site in SITES:

    print("\n" + "=" * 75)
    print(f"Processing Site {site}")
    print("=" * 75)

    for imputer in IMPUTERS:

        print(f"\nImputer : {imputer}")

        for arm in ARMS:

            job_counter += 1

            print(
                "\n"
                f"[Job {job_counter}/{total_jobs}] "
                f"{site} | {imputer} | {arm}"
            )

            start = time.time()

            aps = eval_cell(
                site,
                arm,
                imputer,
            )

            elapsed = time.time() - start

            results[(site, arm, imputer)] = aps

            print(
                f"Finished "
                f"(mean={aps.mean():.4f}, "
                f"std={aps.std():.4f}, "
                f"time={elapsed/60:.2f} min)"
            )


# ============================================================
# FINAL ANALYSIS
# ============================================================

rows = []

print("\n")
print("=" * 75)
print("ABLATION GAP ANALYSIS")
print("=" * 75)

for site in SITES:

    full_med = results[(site, "full", "median")]
    full_mice = results[(site, "full", "mice")]

    ord_med = (
        full_med
        - results[
            (site,
             "lab_indicators_only",
             "median")
        ]
    )

    ord_mice = (
        full_mice
        - results[
            (site,
             "lab_indicators_only",
             "mice")
        ]
    )

    val_med = (
        full_med
        - results[
            (site,
             "lab_values_dropped",
             "median")
        ]
    )

    val_mice = (
        full_mice
        - results[
            (site,
             "lab_values_dropped",
             "mice")
        ]
    )

    analyses = [

        (
            "ordering",
            ord_med,
            ord_mice,
            full_med,
            full_mice,
        ),

        (
            "values",
            val_med,
            val_mice,
            full_med,
            full_mice,
        ),

    ]

    print("\n")
    print(f"Site {site}")
    print("-" * 60)

    for (
        channel,
        med,
        mice,
        base_med,
        base_mice,
    ) in analyses:

        pct_med = (
            med.mean()
            / base_med.mean()
            * 100
        )

        pct_mice = (
            mice.mean()
            / base_mice.mean()
            * 100
        )

        delta = mice - med

        mean_delta, p, ci = nb_test(delta)

        stable = (
            "YES"
            if p >= 0.05
            else "NO"
        )

        print(
            f"{channel:10s}"
            f" median={pct_med:+6.2f}%"
            f" mice={pct_mice:+6.2f}%"
            f" Δ={mean_delta*100:+6.2f}pp"
            f" p={p:.4f}"
            f" stable={stable}"
        )

        rows.append(

            {

                "site": site,

                "channel": channel,

                "gap_median_pct": pct_med,

                "gap_mice_pct": pct_mice,

                "delta_pp": mean_delta * 100,

                "p_change": p,

                "ci_halfwidth": ci * 100,

                "stable": stable,

            }

        )


# ============================================================
# SAVE
# ============================================================

out = pd.DataFrame(rows)

out.to_csv(
    OUT_CSV,
    index=False,
)

print("\n")
print("=" * 75)
print("SUMMARY")
print("=" * 75)

print(out)

print("\nCSV saved to")

print(OUT_CSV)

elapsed = time.time() - overall_start

print(
    f"\nTotal runtime : "
    f"{elapsed/60:.2f} minutes"
)

logging.info(
    f"Completed in {elapsed/60:.2f} minutes"
)
