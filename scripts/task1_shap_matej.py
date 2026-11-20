"""Úloha 1 – XAI: SHAP nad LightGBM modelem predikujícím ArrDel15.

Features jsou pouze informace známé před odletem (žádný DepDelay/TaxiOut -> bez leakage).
Výstupy (PNG + CSV + metriky) se ukládají do outputs/task1/.
"""
import importlib.util
import os
import sys
from pathlib import Path

# macOS: LightGBM potřebuje libomp. Bez Homebrew použijeme kopii přibalenou v scikit-learn
# (venv-only řešení) a skript se jednou znovu spustí s nastavenou proměnnou prostředí.
if sys.platform == "darwin" and not os.environ.get("_TASK1_OMP_SET"):
    _spec = importlib.util.find_spec("sklearn")
    _dylibs = Path(_spec.submodule_search_locations[0]) / ".dylibs" if _spec else None
    if _dylibs and (_dylibs / "libomp.dylib").exists():
        os.environ["_TASK1_OMP_SET"] = "1"
        _old = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH")
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = f"{_dylibs}:{_old}" if _old else str(_dylibs)
        os.execv(sys.executable, [sys.executable, *sys.argv])

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "flight_delays_2025_01.csv.gz"
OUT = ROOT / "outputs" / "task1"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
N_SHAP = 10_000  # velikost vzorku z testu pro výpočet SHAP

# ---------------------------------------------------------------- data
df = pd.read_csv(DATA)
df = df[(df["Cancelled"] == 0) & (df["Diverted"] == 0)].copy()
print(f"Platné lety: {len(df):,}, podíl ArrDel15=1: {df['ArrDel15'].mean():.3%}")


def hhmm_to_hour(s: pd.Series) -> pd.Series:
    """hhmm (např. 1435) -> desetinná hodina (14.58)."""
    s = s.astype(int)
    return (s // 100 % 24) + (s % 100) / 60


X = pd.DataFrame(
    {
        "DayOfWeek": df["DayOfWeek"],
        "Airline": df["Reporting_Airline"].astype("category"),
        "Origin": df["Origin"].astype("category"),
        "Dest": df["Dest"].astype("category"),
        "CRSDepHour": hhmm_to_hour(df["CRSDepTime"]),
        "CRSArrHour": hhmm_to_hour(df["CRSArrTime"]),
        "CRSElapsedTime": df["CRSElapsedTime"],
        "Distance": df["Distance"],
    }
)
y = df["ArrDel15"].astype(int)
meta = df[["FlightDate", "Reporting_Airline", "Origin", "Dest", "CRSDepTime"]]

X_tr, X_te, y_tr, y_te, _, meta_te = train_test_split(
    X, y, meta, test_size=0.2, stratify=y, random_state=SEED
)
# z trainu oddělíme validační část pro early stopping (test zůstává nedotčený)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=SEED
)
print(f"train {len(X_tr):,} | val {len(X_val):,} | test {len(X_te):,}")

# ---------------------------------------------------------------- model
model = lgb.LGBMClassifier(
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    cat_smooth=50,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)
model.fit(
    X_tr,
    y_tr,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(50, verbose=False)],
)

proba = model.predict_proba(X_te)[:, 1]
# práh maximalizující F1 (třída je nevyvážená, 0.5 by byl nevhodný)
thr_grid = np.linspace(0.05, 0.7, 66)
best_thr = thr_grid[int(np.argmax([f1_score(y_te, proba >= t) for t in thr_grid]))]
metrics = {
    "best_iteration": model.best_iteration_,
    "ROC_AUC": roc_auc_score(y_te, proba),
    "PR_AUC": average_precision_score(y_te, proba),
    "baseline_PR_AUC (prevalence)": y_te.mean(),
    "F1_threshold": best_thr,
    "F1": f1_score(y_te, proba >= best_thr),
}
print("\n=== Metriky na testu ===")
for k, v in metrics.items():
    print(f"{k:32s} {v:.4f}")
pd.Series(metrics).to_csv(OUT / "metrics.csv", header=["value"])

# ---------------------------------------------------------------- SHAP
rng = np.random.RandomState(SEED)
idx = rng.choice(len(X_te), size=min(N_SHAP, len(X_te)), replace=False)
Xs = X_te.iloc[idx]
explainer = shap.TreeExplainer(model)
sv = explainer(Xs)
# u binární klasifikace může mít výstup tvar (n, p, 2) -> vezmi třídu 1
if sv.values.ndim == 3:
    sv = sv[..., 1]
print("\nSHAP base value (log-odds):", float(np.ravel(sv.base_values)[0]))

# --- globální: beeswarm + bar
# kategorické proměnné nemají číselné barvy, proto beeswarm kreslíme nad kódy
Xs_num = Xs.copy()
for c in ["Airline", "Origin", "Dest"]:
    Xs_num[c] = Xs_num[c].cat.codes
sv_num = shap.Explanation(
    values=sv.values, base_values=sv.base_values, data=Xs_num.values,
    feature_names=list(Xs.columns),
)
plt.figure()
shap.plots.beeswarm(sv_num, show=False, max_display=len(Xs.columns))
plt.title("SHAP beeswarm (log-odds ArrDel15=1; kategorie jako kódy)")
plt.tight_layout()
plt.savefig(OUT / "global_beeswarm.png", dpi=150)
plt.close()

plt.figure()
shap.plots.bar(sv, show=False, max_display=len(Xs.columns))
plt.title("Průměrná |SHAP| hodnota")
plt.tight_layout()
plt.savefig(OUT / "global_bar.png", dpi=150)
plt.close()

imp = pd.Series(np.abs(sv.values).mean(0), index=Xs.columns).sort_values(ascending=False)
imp.to_csv(OUT / "global_importance.csv", header=["mean_abs_shap"])
print("\n=== Globální důležitost (mean |SHAP|) ===")
print(imp.round(4).to_string())

# --- dependence plots pro 3 nejdůležitější numerické proměnné
num_cols = [c for c in imp.index if c not in ("Airline", "Origin", "Dest")][:3]
for c in num_cols:
    plt.figure()
    shap.plots.scatter(sv[:, c], color=sv, show=False)
    plt.title(f"Dependence: {c}")
    plt.tight_layout()
    plt.savefig(OUT / f"dependence_{c}.png", dpi=150)
    plt.close()

# --- agregace SHAP po aerolinkách (kategorie čitelněji než beeswarm)
for c in ["Airline"]:
    agg = (
        pd.DataFrame({c: Xs[c].astype(str).values, "shap": sv.values[:, list(Xs.columns).index(c)]})
        .groupby(c)["shap"].agg(["mean", "count"]).sort_values("mean")
    )
    agg.to_csv(OUT / f"shap_by_{c}.csv")
    plt.figure(figsize=(6, 5))
    plt.barh(agg.index, agg["mean"])
    plt.axvline(0, color="k", lw=0.5)
    plt.xlabel("Průměrná SHAP hodnota (log-odds)")
    plt.title(f"Příspěvek proměnné {c}")
    plt.tight_layout()
    plt.savefig(OUT / f"shap_by_{c}.png", dpi=150)
    plt.close()

# --- lokální: waterfall pro 3 lety (jistě zpožděný, jistě včasný, hraniční)
p_s = model.predict_proba(Xs)[:, 1]
y_s = y_te.iloc[idx].values
picks = {
    "delayed_confident": int(np.argmax(np.where(y_s == 1, p_s, -1))),
    "ontime_confident": int(np.argmin(np.where(y_s == 0, p_s, 2))),
    "borderline": int(np.argmin(np.abs(p_s - best_thr))),
}
print("\n=== Lokální vysvětlení ===")
for name, i in picks.items():
    m = meta_te.iloc[idx[i]]
    print(
        f"{name:18s} {m['FlightDate']} {m['Reporting_Airline']} {m['Origin']}->{m['Dest']} "
        f"CRSDep={int(m['CRSDepTime']):04d}  P(delay)={p_s[i]:.3f}  skutečnost={y_s[i]}"
    )
    plt.figure()
    shap.plots.waterfall(sv[i], show=False, max_display=len(Xs.columns))
    plt.title(f"{name}: P={p_s[i]:.2f}, skutečnost={y_s[i]}")
    plt.tight_layout()
    plt.savefig(OUT / f"local_waterfall_{name}.png", dpi=150, bbox_inches="tight")
    plt.close()

print(f"\nVýstupy uloženy do {OUT}")
