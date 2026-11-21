"""Úloha 4 – Hledání podskupin s chybami modelu (slice discovery) + kalibrace.

Model: stejný LightGBM jako v úloze 1 (features známé před odletem).
Podskupiny: beam search nad konjunkcemi selektorů (vlastní implementace, bez pysubgroup).
  - kvalita = t-statistika průměrného rezidua (y - p) -> podskupiny, kde je model
    systematicky podhodnocený (y > p) nebo nadhodnocený (y < p); zohledňuje velikost.
  - objev na polovině testu (A), ověření na druhé polovině (B) -> kontrola, že nejde o šum.
  - jako popisné atributy slouží i FlightDate/týden, které model nevidí.
Výstupy do outputs/task4/.
"""
import importlib.util
import os
import sys
from pathlib import Path

# macOS: LightGBM potřebuje libomp (viz task1)
if sys.platform == "darwin" and not os.environ.get("_TASK4_OMP_SET"):
    _spec = importlib.util.find_spec("sklearn")
    _dylibs = Path(_spec.submodule_search_locations[0]) / ".dylibs" if _spec else None
    if _dylibs and (_dylibs / "libomp.dylib").exists():
        os.environ["_TASK4_OMP_SET"] = "1"
        _old = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH")
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = f"{_dylibs}:{_old}" if _old else str(_dylibs)
        os.execv(sys.executable, [sys.executable, *sys.argv])

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "flight_delays_2025_01.csv.gz"
OUT = ROOT / "outputs" / "task4"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
MIN_SIZE = 500     # minimální velikost podskupiny (na polovině A)
BEAM = 30
DEPTH = 3
TOP_K = 25

# ---------------------------------------------------------------- data + model
df = pd.read_csv(DATA)
df = df[(df["Cancelled"] == 0) & (df["Diverted"] == 0)].copy().reset_index(drop=True)


def hhmm_to_hour(s):
    s = s.astype(int)
    return (s // 100 % 24) + (s % 100) / 60


X = pd.DataFrame({
    "DayOfWeek": df["DayOfWeek"],
    "Airline": df["Reporting_Airline"].astype("category"),
    "Origin": df["Origin"].astype("category"),
    "Dest": df["Dest"].astype("category"),
    "CRSDepHour": hhmm_to_hour(df["CRSDepTime"]),
    "CRSArrHour": hhmm_to_hour(df["CRSArrTime"]),
    "CRSElapsedTime": df["CRSElapsedTime"],
    "Distance": df["Distance"],
})
y = df["ArrDel15"].astype(int)
idx_all = np.arange(len(df))

tr_i, te_i = train_test_split(idx_all, test_size=0.2, stratify=y, random_state=SEED)
tr_i, val_i = train_test_split(tr_i, test_size=0.1, stratify=y.iloc[tr_i], random_state=SEED)

model = lgb.LGBMClassifier(
    n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, cat_smooth=50,
    random_state=SEED, n_jobs=-1, verbose=-1,
)
model.fit(X.iloc[tr_i], y.iloc[tr_i], eval_set=[(X.iloc[val_i], y.iloc[val_i])],
          eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])

# test rozdělíme na A (objev podskupin + fit rekalibrace) a B (ověření)
A_i, B_i = train_test_split(te_i, test_size=0.5, stratify=y.iloc[te_i], random_state=SEED)
p = model.predict_proba(X)[:, 1]
print(f"Test AUC={roc_auc_score(y.iloc[te_i], p[te_i]):.4f}, "
      f"A={len(A_i):,}, B={len(B_i):,}")

# ---------------------------------------------------------------- 1) globální kalibrace
def ece(yt, pp, bins=10):
    b = pd.qcut(pp, bins, duplicates="drop")
    g = pd.DataFrame({"y": yt, "p": pp, "b": b}).groupby("b", observed=True).agg(
        y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((g["n"] * (g["y"] - g["p"]).abs()).sum() / g["n"].sum())


# isotonická rekalibrace naučená na A, vyhodnocená na B
iso = IsotonicRegression(out_of_bounds="clip").fit(p[A_i], y.iloc[A_i])
p_iso = iso.predict(p)
cal_rows = []
for name, pp in [("LightGBM raw", p), ("LightGBM + isotonic (fit on A)", p_iso)]:
    cal_rows.append({
        "model": name, "set": "B (holdout)",
        "Brier": brier_score_loss(y.iloc[B_i], pp[B_i]),
        "LogLoss": log_loss(y.iloc[B_i], np.clip(pp[B_i], 1e-6, 1 - 1e-6)),
        "ECE": ece(y.iloc[B_i].values, pp[B_i]),
        "mean_pred": pp[B_i].mean(), "mean_true": y.iloc[B_i].mean(),
    })
cal = pd.DataFrame(cal_rows)
cal.to_csv(OUT / "calibration_global.csv", index=False)
print("\n=== Globální kalibrace (B) ===")
print(cal.round(4).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for name, pp in [("raw", p), ("isotonic", p_iso)]:
    fy, fp = calibration_curve(y.iloc[B_i], pp[B_i], n_bins=15, strategy="quantile")
    ax[0].plot(fp, fy, marker="o", label=name)
ax[0].plot([0, 1], [0, 1], "k--", lw=0.7)
ax[0].set(xlabel="Predikovaná pravděpodobnost", ylabel="Skutečný podíl ArrDel15",
          title="Reliability diagram (holdout B)")
ax[0].legend()
ax[1].hist(p[B_i], bins=50, color="steelblue")
ax[1].set(xlabel="Predikovaná pravděpodobnost", title="Rozdělení predikcí")
plt.tight_layout()
plt.savefig(OUT / "calibration_global.png", dpi=150)
plt.close()

# ---------------------------------------------------------------- 2) selektory
res = (y.values - p)  # reziduum: >0 = model podhodnocuje riziko zpoždění
D = pd.DataFrame({
    "Airline": df["Reporting_Airline"],
    "DayOfWeek": df["DayOfWeek"].map(dict(enumerate(["", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]))),
    "DepBlock": pd.cut(hhmm_to_hour(df["CRSDepTime"]), [0, 6, 9, 12, 15, 18, 21, 24],
                       right=False, labels=["00-06", "06-09", "09-12", "12-15", "15-18", "18-21", "21-24"]).astype(str),
    "ArrBlock": pd.cut(hhmm_to_hour(df["CRSArrTime"]), [0, 6, 9, 12, 15, 18, 21, 24],
                       right=False, labels=["00-06", "06-09", "09-12", "12-15", "15-18", "18-21", "21-24"]).astype(str),
    "Distance": pd.qcut(df["Distance"], 5, duplicates="drop").astype(str),
    "Elapsed": pd.qcut(df["CRSElapsedTime"], 5, duplicates="drop").astype(str),
    "Week": "W" + ((df["DayofMonth"] - 1) // 7 + 1).astype(str),
    "Day": "Jan" + df["DayofMonth"].astype(str).str.zfill(2),
    "Origin": df["Origin"],
    "Dest": df["Dest"],
})
# vysoce kardinální atributy (letiště): jen ta s dostatkem letů
for c in ["Origin", "Dest"]:
    vc = D.loc[A_i, c].value_counts()
    D.loc[~D[c].isin(vc[vc >= MIN_SIZE].index), c] = "other"

selectors = {}
for c in D.columns:
    for v in D[c].unique():
        if v == "other":
            continue
        selectors[f"{c}={v}"] = (D[c] == v).values

# ---------------------------------------------------------------- 3) beam search na A
mask_A = np.zeros(len(df), bool)
mask_A[A_i] = True
resA = res[mask_A]
sd = resA.std()


def quality(m):
    """t-statistika průměrného rezidua v podskupině (v A)."""
    n = m.sum()
    if n < MIN_SIZE:
        return None
    return resA[m[mask_A]].mean() * np.sqrt(n) / sd


def search(sign, allowed):
    """sign=+1: model podhodnocuje (res>0); sign=-1: nadhodnocuje."""
    beam = [((), np.ones(len(df), bool), 0.0)]
    found = {}
    for depth in range(DEPTH):
        cand = {}
        for conds, m, _ in beam:
            used = {c.split("=")[0] for c in conds}
            for name, sm in selectors.items():
                if name.split("=")[0] in used or name.split("=")[0] not in allowed:
                    continue
                key = frozenset(conds + (name,))
                if key in cand:
                    continue
                nm = m & sm
                q = quality(nm)
                if q is None:
                    continue
                cand[key] = (tuple(sorted(key)), nm, sign * q)
        ranked = sorted(cand.values(), key=lambda t: -t[2])
        beam = ranked[:BEAM]
        for conds, nm, q in beam:
            found[conds] = q
    return sorted(found.items(), key=lambda kv: -kv[1])


rows = []
mask_B = np.zeros(len(df), bool)
mask_B[B_i] = True
DATE_ATTRS = {"Day", "Week"}
ALL_ATTRS = set(D.columns)
MODEL_ATTRS = ALL_ATTRS - DATE_ATTRS
for scope, allowed in [("all", ALL_ATTRS), ("model_visible", MODEL_ATTRS)]:
  for sign, label in [(+1, "underestimated"), (-1, "overestimated")]:
    seen = set()
    n_kept = 0
    for conds, q in search(sign, allowed):
        if n_kept >= TOP_K:
            break
        m = np.ones(len(df), bool)
        for c in conds:
            m &= selectors[c]
        sig = hash(m[mask_A].tobytes())  # shodné pokrytí = redundantní popis téže podskupiny
        if sig in seen:
            continue
        seen.add(sig)
        n_kept += 1
        rec = {"scope": scope, "direction": label, "subgroup": " & ".join(conds)}
        for nm_, mk in [("A", mask_A), ("B", mask_B)]:
            mm = m & mk
            n = mm.sum()
            r = res[mm]
            rec.update({
                f"n_{nm_}": int(n),
                f"true_rate_{nm_}": y.values[mm].mean(),
                f"mean_pred_{nm_}": p[mm].mean(),
                f"mean_resid_{nm_}": r.mean(),
                f"t_{nm_}": r.mean() * np.sqrt(n) / res[mk].std(),
                f"AUC_{nm_}": roc_auc_score(y.values[mm], p[mm]) if 0 < y.values[mm].sum() < n else np.nan,
            })
        # po rekalibraci na A: zbyde v B stále chyba?
        rec["mean_resid_B_after_isotonic"] = (y.values[m & mask_B] - p_iso[m & mask_B]).mean()
        rec["replicated_in_B"] = bool(np.sign(rec["mean_resid_B"]) == sign and abs(rec["t_B"]) > 2)
        rows.append(rec)

sub = pd.DataFrame(rows)
sub.to_csv(OUT / "subgroups.csv", index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 60)
cols = ["subgroup", "n_A", "true_rate_A", "mean_pred_A", "mean_resid_A", "t_A", "n_B", "mean_resid_B", "t_B",
        "replicated_in_B"]
for sc in ["all", "model_visible"]:
  for d in ["underestimated", "overestimated"]:
    print(f"\n=== [{sc}] Top podskupiny: model {d} riziko ===")
    print(sub[(sub.direction == d) & (sub.scope == sc)][cols].head(8).round(3).to_string(index=False))
print(f"\nReplikováno v B: {sub.replicated_in_B.sum()}/{len(sub)}")

# ---------------------------------------------------------------- 4) kalibrace v top podskupinách
top = pd.concat([sub[(sub.direction == d) & (sub.scope == "model_visible")].head(4) for d in ["underestimated", "overestimated"]])
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.plot([0, 1], [0, 1], "k--", lw=0.7)
for _, r in top.iterrows():
    m = np.ones(len(df), bool)
    for c in r["subgroup"].split(" & "):
        m &= selectors[c]
    m &= mask_B
    if m.sum() < 200:
        continue
    fy, fp = calibration_curve(y.values[m], p[m], n_bins=5, strategy="quantile")
    lbl = r["subgroup"] if len(r["subgroup"]) < 45 else r["subgroup"][:42] + "…"
    ax.plot(fp, fy, marker="o", ms=3, lw=1, label=lbl)
ax.set(xlabel="Predikovaná pravděpodobnost", ylabel="Skutečný podíl", title="Kalibrace v top podskupinách bez datumu (holdout B)")
ax.legend(fontsize=6)
plt.tight_layout()
plt.savefig(OUT / "calibration_subgroups.png", dpi=150)
plt.close()

# bar: reziduum top podskupin (A vs B)
fig, ax = plt.subplots(figsize=(9, 7))
t = pd.concat([sub[(sub.direction == d) & (sub.scope == "model_visible")].head(8) for d in ["underestimated", "overestimated"]])
yy = np.arange(len(t))
ax.barh(yy - 0.2, t["mean_resid_A"], 0.4, label="A (objev)")
ax.barh(yy + 0.2, t["mean_resid_B"], 0.4, label="B (ověření)")
ax.set_yticks(yy)
ax.set_yticklabels([s if len(s) < 55 else s[:52] + "…" for s in t["subgroup"]], fontsize=7)
ax.invert_yaxis()
ax.axvline(0, color="k", lw=0.5)
ax.set(xlabel="Průměrné reziduum y − p (>0: model podhodnocuje)", title="Podskupiny s největší systematickou chybou")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "top_subgroups.png", dpi=150)
plt.close()

# ---------------------------------------------------------------- 5) jednorozměrný přehled (slice finder baseline)
uni = []
for c in D.columns:
    for v, g in D.loc[B_i].groupby(c):
        if len(g) < 300:
            continue
        ii = g.index.values
        uni.append({"attribute": c, "value": v, "n": len(ii), "true_rate": y.values[ii].mean(),
                    "mean_pred": p[ii].mean(), "mean_resid": res[ii].mean(),
                    "logloss": log_loss(y.values[ii], np.clip(p[ii], 1e-6, 1 - 1e-6), labels=[0, 1])})
uni = pd.DataFrame(uni).sort_values("mean_resid")
uni.to_csv(OUT / "univariate_slices_B.csv", index=False)
print("\n=== Nejhorší jednorozměrné řezy (B, |reziduum|) ===")
print(uni.reindex(uni.mean_resid.abs().sort_values(ascending=False).index).head(8).round(3).to_string(index=False))

print(f"\nVýstupy uloženy do {OUT}")
