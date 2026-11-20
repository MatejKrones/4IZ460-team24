"""Úloha 2 – Surrogate pravidla: rozhodovací strom napodobující LightGBM z úlohy 1.

Postup:
 1. Stejná data, split, seed a LightGBM jako v úloze 1 (viz task1_shap_matej.py).
 2. Cíl surrogate = binární predikce modelu (proba >= práh maximalizující F1 na testu).
 3. Surrogate = mělký rozhodovací strom (trénovaný na predikcích modelu na trainu),
    fidelita se měří na testu (shoda se SAMOTNÝM modelem, ne s pravdou).
 4. Z listů stromu se vytáhnou pravidla (support, confidence, lift jako u GUHA).
 5. Srovnání s pravidly 4ft-Mineru z Bloku 1 (guha/run16_matej.json): GUHA pravidla se
    přepočítají na originálních datech a ověří se proti kvantifikátorům z JSONu.

Výstupy: outputs/task2/
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

# macOS: LightGBM potřebuje libomp (viz úloha 1) – re-exec s DYLD_FALLBACK_LIBRARY_PATH.
if sys.platform == "darwin" and not os.environ.get("_TASK2_OMP_SET"):
    _spec = importlib.util.find_spec("sklearn")
    _dylibs = Path(_spec.submodule_search_locations[0]) / ".dylibs" if _spec else None
    if _dylibs and (_dylibs / "libomp.dylib").exists():
        os.environ["_TASK2_OMP_SET"] = "1"
        _old = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH")
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = f"{_dylibs}:{_old}" if _old else str(_dylibs)
        os.execv(sys.executable, [sys.executable, *sys.argv])

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "flight_delays_2025_01.csv.gz"
GUHA_JSON = ROOT / "guha" / "run16_matej.json"
OUT = ROOT / "outputs" / "task2"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
TOP_K = 15  # kolik nejčastějších Origin/Dest dostane vlastní 0/1 sloupec
MAIN_DEPTH = 4  # hloubka "hlavního" čitelného stromu (<= 16 listů)
MIN_LEAF = 500  # min. počet letů v listu (stabilita a čitelnost pravidel)

# ================================================================ 1) data + model (jako úloha 1)
df = pd.read_csv(DATA)
df = df[(df["Cancelled"] == 0) & (df["Diverted"] == 0)].copy().reset_index(drop=True)
print(f"Platné lety: {len(df):,}, podíl ArrDel15=1: {df['ArrDel15'].mean():.3%}")


def hhmm_to_hour(s: pd.Series) -> pd.Series:
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
idx_all = np.arange(len(df))

# stejné volání a pořadí jako v úloze 1 -> identické množiny train/val/test
X_tr, X_te, y_tr, y_te, i_tr, i_te = train_test_split(
    X, y, idx_all, test_size=0.2, stratify=y, random_state=SEED
)
X_tr, X_val, y_tr, y_val, i_tr, i_val = train_test_split(
    X_tr, y_tr, i_tr, test_size=0.1, stratify=y_tr, random_state=SEED
)
print(f"train {len(X_tr):,} | val {len(X_val):,} | test {len(X_te):,}")

model = lgb.LGBMClassifier(
    n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, cat_smooth=50,
    random_state=SEED, n_jobs=-1, verbose=-1,
)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="auc",
          callbacks=[lgb.early_stopping(50, verbose=False)])

proba_tr = model.predict_proba(X_tr)[:, 1]
proba_te = model.predict_proba(X_te)[:, 1]
proba_all = model.predict_proba(X)[:, 1]

thr_grid = np.linspace(0.05, 0.7, 66)
THR = thr_grid[int(np.argmax([f1_score(y_te, proba_te >= t) for t in thr_grid]))]
print(f"LightGBM: ROC AUC test {roc_auc_score(y_te, proba_te):.4f}, práh F1 = {THR:.2f}")

# cílová proměnná surrogate: co by model označil jako "zpožděný let"
m_tr = (proba_tr >= THR).astype(int)
m_te = (proba_te >= THR).astype(int)
m_all = (proba_all >= THR).astype(int)
print(f"Model označuje jako zpožděné: train {m_tr.mean():.1%}, test {m_te.mean():.1%}")


# ================================================================ 2) feature matice pro strom
def build_tree_features(X_: pd.DataFrame, top_origin, top_dest, airlines) -> pd.DataFrame:
    """Numerické sloupce + 0/1 indikátory (čitelná pravidla místo kategorických splitů)."""
    F = X_[["DayOfWeek", "CRSDepHour", "CRSArrHour", "CRSElapsedTime", "Distance"]].copy()
    for a in airlines:
        F[f"Airline={a}"] = (X_["Airline"].astype(str) == a).astype(int)
    for o in top_origin:
        F[f"Origin={o}"] = (X_["Origin"].astype(str) == o).astype(int)
    for d in top_dest:
        F[f"Dest={d}"] = (X_["Dest"].astype(str) == d).astype(int)
    return F


top_origin = X_tr["Origin"].astype(str).value_counts().index[:TOP_K].tolist()
top_dest = X_tr["Dest"].astype(str).value_counts().index[:TOP_K].tolist()
airlines = sorted(X_tr["Airline"].astype(str).unique())
F_tr = build_tree_features(X_tr, top_origin, top_dest, airlines)
F_te = build_tree_features(X_te, top_origin, top_dest, airlines)
F_all = build_tree_features(X, top_origin, top_dest, airlines)
feat_names = F_tr.columns.tolist()


# ================================================================ 3) fidelita vs. hloubka
def fidelity(tree, F, m, y_true=None):
    pred = tree.predict(F)
    pr = tree.predict_proba(F)[:, 1]
    out = {
        "fidelity_accuracy": accuracy_score(m, pred),
        "fidelity_balanced_acc": balanced_accuracy_score(m, pred),
        "fidelity_F1_pos": f1_score(m, pred),
        "fidelity_AUC": roc_auc_score(m, pr),
    }
    if y_true is not None:
        out["F1_vs_truth"] = f1_score(y_true, pred)
    return out


rows = []
for depth in range(1, 11):
    t = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=MIN_LEAF, random_state=SEED)
    t.fit(F_tr, m_tr)
    rows.append({"max_depth": depth, "n_leaves": t.get_n_leaves(),
                 **fidelity(t, F_te, m_te, y_te.values)})
curve = pd.DataFrame(rows)
curve.to_csv(OUT / "fidelity_vs_depth.csv", index=False)
print("\n=== Fidelita vs. hloubka (test) ===")
print(curve.round(4).to_string(index=False))
print(f"Reference: F1 samotného LightGBM vs. pravda = {f1_score(y_te, m_te):.4f}")

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(curve["max_depth"], curve["fidelity_accuracy"], "o-", label="Accuracy (shoda s modelem)")
ax.plot(curve["max_depth"], curve["fidelity_F1_pos"], "s-", label="F1 pozitivní třídy (shoda s modelem)")
ax.plot(curve["max_depth"], curve["fidelity_balanced_acc"], "^-", label="Balanced accuracy")
ax.axvline(MAIN_DEPTH, color="grey", ls="--", lw=1)
ax.set_xlabel("max_depth surrogate stromu")
ax.set_ylabel("Fidelita na testu")
ax.set_title("Fidelita surrogate vs. hloubka stromu")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "fidelity_vs_depth.png", dpi=150)
plt.close(fig)

# ================================================================ 4) hlavní strom + pravidla
tree = DecisionTreeClassifier(max_depth=MAIN_DEPTH, min_samples_leaf=MIN_LEAF, random_state=SEED)
tree.fit(F_tr, m_tr)
fid = fidelity(tree, F_te, m_te, y_te.values)
fid_tr = fidelity(tree, F_tr, m_tr)
pd.DataFrame({"test": fid, "train": fid_tr}).to_csv(OUT / "fidelity_main_tree.csv")
print(f"\n=== Hlavní strom (depth={MAIN_DEPTH}, {tree.get_n_leaves()} listů) ===")
print({k: round(v, 4) for k, v in fid.items()})
(OUT / "tree_text.txt").write_text(export_text(tree, feature_names=feat_names, show_weights=True))

fig, ax = plt.subplots(figsize=(26, 11))
plot_tree(tree, feature_names=feat_names, class_names=["model: OK", "model: zpožděn"],
          filled=True, rounded=True, impurity=False, proportion=True, fontsize=7, ax=ax)
ax.set_title(f"Surrogate strom (depth={MAIN_DEPTH}) napodobující LightGBM; fidelita acc={fid['fidelity_accuracy']:.3f}")
fig.tight_layout()
fig.savefig(OUT / "surrogate_tree.png", dpi=130)
plt.close(fig)

base_rate = y.mean()  # 18,8 % – stejné referenční konfidence jako u GUHA (aad = conf/base - 1)



DAY = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}


def _simplify(conds):
    """Sloučí podmínky po featurách (tightest bounds), DayOfWeek zobrazí jako množinu dnů."""
    lo, hi = {}, {}
    for f, op, th in conds:
        if op == "<=":
            hi[f] = min(hi.get(f, np.inf), th)
        else:
            lo[f] = max(lo.get(f, -np.inf), th)
    out, eq = [], set()
    for f in feat_names:
        if f not in lo and f not in hi:
            continue
        l, h = lo.get(f, -np.inf), hi.get(f, np.inf)
        if "=" in f:  # indikátor 0/1
            if l >= 0.5:
                out.append(f.replace("=", " == "))
                eq.add(f.split("=")[0])
            else:
                out.append(f.replace("=", " != "))
        elif f == "DayOfWeek":
            days = [DAY[d] for d in range(1, 8) if l < d <= h]
            out.append("DayOfWeek in {" + ", ".join(days) + "}")
        else:
            if np.isfinite(l) and np.isfinite(h):
                out.append(f"{l:.2f} < {f} <= {h:.2f}")
            elif np.isfinite(l):
                out.append(f"{f} > {l:.2f}")
            else:
                out.append(f"{f} <= {h:.2f}")
    # "Origin != X" je zbytečné, když už je "Origin == Y"
    return [t for t in out if not (" != " in t and t.split(" != ")[0] in eq)]


def leaf_rules(tree, F: pd.DataFrame) -> pd.DataFrame:
    """Každý list = pravidlo (cesta od kořene). Metriky na celých datech (srovnatelné s GUHA) i na testu."""
    t = tree.tree_
    path_conds = {}

    def walk(node, conds):
        if t.children_left[node] == t.children_right[node]:
            path_conds[node] = conds
            return
        f, th = feat_names[t.feature[node]], t.threshold[node]
        walk(t.children_left[node], conds + [(f, "<=", th)])
        walk(t.children_right[node], conds + [(f, ">", th)])

    walk(0, [])
    leaf_all = tree.apply(F_all)
    leaf_te = tree.apply(F_te)
    recs = []
    for leaf, conds in path_conds.items():
        txt = _simplify(conds)
        a = leaf_all == leaf
        te = leaf_te == leaf
        n = int(a.sum())
        conf_true = y.values[a].mean()
        recs.append(
            {
                "leaf": leaf,
                "rule": " AND ".join(txt) if txt else "(vše)",
                "predicts_delayed": int(tree.predict_proba(F_all[a].iloc[:1])[0, 1] >= 0.5),
                "support_n": n,
                "support_rel": n / len(df),
                "model_pos_share": m_all[a].mean(),       # konfidence vůči modelu (fidelita listu)
                "conf_ArrDel15": conf_true,                # konfidence vůči pravdě (jako GUHA conf)
                "lift": conf_true / base_rate,             # GUHA aad = lift - 1
                "aad": conf_true / base_rate - 1,
                "mean_model_proba": proba_all[a].mean(),
                "test_n": int(te.sum()),
                "test_conf_ArrDel15": y_te.values[te].mean() if te.any() else np.nan,
            }
        )
    return pd.DataFrame(recs).sort_values(["predicts_delayed", "lift"], ascending=False)


rules = leaf_rules(tree, F_all)
rules.to_csv(OUT / "surrogate_rules.csv", index=False)
print("\n=== Pravidla surrogate (listy), řazeno podle lift ===")
pd.set_option("display.width", 250, "display.max_colwidth", 110)
print(rules[["rule", "predicts_delayed", "support_n", "model_pos_share", "conf_ArrDel15", "lift"]]
      .round(3).to_string(index=False))

# ================================================================ 5) srovnání s GUHA (4ft-Miner)
# Binning SchedDepTime odvozen z názvů kategorií v JSONu; správnost ověřuje shoda čtyřpolních tabulek.
hour = df["CRSDepTime"].astype(int) // 100 % 24
sched = pd.cut(hour, bins=[-1, 5, 9, 13, 17, 23],
               labels=["1_Night_0_6", "2_Morning_6_10", "3_Midday_10_14",
                       "4_Afternoon_14_18", "5_Evening_18_24"]).astype(str)
dow = df["DayOfWeek"].map({1: "1_Mon", 2: "2_Tue", 3: "3_Wed", 4: "4_Thu", 5: "5_Fri", 6: "6_Sat", 7: "7_Sun"})
cols = {"DayOfWeek": dow, "SchedDepTime": sched}
is_jan6 = (df["FlightDate"] == "2025-01-06").values

guha = json.loads(GUHA_JSON.read_text())["result"]["rules"]
grecs = []
for r in guha:
    mask = np.ones(len(df), bool)
    for part in r["structure"]["ante"]:
        mask &= cols[part["variable"]].isin(part["categories"]).values
    n = int(mask.sum())
    a = int((mask & (y.values == 1)).sum())
    q = r["quantifiers"]
    ok = (a == q["fourfold"][0]) and (n - a == q["fourfold"][1])  # a, b z čtyřpolní tabulky
    conf = a / n
    # kontrola artefaktu 6. 1.: stejná pravidla bez tohoto dne
    m2 = mask & ~is_jan6
    conf_no6 = (y.values[m2] == 1).mean() if m2.any() else np.nan
    # co o antecedentu říká model a surrogate
    sur_pred = tree.predict(F_all[mask])
    grecs.append(
        {
            "guha_id": r["id"],
            "rule": r["text"].split("=>")[0].strip(),
            "reproduced_fourfold": ok,
            "base": a, "ante_n": n,
            "conf": conf, "conf_json": q["conf"],
            "aad": conf / base_rate - 1, "aad_json": q["aad"],
            "conf_without_2025_01_06": conf_no6,
            "aad_without_2025_01_06": conf_no6 / base_rate - 1,
            "model_pos_share": m_all[mask].mean(),
            "mean_model_proba": proba_all[mask].mean(),
            "surrogate_pos_share": sur_pred.mean(),
            "surrogate_model_agreement": (sur_pred == m_all[mask]).mean(),
        }
    )
gdf = pd.DataFrame(grecs)
gdf.to_csv(OUT / "guha_vs_model.csv", index=False)
print("\n=== GUHA pravidla (4ft-Miner) přepočtená na originálních datech ===")
print(f"Čtyřpolní tabulky reprodukovány: {int(gdf['reproduced_fourfold'].sum())}/{len(gdf)}")
print(gdf[["guha_id", "rule", "reproduced_fourfold", "base", "conf", "aad",
           "conf_without_2025_01_06", "model_pos_share", "surrogate_pos_share"]]
      .round(3).to_string(index=False))

# --- obrácený směr: jak moc "vidí" GUHA antecedenty model? a jak vysvětluje model důvody pravidel surrogate?
union = np.zeros(len(df), bool)
for r in guha:
    mk = np.ones(len(df), bool)
    for part in r["structure"]["ante"]:
        mk &= cols[part["variable"]].isin(part["categories"]).values
    union |= mk
pos_leaves = rules[rules["predicts_delayed"] == 1]["leaf"].tolist()
sur_pos = np.isin(tree.apply(F_all), pos_leaves)
summary = {
    "GUHA_union_n": int(union.sum()),
    "GUHA_union_conf": y.values[union].mean(),
    "GUHA_union_model_pos_share": m_all[union].mean(),
    "surrogate_pos_n": int(sur_pos.sum()),
    "surrogate_pos_conf_ArrDel15": y.values[sur_pos].mean() if sur_pos.any() else np.nan,
    "surrogate_pos_lift": y.values[sur_pos].mean() / base_rate if sur_pos.any() else np.nan,
    "overlap_n(GUHA union & surrogate pos)": int((union & sur_pos).sum()),
    "overlap_share_of_surrogate_pos": (union & sur_pos).sum() / max(sur_pos.sum(), 1),
    "overlap_share_of_GUHA_union": (union & sur_pos).sum() / union.sum(),
    "base_rate": base_rate,
}
pd.Series(summary).to_csv(OUT / "guha_vs_surrogate_summary.csv", header=["value"])
print("\n=== Souhrn překryvu GUHA vs. surrogate ===")
for k, v in summary.items():
    print(f"{k:42s} {v:.4f}" if isinstance(v, float) else f"{k:42s} {v}")

# graf: konfidence GUHA pravidel (s a bez 6. 1.) vs. model
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(gdf))
ax.bar(x - 0.2, gdf["conf"], 0.4, label="GUHA conf (celá data)")
ax.bar(x + 0.2, gdf["conf_without_2025_01_06"], 0.4, label="GUHA conf bez 6. 1.")
ax.plot(x, gdf["mean_model_proba"], "kD-", label="Ø proba LightGBM v antecedentu")
ax.axhline(base_rate, color="red", ls="--", lw=1, label=f"základ {base_rate:.1%}")
ax.set_xticks(x)
ax.set_xticklabels([f"#{i}" for i in gdf["guha_id"]])
ax.set_xlabel("GUHA pravidlo (4ft-Miner, run 16)")
ax.set_ylabel("P(ArrDel15=1 | antecedent)")
ax.set_title("GUHA pravidla vs. model: vliv 6. 1. a pohled modelu")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT / "guha_vs_model.png", dpi=150)
plt.close(fig)

print(f"\nHotovo, výstupy v {OUT}")
