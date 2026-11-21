"""Úloha 3 – Detekce anomálií: Isolation Forest na profilech letiště (Origin) × den.

Každý řádek = jedno odletové letiště v jednom dni (profil z odletů). Isolation Forest
hledá neobvyklé kombinace (zpoždění, rušení, taxi-out, příčiny zpoždění...).
Vysvětlení: robustní z-skóre příznaků vůči typickému profilu letiště.
Výstupy (PNG + CSV) se ukládají do outputs/task3/.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "flight_delays_2025_01.csv.gz"
OUT = ROOT / "outputs" / "task3"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
MIN_FLIGHTS = 20  # minimální počet odletů za den, aby byl profil spolehlivý
CONTAMINATION = 0.01

# ---------------------------------------------------------------- data
df = pd.read_csv(DATA)
df["FlightDate"] = pd.to_datetime(df["FlightDate"])
valid = df[(df["Cancelled"] == 0) & (df["Diverted"] == 0)]

# ------------------------------------------------- profily letiště × den
g_all = df.groupby(["Origin", "FlightDate"])
prof = pd.DataFrame(
    {
        "n_flights": g_all.size(),
        "cancel_rate": g_all["Cancelled"].mean(),
        "divert_rate": g_all["Diverted"].mean(),
    }
)
g = valid.groupby(["Origin", "FlightDate"])
prof["arr_del15_rate"] = g["ArrDel15"].mean()
prof["dep_del15_rate"] = g["DepDel15"].mean()
prof["mean_dep_delay"] = g["DepDelay"].mean()
prof["p90_dep_delay"] = g["DepDelay"].quantile(0.9)
prof["mean_arr_delay"] = g["ArrDelay"].mean()
prof["mean_taxi_out"] = g["TaxiOut"].mean()
# podíly příčin na celkových minutách zpoždění (jen u ArrDel15 == 1)
causes = ["CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]
tot = valid.groupby(["Origin", "FlightDate"])[causes].sum()
tot_sum = tot.sum(axis=1).replace(0, np.nan)
for c in causes:
    prof[f"share_{c}"] = (tot[c] / tot_sum).fillna(0)
prof["mean_delay_minutes_per_flight"] = tot.sum(axis=1) / g.size().reindex(tot.index)
prof = prof.reset_index()
prof = prof[prof["n_flights"] >= MIN_FLIGHTS].dropna().reset_index(drop=True)
prof["day"] = prof["FlightDate"].dt.day
prof["dow"] = prof["FlightDate"].dt.day_of_week
print(f"Profilů letiště×den: {len(prof):,} ({prof['Origin'].nunique()} letišť)")

feat_cols = [c for c in prof.columns if c not in ("Origin", "FlightDate", "n_flights", "day", "dow")]
feat_cols = ["n_flights"] + feat_cols
Xs = RobustScaler().fit_transform(prof[feat_cols])

# ---------------------------------------------------------------- model
iso = IsolationForest(
    n_estimators=500, contamination=CONTAMINATION, random_state=SEED, n_jobs=-1
).fit(Xs)
prof["anomaly_score"] = -iso.score_samples(Xs)  # vyšší = anomálnější
prof["is_anomaly"] = iso.predict(Xs) == -1
print(f"Anomálií: {prof['is_anomaly'].sum()} z {len(prof)}")

# ------------------------------- vysvětlení: odchylka od typického profilu letiště
# z-skóre vůči mediánu/MAD téhož letiště (fallback na globální, pokud MAD = 0)
grp = prof.groupby("Origin")[feat_cols]
med = grp.transform("median")
mad = grp.transform(lambda s: (s - s.median()).abs().median()) * 1.4826
glob_mad = (prof[feat_cols] - prof[feat_cols].median()).abs().median() * 1.4826
mad = mad.where(mad > 0, other=glob_mad, axis=1).replace(0, np.nan)
z = ((prof[feat_cols] - med) / mad).fillna(0)


def top_reasons(i: int, k: int = 3) -> str:
    row = z.loc[i]
    idx = row.abs().sort_values(ascending=False).index[:k]
    return "; ".join(f"{c} ({row[c]:+.1f}σ)" for c in idx)


anom = prof[prof["is_anomaly"]].sort_values("anomaly_score", ascending=False).copy()
anom["reasons"] = [top_reasons(i) for i in anom.index]
anom_cols = ["Origin", "FlightDate", "anomaly_score", "n_flights", "cancel_rate",
             "arr_del15_rate", "mean_dep_delay", "mean_taxi_out", "reasons"]
anom[anom_cols].to_csv(OUT / "anomalies.csv", index=False)
prof.to_csv(OUT / "profiles_scored.csv", index=False)
print(anom[["Origin", "FlightDate", "anomaly_score", "reasons"]].head(15).to_string(index=False))

# ------------------------------------------------------ agregace po dnech
daily = prof.groupby("FlightDate").agg(
    n_anomalies=("is_anomaly", "sum"), mean_score=("anomaly_score", "mean")
)
daily.to_csv(OUT / "anomalies_by_day.csv")
by_airport = (
    prof.groupby("Origin").agg(n_anomalies=("is_anomaly", "sum"), n_days=("is_anomaly", "size"))
    .sort_values("n_anomalies", ascending=False)
)
by_airport.head(20).to_csv(OUT / "anomalies_by_airport.csv")

# ---------------------------------------------------------------- grafy
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(daily.index, daily["n_anomalies"], color="#c0392b")
ax.set_title("Počet anomálních profilů letiště × den")
ax.set_ylabel("počet anomálií")
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(OUT / "anomalies_by_day.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 6))
norm = prof[~prof["is_anomaly"]]
ax.scatter(norm["mean_dep_delay"], norm["cancel_rate"], s=6, alpha=0.3, c="#7f8c8d", label="normální")
ax.scatter(anom["mean_dep_delay"], anom["cancel_rate"], s=25, c="#c0392b", label="anomálie")
for _, r in anom.head(8).iterrows():
    ax.annotate(f"{r['Origin']} {r['FlightDate']:%d.%m.}", (r["mean_dep_delay"], r["cancel_rate"]), fontsize=7)
ax.set_xlabel("průměrné zpoždění odletu [min]")
ax.set_ylabel("podíl zrušených letů")
ax.set_title("Isolation Forest: letiště × den")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "scatter_delay_cancel.png", dpi=150)
plt.close(fig)

top = anom.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh([f"{o} {d:%d.%m.}" for o, d in zip(top["Origin"], top["FlightDate"])],
        top["anomaly_score"], color="#c0392b")
ax.set_xlabel("anomaly score")
ax.set_title("Top 15 anomálií")
fig.tight_layout()
fig.savefig(OUT / "top_anomalies.png", dpi=150)
plt.close(fig)

# heatmapa: top 15 letišť podle počtu anomálií × dny
top_ap = by_airport.head(15).index
hm = prof[prof["Origin"].isin(top_ap)].pivot(index="Origin", columns="day", values="anomaly_score").reindex(top_ap)
fig, ax = plt.subplots(figsize=(11, 5))
im = ax.imshow(hm.values, aspect="auto", cmap="magma_r")
ax.set_yticks(range(len(hm.index)), hm.index)
ax.set_xticks(range(len(hm.columns)), hm.columns, fontsize=7)
ax.set_xlabel("den v lednu 2025")
ax.set_title("Anomaly score – letiště s nejvíce anomáliemi")
fig.colorbar(im, label="anomaly score")
fig.tight_layout()
fig.savefig(OUT / "heatmap_airport_day.png", dpi=150)
plt.close(fig)
print(f"Hotovo, výstupy v {OUT}")
