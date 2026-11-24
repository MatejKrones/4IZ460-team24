# ============================================================
# ÚLOHA 1 – SDCF-like analýza
# Porovnání rozložení času odletu u krátkých a dlouhých
# zpožděných letů
# ============================================================

# 1. IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


# 2. LOAD DATA
df = pd.read_csv("flights_4ft (1).csv")

print("Dataset shape:", df.shape)


# 3. DATA PREPARATION
# Rozdělení letů podle vzdálenosti:
# Short = méně než 1000 mil
# Long = 1000 mil a více

distance_mapping = {
    "1_Under250": "Short",
    "2_250_499": "Short",
    "3_500_999": "Short",
    "4_1000_1999": "Long",
    "5_2000plus": "Long"
}

df["FlightLength"] = df["Distance"].map(distance_mapping)

print("\nFLIGHT LENGTH COUNTS")
print(df["FlightLength"].value_counts())


# Pro samotnou analýzu necháváme pouze zpožděné lety
# ArrDel15 = Yes znamená zpoždění příletu alespoň 15 minut
delayed = df[df["ArrDel15"] == "Yes"].copy()

print("\nDELAYED FLIGHTS:", len(delayed))


# 4. MAIN ANALYSIS
# Porovnání procentuálního rozložení času odletu
# mezi krátkými a dlouhými zpožděnými lety

short_delayed = delayed[delayed["FlightLength"] == "Short"]
long_delayed = delayed[delayed["FlightLength"] == "Long"]

short_hist = (
    short_delayed["SchedDepTime"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

long_hist = (
    long_delayed["SchedDepTime"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

comparison = pd.DataFrame({
    "Short delayed (%)": short_hist,
    "Long delayed (%)": long_hist
})

comparison["Difference (pp)"] = (
    comparison["Long delayed (%)"]
    - comparison["Short delayed (%)"]
)

print("\nDISTRIBUTION OF DELAYED FLIGHTS")
print(comparison.round(2))


# 5. STATISTICAL VERIFICATION
# Kontingenční tabulka:
# FlightLength x SchedDepTime

contingency = pd.crosstab(
    delayed["FlightLength"],
    delayed["SchedDepTime"]
)

print("\nCONTINGENCY TABLE")
print(contingency)

chi2, p, dof, expected = chi2_contingency(contingency)

n = contingency.to_numpy().sum()

# Cramérův V = míra síly vztahu
min_dim = min(contingency.shape) - 1
cramers_v = np.sqrt(
    chi2 / (n * min_dim)
)

print("\nSTATISTICAL RESULTS")
print(f"Chi-square: {chi2:.2f}")
print(f"p-value: {p:.6g}")
print(f"Cramer's V: {cramers_v:.4f}")


# 6. HUMAN CHECK
# Kontrola, že rozdělení FlightLength pokrývá celý dataset

short_count = (df["FlightLength"] == "Short").sum()
long_count = (df["FlightLength"] == "Long").sum()

print("\nHUMAN CHECK")
print(f"Short flights: {short_count}")
print(f"Long flights: {long_count}")
print(f"Total classified: {short_count + long_count}")
print(f"Original dataset: {len(df)}")

# Kontrola, že procenta v každém histogramu dávají 100 %
print(
    f"Short histogram total: {short_hist.sum():.2f}%"
)
print(
    f"Long histogram total: {long_hist.sum():.2f}%"
)

# Kontrola očekávaných četností pro chi-square test
print(
    f"Minimum expected frequency: {expected.min():.2f}"
)


# 7. VISUALIZATION
comparison[
    ["Short delayed (%)", "Long delayed (%)"]
].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title(
    "Distribution of delayed flights by scheduled departure time"
)
plt.xlabel("Scheduled departure time")
plt.ylabel("Share of delayed flights (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()