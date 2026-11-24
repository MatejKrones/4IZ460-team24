# ============================================================
# ÚLOHA 3 – KL-like analýza
# Závislost mezi aerolinkou a zpožděním podle regionu
# ============================================================

# 1. IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


# 2. LOAD DATA
df = pd.read_csv("flights_4ft (1).csv")

print("Dataset shape:", df.shape)


# 3. REGION OVERVIEW
# Nejprve kontrolujeme velikost jednotlivých regionů.

region_counts = df["OriginRegion"].value_counts()

print("\nREGION COUNTS")
print(region_counts)


# 4. AIRLINE x DELAY ASSOCIATION BY REGION
# Pro každý region vytvoříme kontingenční tabulku:
# Airline x ArrDel15
#
# Potom spočítáme:
# - Chi-square test
# - p-value
# - Cramérův V jako míru síly vztahu

results = []

for region, region_df in df.groupby("OriginRegion"):

    table = pd.crosstab(
        region_df["Airline"],
        region_df["ArrDel15"]
    )

    # Test vyžaduje obě kategorie ArrDel15
    if (
        "Yes" not in table.columns
        or "No" not in table.columns
    ):
        continue

    chi2, p, dof, expected = chi2_contingency(table)

    n = table.to_numpy().sum()

    min_dim = min(table.shape) - 1

    cramers_v = np.sqrt(
        chi2 / (n * min_dim)
    )

    results.append({
        "OriginRegion": region,
        "Flights": n,
        "ChiSquare": chi2,
        "p_value": p,
        "Cramers_V": cramers_v,
        "MinExpected": expected.min()
    })


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "Cramers_V",
    ascending=False
)

print("\nAIRLINE x DELAY ASSOCIATION BY REGION")
print(
    results_df.round(4).to_string(index=False)
)


# 5. SELECTED REGION – NORTHEAST
# Territory má sice nejvyšší Cramérův V,
# ale je výrazně menší.
#
# Pro hlavní interpretaci proto používáme Northeast,
# který má vysoký Cramérův V a zároveň velký počet letů.

selected_region = "Northeast"

selected_region_df = df[
    df["OriginRegion"] == selected_region
].copy()


# 6. AIRLINE DELAY RATES IN NORTHEAST
airline_summary = (
    selected_region_df
    .groupby("Airline")
    .agg(
        Flights=("ID", "count"),
        Delayed=(
            "ArrDel15",
            lambda x: (x == "Yes").sum()
        )
    )
    .reset_index()
)

airline_summary["DelayRate"] = (
    airline_summary["Delayed"]
    / airline_summary["Flights"]
    * 100
)

airline_summary = airline_summary.sort_values(
    "DelayRate",
    ascending=False
)

print(f"\nAIRLINE DELAY RATES - {selected_region}")
print(
    airline_summary
    .round(2)
    .to_string(index=False)
)


# 7. HUMAN CHECK
# Kontrola velikosti vybraného regionu,
# očekávaných četností a malých aerolinek.

selected_table = pd.crosstab(
    selected_region_df["Airline"],
    selected_region_df["ArrDel15"]
)

selected_chi2, selected_p, selected_dof, selected_expected = (
    chi2_contingency(selected_table)
)

print("\nHUMAN CHECK")

print(
    f"{selected_region} flights: "
    f"{len(selected_region_df)}"
)

print(
    f"Minimum expected frequency "
    f"in {selected_region}: "
    f"{selected_expected.min():.2f}"
)

small_airlines = airline_summary[
    airline_summary["Flights"] < 500
]

print("\nAIRLINES WITH < 500 FLIGHTS")
print(
    small_airlines[
        ["Airline", "Flights", "DelayRate"]
    ]
    .round(2)
    .to_string(index=False)
)


# Territory check – vysvětlení, proč ho nevolíme
territory_df = df[
    df["OriginRegion"] == "Territory"
]

territory_table = pd.crosstab(
    territory_df["Airline"],
    territory_df["ArrDel15"]
)

_, _, _, territory_expected = chi2_contingency(
    territory_table
)

print(
    f"\nTerritory flights: "
    f"{len(territory_df)}"
)

print(
    f"Territory minimum expected frequency: "
    f"{territory_expected.min():.2f}"
)


# 8. VISUALIZATION
# Do grafu dáváme jen aerolinky
# s alespoň 500 lety v Northeast.

MIN_FLIGHTS = 500

plot_df = airline_summary[
    airline_summary["Flights"] >= MIN_FLIGHTS
].copy()

plt.figure(figsize=(10, 6))

plt.bar(
    plot_df["Airline"],
    plot_df["DelayRate"]
)

plt.title(
    "Delay rate by airline – Northeast"
)
plt.xlabel("Airline")
plt.ylabel("Delayed flights (%)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()