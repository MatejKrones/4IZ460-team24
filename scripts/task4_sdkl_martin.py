# ============================================================
# ÚLOHA 4 – SDKL-like analýza
# Porovnání vztahu Airline x ArrDel15
# mezi krátkými a dlouhými lety
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


# 4. AIRLINE x DELAY ASSOCIATION BY FLIGHT LENGTH
# Pro Short a Long zvlášť vytvoříme kontingenční tabulku:
# Airline x ArrDel15
#
# Potom spočítáme:
# - Chi-square
# - p-value
# - Cramérův V
# - minimum expected frequency

results = []

for flight_length, group_df in df.groupby("FlightLength"):

    table = pd.crosstab(
        group_df["Airline"],
        group_df["ArrDel15"]
    )

    chi2, p, dof, expected = chi2_contingency(table)

    n = table.to_numpy().sum()

    min_dim = min(table.shape) - 1

    cramers_v = np.sqrt(
        chi2 / (n * min_dim)
    )

    results.append({
        "FlightLength": flight_length,
        "Flights": n,
        "ChiSquare": chi2,
        "p_value": p,
        "Cramers_V": cramers_v,
        "MinExpected": expected.min()
    })


results_df = pd.DataFrame(results)

print("\nAIRLINE x DELAY ASSOCIATION BY FLIGHT LENGTH")
print(
    results_df
    .round(4)
    .to_string(index=False)
)


# 5. DELAY RATE BY AIRLINE AND FLIGHT LENGTH
summary = (
    df.groupby(["FlightLength", "Airline"])
    .agg(
        Flights=("ID", "count"),
        Delayed=(
            "ArrDel15",
            lambda x: (x == "Yes").sum()
        )
    )
    .reset_index()
)

summary["DelayRate"] = (
    summary["Delayed"]
    / summary["Flights"]
    * 100
)


# Vytvoření přehledu Short vs Long pro každou aerolinku
flight_counts = summary.pivot(
    index="Airline",
    columns="FlightLength",
    values="Flights"
)

delay_rates = summary.pivot(
    index="Airline",
    columns="FlightLength",
    values="DelayRate"
)

comparison = pd.DataFrame({
    "ShortFlights": flight_counts["Short"],
    "LongFlights": flight_counts["Long"],
    "ShortRate": delay_rates["Short"],
    "LongRate": delay_rates["Long"]
})

comparison["Difference_pp"] = (
    comparison["LongRate"]
    - comparison["ShortRate"]
)

comparison = comparison.sort_values(
    "Difference_pp",
    ascending=False
)

print("\nDELAY RATE BY AIRLINE AND FLIGHT LENGTH")
print(
    comparison
    .round(2)
    .to_string()
)


# 6. HUMAN CHECK
# Pro hlavní interpretaci používáme pouze aerolinky,
# které mají alespoň 500 letů v obou skupinách.

MIN_FLIGHTS = 500

robust = comparison[
    (comparison["ShortFlights"] >= MIN_FLIGHTS)
    & (comparison["LongFlights"] >= MIN_FLIGHTS)
].copy()

print("\nROBUST AIRLINES (>= 500 flights in both groups)")
print(
    robust
    .round(2)
    .to_string()
)


# Kontrola OH – největší rozdíl, ale malá Long skupina
print("\nOH CHECK")
print(
    comparison.loc["OH"]
    .round(2)
    .to_string()
)


# Hlavní robustní příklady
print("\nHA CHECK")
print(
    comparison.loc["HA"]
    .round(2)
    .to_string()
)

print("\nG4 CHECK")
print(
    comparison.loc["G4"]
    .round(2)
    .to_string()
)


# 7. INTERPRETATION SUPPORT
# HA = robustní příklad vyššího delay rate u dlouhých letů
# G4 = robustní příklad vyššího delay rate u krátkých letů

ha_difference = comparison.loc["HA", "Difference_pp"]
g4_difference = comparison.loc["G4", "Difference_pp"]

print("\nMAIN EXAMPLES")
print(
    f"HA difference (Long - Short): "
    f"{ha_difference:.2f} pp"
)

print(
    f"G4 difference (Long - Short): "
    f"{g4_difference:.2f} pp"
)


# 8. VISUALIZATION
# Graf zobrazuje všechny aerolinky.
# Při interpretaci ale používáme pouze robustní příklady.

plot_df = comparison.dropna().copy()

x = np.arange(len(plot_df))

plt.figure(figsize=(10, 6))

plt.bar(
    x - 0.2,
    plot_df["ShortRate"],
    width=0.4,
    label="Short"
)

plt.bar(
    x + 0.2,
    plot_df["LongRate"],
    width=0.4,
    label="Long"
)

plt.xticks(
    x,
    plot_df.index,
    rotation=45
)

plt.title(
    "Delay rate by airline – short vs long flights"
)
plt.xlabel("Airline")
plt.ylabel("Delayed flights (%)")
plt.legend()

plt.tight_layout()
plt.show()