# ============================================================
# ÚLOHA 2 – Action-rule-like analýza
# Hledání změn času odletu spojených s nižším
# podílem zpožděných letů
# ============================================================

# 1. IMPORTS
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


# 2. LOAD DATA
df = pd.read_csv("flights_4ft (1).csv")

print("Dataset shape:", df.shape)


# 3. DATA PREPARATION
# Pro každou kombinaci:
# Airline + OriginRegion + SchedDepTime
# spočítáme počet letů, počet zpožděných a delay rate.

summary = (
    df.groupby(
        ["Airline", "OriginRegion", "SchedDepTime"]
    )
    .agg(
        Flights=("ID", "count"),
        Delayed=("ArrDel15", lambda x: (x == "Yes").sum())
    )

)

summary["DelayRate"] = (
    summary["Delayed"] / summary["Flights"] * 100
)

print("\nTOTAL SEGMENTS:", len(summary))


# 4. MINIMUM SEGMENT SIZE
# Malé skupiny mohou mít nestabilní delay rate.
# Proto používáme pouze časové segmenty s alespoň 500 lety.

MIN_FLIGHTS = 500

valid = summary[
    summary["Flights"] >= MIN_FLIGHTS
].copy()

print("VALID SEGMENTS (>= 500 flights):", len(valid))


# 5. SEARCH FOR TIME CHANGES
# Pro stejnou kombinaci Airline + OriginRegion
# porovnáváme všechny možné dvojice časových bloků.
#
# Pokud má cílový čas nižší delay rate,
# zaznamenáme rozdíl v procentních bodech.

time_order = [
    "1_Night_0_6",
    "2_Morning_6_10",
    "3_Midday_10_14",
    "4_Afternoon_14_18",
    "5_Evening_18_24"
]

results = []

for (airline, region), group in valid.groupby(
    ["Airline", "OriginRegion"]
):
    group = group.set_index("SchedDepTime")

    for from_time in time_order:
        for to_time in time_order:

            if from_time == to_time:
                continue

            if (
                from_time in group.index
                and to_time in group.index
            ):
                from_rate = group.loc[
                    from_time, "DelayRate"
                ]

                to_rate = group.loc[
                    to_time, "DelayRate"
                ]

                reduction = from_rate - to_rate

                if reduction > 0:
                    results.append({
                        "Airline": airline,
                        "OriginRegion": region,
                        "FromTime": from_time,
                        "ToTime": to_time,
                        "FromRate": from_rate,
                        "ToRate": to_rate,
                        "Reduction_pp": reduction,
                        "FromFlights": group.loc[
                            from_time, "Flights"
                        ],
                        "ToFlights": group.loc[
                            to_time, "Flights"
                        ]
                    })


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "Reduction_pp",
    ascending=False
)

print("\nTOP 20 TIME CHANGES")
print(
    results_df[
        [
            "Airline",
            "OriginRegion",
            "FromTime",
            "ToTime",
            "FromRate",
            "ToRate",
            "Reduction_pp",
            "FromFlights",
            "ToFlights"
        ]
    ]
    .head(20)
    .round(2)
    .to_string(index=False)
)


# 6. SELECTED RESULT – G4 + SOUTH
# Největší nalezený rozdíl:
# Evening 18–24 -> Morning 6–10

selected = summary[
    (summary["Airline"] == "G4")
    & (summary["OriginRegion"] == "South")
].copy()

selected = selected.sort_values("SchedDepTime")

print("\nG4 + SOUTH DETAIL")
print(
    selected[
        [
            "SchedDepTime",
            "Flights",
            "Delayed",
            "DelayRate"
        ]
    ]
    .round(2)
    .to_string(index=False)
)


# 7. STATISTICAL VERIFICATION
# Porovnání Morning 6–10 vs Evening 18–24

morning = selected[
    selected["SchedDepTime"] == "2_Morning_6_10"
].iloc[0]

evening = selected[
    selected["SchedDepTime"] == "5_Evening_18_24"
].iloc[0]

table = [
    [
        int(morning["Delayed"]),
        int(morning["Flights"] - morning["Delayed"])
    ],
    [
        int(evening["Delayed"]),
        int(evening["Flights"] - evening["Delayed"])
    ]
]

chi2, p, dof, expected = chi2_contingency(table)

difference_pp = (
    evening["DelayRate"] - morning["DelayRate"]
)

print("\nMORNING VS EVENING")
print(
    f"Morning delay rate: "
    f"{morning['DelayRate']:.2f}%"
)
print(
    f"Evening delay rate: "
    f"{evening['DelayRate']:.2f}%"
)
print(
    f"Difference: {difference_pp:.2f} pp"
)
print(f"Chi-square: {chi2:.2f}")
print(f"p-value: {p:.6g}")


# 8. HUMAN CHECK
# Kontrola, že hlavní výsledek není založený
# na příliš malých skupinách.

print("\nHUMAN CHECK")

print(
    f"Morning flights: {int(morning['Flights'])}"
)
print(
    f"Evening flights: {int(evening['Flights'])}"
)
print(
    f"Minimum required flights: {MIN_FLIGHTS}"
)

print(
    "Morning segment valid:",
    morning["Flights"] >= MIN_FLIGHTS
)

print(
    "Evening segment valid:",
    evening["Flights"] >= MIN_FLIGHTS
)

print(
    f"Minimum expected frequency: "
    f"{expected.min():.2f}"
)

# Noční segment je malý a není použit
# jako hlavní interpretační příklad.
night = selected[
    selected["SchedDepTime"] == "1_Night_0_6"
]

if not night.empty:
    print(
        f"Night flights (not interpreted): "
        f"{int(night.iloc[0]['Flights'])}"
    )


# 9. VISUALIZATION
plt.figure(figsize=(10, 6))

plt.bar(
    selected["SchedDepTime"],
    selected["DelayRate"]
)

plt.title(
    "Delay rate by scheduled departure time – G4, South"
)
plt.xlabel("Scheduled departure time")
plt.ylabel("Delayed flights (%)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()