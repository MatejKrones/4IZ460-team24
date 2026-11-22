# CLAUDE.md – semestrální projekt 4IZ457 (VŠE)

## Zadání a pravidla
- Kurz 4IZ457, dataset **US Flight Delays 2025-01** (BTS On-Time Performance, leden 2025, Kaggle). Týmová práce, tým řeší 8 úloh v Bloku 2, uživatel 4.
- **Blok 1 je hotový** (analýza kontingenční tabulky ve Wordu, 4ft-Miner, SD4ft-Miner, CF-Miner s interpretacemi).
- **Blok 2** musí být udělán explicitně za pomoci AI. Uživatel dokumentuje použité prompty, takže drž kroky přehledné a reprodukovatelné.
- **Nepoužívat** (dělá kolega): SDCF-like (rozložení času odletu zpožděných letů krátké vs. dlouhé), Action-rule-like (změna časového bloku), KL-like (závislost Airline/OriginRegion a ArrDel15), SDKL-like (Airline × ArrDel15 krátké vs. dlouhé lety).
- Skripty se budují postupně, jeden po druhém. Po každém skriptu ho spusť a ověř výstupy.
- **Výstupy:** všechny skripty musí ukládat výsledky (grafy, CSV, metriky) do složky `outputs/` (podsložka per úloha, např. `outputs/task1/`). Složka je v `.gitignore`.

## Úlohy uživatele v Bloku 2
1. **XAI: SHAP** – LightGBM predikce ArrDel15, globální a lokální vysvětlení.
2. **Surrogate pravidla** – rozhodovací strom, RuleFit nebo skope-rules nad predikcemi modelu, fidelita, srovnání s GUHA pravidly z Bloku 1.
3. **Detekce anomálií** – Isolation Forest na profilech letiště × den.
4. **Hledání podskupin s chybami modelu** (slice discovery) – pysubgroup nebo beam search, kalibrace.

## Data
- Originál: `flight_delays_2025_01.csv.gz` (18 MB komprimovaně, 107 MB rozbaleně, 539 747 řádků; čti přímo přes `pd.read_csv("flight_delays_2025_01.csv.gz")`). Pro Blok 2 používat **originál**, ne diskretizované `flights_4ft.csv`.
- Sloupce: Year, Quarter, Month, DayofMonth, DayOfWeek, FlightDate, Reporting_Airline, Flight_Number_Reporting_Airline, Origin, OriginCityName, OriginState, Dest, DestCityName, DestState, CRSDepTime, DepTime, DepDelay, DepDelayMinutes, DepDel15, TaxiOut, WheelsOff, WheelsOn, TaxiIn, CRSArrTime, ArrTime, ArrDelay, ArrDelayMinutes, ArrDel15, Cancelled, CancellationCode, Diverted, CRSElapsedTime, ActualElapsedTime, AirTime, Distance, CarrierDelay, WeatherDelay, NASDelay, SecurityDelay, LateAircraftDelay.
- CRSDepTime a CRSArrTime jsou ve formátu hhmm.
- V originále jsou u zrušených a přesměrovaných letů skutečné NaN. Soubor `_filled` obsahuje sentinely (-999, -1), proto ho nepoužívat.
- **Platné lety:** `Cancelled == 0 & Diverted == 0` → 522 269 řádků. Podíl `ArrDel15 = 1` je 18,8 % (98 130 letů).
- Sloupce s příčinami zpoždění (Carrier, Weather, NAS, Security, LateAircraft) existují jen pro `ArrDel15 == 1` a jejich součet je ArrDelay.
- Pondělí 6. 1. 2025 je jednodenní artefakt, který táhne silná pravidla typu „Mon & Days1_10“ z Bloku 1.
