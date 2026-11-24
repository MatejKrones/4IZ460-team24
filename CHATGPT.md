# CHATGPT.md – semestrální projekt 4IZ457 (VŠE)

## Účel souboru

Tento soubor dokumentuje způsob použití ChatGPT při zpracování části semestrálního projektu v kurzu **4IZ457 – Pokročilé přístupy k dobývání znalostí z databází** na VŠE.

Cílem není pouze uvést, že byla použita AI, ale popsat dostatečný kontext, analytické otázky, metodiku, strukturu dat a kontrolní pravidla tak, aby jiný uživatel mohl s pomocí ChatGPT nebo obdobného AI nástroje vytvořit analyticky velmi podobné řešení.

Výsledný kód byl vytvořen iterativně:
1. byla definována analytická otázka,
2. ChatGPT navrhl postup,
3. byl vytvořen Python skript,
4. skript byl spuštěn v PyCharmu,
5. výsledky byly ručně zkontrolovány,
6. podle výsledků byl kód případně upraven,
7. finální interpretace byla zapsána do podkladu projektu.

AI tedy nebyla použita pouze k jednorázovému vygenerování kódu, ale jako pomocník při návrhu, kontrole, interpretaci a zpřehlednění jednotlivých analytických kroků.

---

# 1. Kontext projektu

Projekt pracuje s daty o letech v USA z datasetu:

**US Flight Delays 2025-01**

Zdroj dat vychází z BTS On-Time Performance za leden 2025.

Týmová práce obsahuje dvě hlavní části:

- **Blok 1** – analýzy pomocí metod GUHA / CleverMiner,
- **Blok 2** – pokročilejší analýzy realizované s využitím Pythonu a AI.

Tento dokument se vztahuje pouze ke čtyřem Python úlohám zpracovaným v této části projektu.

---

# 2. Použitý dataset

Používaný soubor:

```text
flights_4ft (1).csv
```

Dataset má:

```text
522 269 řádků
15 atributů
```

Použité sloupce:

```text
ID
Airline
DayOfWeek
MonthPart
SchedDepTime
SchedArrTime
OriginSize
DestSize
OriginRegion
DestRegion
Distance
SchedDuration
DepDelay
ArrDelay
ArrDel15
```

Hlavním sledovaným výsledkovým atributem je:

```text
ArrDel15
```

Význam:

```text
Yes = let přiletěl se zpožděním alespoň 15 minut
No  = let tuto hranici zpoždění nesplnil
```

Počet zpožděných letů:

```text
98 130
```

Celkový podíl zpožděných letů je přibližně:

```text
18,8 %
```

---

# 3. Kategorie použité v datech

## Distance

Atribut `Distance` je kategorizovaný:

```text
1_Under250
2_250_499
3_500_999
4_1000_1999
5_2000plus
```

Pro některé analýzy byly kategorie sloučeny:

```text
Short:
1_Under250
2_250_499
3_500_999

Long:
4_1000_1999
5_2000plus
```

Tedy:

```text
Short = vzdálenost pod 1 000 mil
Long  = vzdálenost 1 000 mil a více
```

Počty:

```text
Short = 361 870
Long  = 160 399
```

Součet:

```text
361 870 + 160 399 = 522 269
```

Tím bylo ověřeno, že rozdělení pokrývá celý dataset.

---

## SchedDepTime

Plánovaný čas odletu je rozdělen do pěti bloků:

```text
1_Night_0_6
2_Morning_6_10
3_Midday_10_14
4_Afternoon_14_18
5_Evening_18_24
```

---

# 4. Použité Python knihovny

Základní analýzy využívají:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
```

Hlavní role knihoven:

```text
pandas      – práce s tabulkovými daty
numpy       – numerické operace
matplotlib  – tvorba grafů
scipy       – statistické testování
```

---

# 5. Obecná pravidla pro tvorbu kódu

Při generování nebo úpravách skriptů zachovávat jednoduchou a čitelnou strukturu.

Preferovaná struktura:

```python
# 1. IMPORTS
# 2. LOAD DATA
# 3. DATA PREPARATION
# 4. MAIN ANALYSIS
# 5. STATISTICAL VERIFICATION
# 6. HUMAN CHECK
# 7. VISUALIZATION
```

Kód má být:

- čitelný,
- reprodukovatelný,
- jednoduše vysvětlitelný při obhajobě,
- bez zbytečně komplikovaných funkcí a abstrakcí,
- doplněný komentáři vysvětlujícími analytickou logiku.

Pokud se upravuje již existující skript, nesmí se bez důvodu měnit analytická otázka ani hlavní metodika.

Po každé významnější změně je nutné skript znovu spustit a porovnat hlavní výsledky s původní verzí.

---

# 6. Statistická pravidla

Ve více úlohách je použit Pearsonův chí-kvadrát test.

Python:

```python
chi2, p, dof, expected = chi2_contingency(table)
```

Chí-kvadrát test odpovídá především na otázku:

```text
Existuje statisticky významná závislost mezi dvěma kategoriálními proměnnými?
```

Samotná p-hodnota ale nestačí pro posouzení praktické významnosti.

Dataset obsahuje více než 500 tisíc letů, proto mohou být i velmi malé rozdíly statisticky významné.

Z tohoto důvodu se používá také **Cramérův V**.

Výpočet:

```python
n = table.to_numpy().sum()
min_dim = min(table.shape) - 1

cramers_v = np.sqrt(
    chi2 / (n * min_dim)
)
```

Interpretace:

```text
hodnota blízko 0 = velmi slabý vztah
vyšší hodnota    = silnější vztah
```

Cramérův V se používá k posouzení síly vztahu, zatímco p-hodnota k posouzení statistické významnosti.

---

# 7. Human check

Každá úloha musí obsahovat ruční nebo programovou kontrolu výsledků.

Kontrolovat zejména:

1. počet analyzovaných pozorování,
2. zda vytvořené skupiny pokrývají očekávaný počet řádků,
3. zda procentní distribuce dávají přibližně 100 %,
4. velikost jednotlivých podskupin,
5. minimální očekávanou četnost u chí-kvadrát testu,
6. zda extrémní výsledek není způsoben malým vzorkem,
7. zda p-hodnota není zaměňována za sílu vztahu,
8. zda interpretace odpovídá skutečně vypočteným hodnotám.

Výsledky z pozorovaných dat jsou interpretovány jako **asociace**, nikoliv jako důkaz kauzálního vztahu.

---

# 8. Úloha 1 – SDCF-like analýza

Soubor:

```text
task1_sdcf.py
```

## Analytická otázka

```text
Liší se rozložení plánovaného času odletu
u zpožděných krátkých a dlouhých letů?
```

## Logika

1. Načíst dataset.
2. Převést `Distance` na dvě kategorie `Short` a `Long`.
3. Vybrat pouze:

```text
ArrDel15 = Yes
```

4. Pro Short a Long samostatně vypočítat procentní distribuci:

```text
SchedDepTime
```

5. Porovnat obě distribuce.
6. Vytvořit kontingenční tabulku:

```text
FlightLength × SchedDepTime
```

7. Použít Pearsonův chí-kvadrát test.
8. Spočítat Cramérův V.
9. Zkontrolovat velikosti skupin a součty procent.
10. Vytvořit graf distribuce.

## Důležitá metodická poznámka

Analýza neporovnává pravděpodobnost zpoždění mezi Short a Long.

Analyzuje pouze:

```text
jak jsou plánované časy odletu rozloženy mezi lety,
které již byly označeny jako zpožděné
```

## Očekávané výsledky

Distribuce Short:

```text
Night      2,03 %
Morning   20,04 %
Midday    24,08 %
Afternoon 28,29 %
Evening   25,56 %
```

Distribuce Long:

```text
Night      1,44 %
Morning   22,92 %
Midday    24,66 %
Afternoon 27,16 %
Evening   23,82 %
```

Největší rozdíl:

```text
Morning:
Long - Short = +2,87 procentního bodu
```

Statistika:

```text
χ² = 160,16
p < 0,001
Cramérův V = 0,0404
```

Minimální očekávaná četnost:

```text
559,98
```

## Interpretace

Rozdělení se statisticky významně liší, ale:

```text
Cramérův V = 0,0404
```

ukazuje na velmi slabý praktický vztah.

Hlavní závěr:

```text
Rozdíl existuje, ale jeho praktická velikost je velmi malá.
```

---

# 9. Úloha 2 – Action-rule-like analýza

Soubor:

```text
task2_action_rules.py
```

## Analytická otázka

```text
U kterých kombinací aerolinky a regionu
je změna plánovaného času odletu spojena
s největším poklesem podílu zpožděných letů?
```

## Role atributů

Stabilní atributy:

```text
Airline
OriginRegion
```

Potenciálně měnitelný atribut:

```text
SchedDepTime
```

Výsledkový atribut:

```text
ArrDel15
```

## Logika

1. Načíst dataset.
2. Pomocí `groupby` vytvořit segmenty:

```text
Airline + OriginRegion + SchedDepTime
```

3. Pro každý segment vypočítat:

```text
Flights
Delayed
DelayRate
```

kde:

```python
DelayRate = Delayed / Flights * 100
```

4. Do dalšího hledání zahrnout pouze segmenty s minimálně:

```text
500 lety
```

5. Pro každou kombinaci:

```text
Airline + OriginRegion
```

porovnat jednotlivé časové bloky mezi sebou.

6. Hledat změny:

```text
vyšší DelayRate → nižší DelayRate
```

7. Seřadit výsledky podle velikosti rozdílu.
8. Nejlepší nalezený výsledek ověřit pomocí chí-kvadrát testu.
9. Zkontrolovat skutečné počty letů.
10. Zobrazit průběh DelayRate během dne.

## Očekávaný hlavní výsledek

Segment:

```text
Airline = G4
OriginRegion = South
```

Morning:

```text
1 211 letů
128 zpožděných
DelayRate = 10,57 %
```

Evening:

```text
765 letů
270 zpožděných
DelayRate = 35,29 %
```

Rozdíl:

```text
35,29 - 10,57 = 24,72 procentního bodu
```

Statistika:

```text
χ² = 176,64
p < 0,001
```

Minimální očekávaná četnost při hlavním testu:

```text
154,08
```

## Human check

Noční segment:

```text
18 letů
```

Je příliš malý a nesmí být použit jako hlavní interpretační výsledek.

## Interpretace

Správná interpretace:

```text
U G4 + South je ranní odlet v pozorovaných datech
spojen s výrazně nižším podílem zpožděných letů
než večerní odlet.
```

Nesprávná interpretace:

```text
Přesun letu z večera na ráno způsobí pokles zpoždění o 24,72 p. b.
```

Analýza je **action-rule-like**, nikoliv kauzální experiment.

---

# 10. Úloha 3 – KL-like analýza

Soubor:

```text
task3_kl.py
```

## Analytická otázka

```text
Ve kterých regionech je nejsilnější vztah
mezi aerolinkou a výskytem zpoždění příletu?
```

## Logika

1. Načíst dataset.
2. Rozdělit data podle:

```text
OriginRegion
```

3. V každém regionu vytvořit kontingenční tabulku:

```text
Airline × ArrDel15
```

4. Pro každý region vypočítat:

```text
Chi-square
p-value
Cramérův V
minimum expected frequency
```

5. Seřadit regiony podle Cramérova V.
6. Vybrat metodicky vhodný region pro detailní interpretaci.
7. Spočítat DelayRate jednotlivých aerolinek.
8. Zkontrolovat malé aerolinky.
9. Vytvořit graf.

## Očekávané výsledky

```text
Territory:
Flights = 4 123
Cramérův V = 0,1514
MinExpected = 2,11

Northeast:
Flights = 59 506
Cramérův V = 0,1159
MinExpected = 10,05

Midwest:
Flights = 79 214
Cramérův V = 0,0995

South:
Flights = 208 980
Cramérův V = 0,0745

West:
Flights = 170 446
Cramérův V = 0,0538
```

## Výběr regionu

Nejvyšší Cramérův V má:

```text
Territory
```

Territory ale není vhodný jako hlavní příklad kvůli:

```text
malému počtu letů
a velmi malé minimální očekávané četnosti
```

Proto pro hlavní interpretaci používat:

```text
Northeast
```

## Northeast – očekávané výsledky

Příklady:

```text
F9 = 34,86 %
OH = 28,20 %
B6 = 27,95 %
MQ = 27,07 %
OO = 25,70 %
YX = 16,06 %
AS = 14,18 %
```

HA:

```text
47 letů
14,89 %
```

Kvůli pouze 47 letům HA samostatně neinterpretovat.

## Interpretace

V Northeast existuje vztah mezi aerolinkou a výskytem zpoždění:

```text
Cramérův V = 0,1159
```

Vztah je ale stále pouze slabý až mírný.

Analýza neprokazuje, že samotná aerolinka způsobuje zpoždění.

---

# 11. Úloha 4 – SDKL-like analýza

Soubor:

```text
task4_sdkl.py
```

## Analytická otázka

```text
Liší se síla vztahu mezi aerolinkou a výskytem
zpoždění příletu u krátkých a dlouhých letů?
```

## Logika

1. Načíst dataset.
2. Převést `Distance` na:

```text
Short
Long
```

3. Pro každou skupinu vytvořit:

```text
Airline × ArrDel15
```

4. Samostatně vypočítat:

```text
χ²
p-value
Cramérův V
minimum expected frequency
```

5. Pro každou aerolinku spočítat:

```text
ShortFlights
LongFlights
ShortRate
LongRate
Difference_pp
```

kde:

```text
Difference_pp = LongRate - ShortRate
```

6. Pro hlavní interpretaci používat pouze aerolinky, které mají minimálně:

```text
500 Short letů
a zároveň
500 Long letů
```

7. Zkontrolovat extrémní výsledky.
8. Vytvořit Short vs. Long graf.

## Očekávané celkové výsledky

Short:

```text
Flights = 361 870
χ² = 2 432,12
Cramérův V = 0,082
MinExpected = 845,77
```

Long:

```text
Flights = 160 399
χ² = 763,93
Cramérův V = 0,069
MinExpected = 36,32
```

## Největší raw rozdíl

OH:

```text
Short:
18 959 letů
25,37 %

Long:
192 letů
36,98 %

Difference:
+11,61 p. b.
```

Tento příklad nepoužívat jako hlavní, protože Long skupina má pouze:

```text
192 letů
```

## Robustní příklad HA

```text
Short:
4 515 letů
11,52 %

Long:
2 025 letů
18,07 %

Difference:
+6,56 p. b.
```

## Opačný příklad G4

```text
Short:
5 111 letů
25,71 %

Long:
4 095 letů
21,22 %

Difference:
-4,49 p. b.
```

## Interpretace

Vztah:

```text
Airline × ArrDel15
```

je mírně silnější u Short letů:

```text
Short V = 0,082
Long V = 0,069
```

V obou případech ale jde pouze o slabý vztah.

Rozdíl mezi Short a Long navíc není u všech aerolinek stejný.

---

# 12. Klíčové pandas operace

Při reprodukci kódu lze používat standardní jednoduché operace.

## Načtení dat

```python
df = pd.read_csv("flights_4ft (1).csv")
```

## Převod kategorií

```python
df["FlightLength"] = df["Distance"].map({
    "1_Under250": "Short",
    "2_250_499": "Short",
    "3_500_999": "Short",
    "4_1000_1999": "Long",
    "5_2000plus": "Long"
})
```

## Filtrování

```python
df[df["ArrDel15"] == "Yes"]
```

## Groupby

Používat pro výpočty podle skupin:

```python
df.groupby(...)
```

## Agregace

Například:

```python
.agg(
    Flights=("ID", "count"),
    Delayed=("ArrDel15", lambda x: (x == "Yes").sum())
)
```

## Kontingenční tabulka

```python
pd.crosstab(
    df["Airline"],
    df["ArrDel15"]
)
```

## Pivot

Používat pro přehledné porovnání Short vs. Long:

```python
summary.pivot(...)
```

---

# 13. Vizualizace

Grafy mají být jednoduché a sloužit především k interpretaci výsledků.

Používat:

```python
plt.figure(...)
plt.bar(...)
plt.title(...)
plt.xlabel(...)
plt.ylabel(...)
plt.xticks(...)
plt.legend(...)
plt.tight_layout()
plt.show()
```

Není potřeba používat pokročilé vizualizační knihovny.

Graf musí odpovídat číselným výsledkům vypsaným skriptem.

---

# 14. Jak postupovat při reprodukci projektu

Doporučený postup:

## Krok 1

Umístit do stejného projektu:

```text
flights_4ft (1).csv
task1_sdcf.py
task2_action_rules.py
task3_kl.py
task4_sdkl.py
```

## Krok 2

Nainstalovat potřebné balíčky:

```bash
pip install pandas numpy matplotlib scipy
```

## Krok 3

Spouštět skripty samostatně:

```bash
python task1_sdcf.py
python task2_action_rules.py
python task3_kl.py
python task4_sdkl.py
```

## Krok 4

Výstupy porovnat s očekávanými hodnotami uvedenými v tomto dokumentu.

Malé rozdíly způsobené pouze formátováním nebo zaokrouhlením jsou přijatelné.

Významně odlišné hodnoty znamenají:

- jiný vstupní dataset,
- změnu filtrace,
- změnu kategorií,
- nebo změnu analytického postupu.

---

# 15. Doporučený způsob práce s ChatGPT

Při reprodukci lze ChatGPT zadat například následující kontext:

```text
Pracuji na semestrálním projektu 4IZ457 nad datasetem
US Flight Delays 2025-01.

Používám soubor flights_4ft (1).csv s 522 269 řádky.

Chci vytvářet jednoduché Python analýzy, které musí být:
- reprodukovatelné,
- vysvětlitelné při obhajobě,
- statisticky ověřené,
- doplněné human checkem.

Nepoužívej zbytečně složitý kód.

U statisticky významných výsledků kontroluj také praktickou sílu vztahu,
velikosti skupin a očekávané četnosti.

Výsledky interpretuj jako asociace, nikoliv kauzální vztahy.
```

Poté zadávat jednotlivé analytické otázky podle kapitol 8–11 tohoto dokumentu.

---

# 16. Kontrola výsledků

Před považováním skriptu za finální vždy ověřit:

```text
[ ] Skript se spustí bez chyby.
[ ] Používá správný dataset.
[ ] Počet řádků odpovídá 522 269.
[ ] Výsledky odpovídají očekávaným hodnotám.
[ ] Hlavní skupiny mají dostatečný počet pozorování.
[ ] Chí-kvadrát test má přijatelné očekávané četnosti.
[ ] P-hodnota není interpretována jako síla vztahu.
[ ] Je uveden Cramérův V tam, kde je relevantní.
[ ] Extrémní malé skupiny nejsou používány jako hlavní příklady.
[ ] Výsledek není interpretován kauzálně.
[ ] Graf odpovídá tabulkovým výsledkům.
```

---

# 17. Role AI a odpovědnost autora

ChatGPT byl použit pro:

- návrh analytických otázek,
- návrh metodiky,
- tvorbu Python kódu,
- zpřehlednění kódu,
- statistickou interpretaci,
- návrh human checků,
- kontrolu konzistence mezi kódem a výsledky,
- přípravu podkladů pro prezentaci.

Výsledný kód však nebyl převzat bez kontroly.

Každý finální skript byl spuštěn v lokálním prostředí a hlavní výsledky byly ověřeny proti skutečným výstupům programu.

Při interpretaci byly navíc kontrolovány velikosti jednotlivých skupin, protože extrémní procentní hodnoty mohou vzniknout u malého počtu pozorování.

Zvláštní pozornost byla věnována rozdílu mezi:

```text
statistickou významností
a
praktickou významností
```

a mezi:

```text
asociací
a
kauzalitou
```

---

# 18. Finální soubory této části projektu

Hlavní skripty:

```text
task1_sdcf.py
task2_action_rules.py
task3_kl.py
task4_sdkl.py
```

Dokumentace AI postupu:

```text
CHATGPT.md
```

Všechny čtyři skripty jsou navrženy tak, aby jejich logika byla přímo dohledatelná z tohoto dokumentu a aby bylo možné při použití stejného datasetu reprodukovat stejné nebo velmi podobné výsledky.