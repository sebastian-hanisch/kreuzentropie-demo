# Kreuzentropie: die Hilfsverteilung lernen – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-kreuzentropie-demo.streamlit.app/)**

Achtes Stück der Linie **„Monte-Carlo-Schätzung und Varianzreduktion“** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (die Linie wird erst vollständig veröffentlicht). Nachfolger von [importance-sampling-demo](https://github.com/sebastian-hanisch/importance-sampling-demo) (Stück 7, feste Kippung, deren Optimum man nur bei bekannter Struktur kennt). Hier hebt das Stück genau diese Annahme auf: Die Kippung wird **aus den Ziehungen gelernt**. Vehikel ist wie in Stück 1 bis 7 der Ausfall eines Gitternetzes (Ereignis: das Netz hängt **nicht** zusammen), dazu die Hantel. Die Wahrheit ist exakt aufgezählt; verglichen wird bei gleichem Budget mit einfachem Ziehen, dem unbekannten Optimum und **adaptivem Splitting** (das [splitting-demo](https://github.com/sebastian-hanisch/splitting-demo) aus den Warteschlangen nur mit festen Stufen rechnet).

## Warum dieses Problem

Importance Sampling braucht eine Hilfsverteilung. Die **Kreuzentropie-Methode** lernt sie stufenweise: Man zieht mit dem aktuellen q', behält die **Elite** (die besten Ziehungen nach einer Bewertungsfunktion), setzt q' auf den **gewichteten Ausfallanteil der Elite** und wiederholt, bis die Elite das Ereignis selbst ist. Die Fragen: **Wie viele Stufen, wie gut ist das gelernte q', wann fährt sich das Lernen fest, und wie schlägt es sich gegen Splitting?**

## Befunde und Korrekturen gegenüber dem Plan

Vor dem Bau wurden die Hypothesen der Gliederung an Messreihen geprüft (Vormessung im Website-Repo, `docs/planung/monte-carlo-planung/vormessung/`). Was sich geändert hat:

- **„Wenige Stufen genügen“ stimmt, aber nur mit kleinem ρ.** Mit einer Zahl als Familie genügen bei dieser Bewertung (zwei Niveaus) **zwei Stufen**; ist ρ größer als die Ausfallwahrscheinlichkeit unter dem erreichten q' (hier 0,0225), bleibt q' in einem **Zwischen-Fixpunkt** und erreicht das Ereignis nicht (bei ρ = 0,05 in 100 % der Läufe ab N = 1000, bei N = 100 in 62 %). Der Fixpunkt ist aus der Stufenfolge vorhersagbar: das Update ist als Verhältnis der Erwartungswerte bei fester Elite E_q[f | Elite]/m, unabhängig vom aktuellen q' (wegen der datenabhängigen Schwelle nur näherungsweise).
- **Die erste Stufe braucht Treffer.** Bei q = 0,001 scheitert das Lernen in 56 % der Läufe, wenn die Stufe zu klein ist.
- **Splitting hinkt hier nicht zufällig hinterher.** Die Bewertung hat nur zwei Niveaus mit den Stufenwahrscheinlichkeiten 0,078 und 0,0054, und der Fehler folgt in guter Näherung aus Σ (1 − p_t)/(p_t N).

## Modell

- **Ereignis und Bewertung:** A = das Netz hängt nicht zusammen; jede Kante fällt unabhängig mit q aus. Bewertung s = 0 bei Ausfall, sonst minus dem kleinsten Grad im Netz der haltenden Kanten; die Niveaumengen sind geschachtelt und {s ≥ 0} = A.
- **Kreuzentropie** (eine Kippung q' für alle Kanten): Stufe: N Ziehungen mit q'; Schwelle = Bewertung der ⌈ρN⌉-besten (höchstens 0); Elite = alle mit s ≥ Schwelle; q'_neu = Σ W f / (m Σ W) über die Elite, W = Dichtequotient. Ende, wenn die Schwelle 0 ist; festgefahren, wenn alle Ziehungen Elite sind oder nach 8 Stufen. Danach Schätzen mit frischen Ziehungen.
- **Adaptives Splitting:** Fixed Effort mit Quantil-Stufen; bei diskreter Bewertung springt die Stufe zum nächsthöheren vorhandenen Niveau; Teilchen werden mit Zurücklegen gezogen und durch zwei Gibbs-Durchläufe verändert (jede Kante aus der Originalverteilung, angenommen bei gehaltener Schwelle); Schätzung = Produkt der Stufenanteile. Kosten: ein Gibbs-Durchlauf eines Teilchens zählt wie eine Ziehung (eine **Annahme**).
- **Exakt:** CV² der Kippung aus den Zählungen N_f (wie in Stück 7); die deterministische Stufenfolge des Lernens aus den Niveauwahrscheinlichkeiten; die relative Varianz des Splittings aus den Stufenwahrscheinlichkeiten. Mathematik im 📐-Expander der App.

## Methodik

Zufallszahlen aus festen SplitMix64-Strömen (nie ein einzelner Seed). Wahrheit, Niveauwahrscheinlichkeiten und Update-Ziele aus der vollständigen Aufzählung. Lernläufe und Budgetvergleiche mit 150 Wiederholungen; die ρ-N-Tabelle mit je 150 Läufen.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. Standardfall: Gitter 3×4, q = 0,01 (P = 4,2·10⁻⁴), 1000 Ziehungen je Stufe, ρ = 0,01.

| Frage | Befund |
|---|---|
| Stufenwahrscheinlichkeiten, 3×4 (exakt) | q = 0,01: P(s ≥ −1) = **0,0776**, danach P(A \| s ≥ −1) = **0,0054**; q = 0,001: **0,0080** und **0,0005** |
| Exakte Stufenfolge, q = 0,01, ρ = 0,01 | q' = 0,010 → **0,0665** (Niveau −1, Wahrscheinlichkeit 0,0776) → **0,1288** (Niveau 0, Wahrscheinlichkeit 0,0225); optimales q' = 0,1185; das Ereignis wird für ρ bis etwa **0,022** erreicht (die Grenze ist die Wahrscheinlichkeit 0,0225 des Ausfalls unter q' = 0,0665; das Raster der App zeigt 0,021) |
| Gemessen, Standardfall (Bewertung mit zwei Niveaus) | **2,0 Stufen**, 2000 Ziehungen, nie festgefahren; gelerntes q' im Median **0,129** (10 bis 90 %: 0,125 bis 0,137); CV² des gelernten q' **1,01-mal** so groß wie beim Optimum |
| Fixpunkt-Falle, ρ = 0,05 | exakt: q' = 0,010 → 0,0665 → **0,0665** (Niveau bleibt −1, das Ereignis hat unter 0,0665 nur 0,0225 < ρ); gemessen **100 %** festgefahren |
| ρ und N je Stufe (q = 0,01, Anteil festgefahrener Läufe, N = 100 / 300 / 1000 / 3000) | ρ = 0,005 und 0,01: **0 / 0 / 0 / 0**; ρ = 0,02: 0,01 / 0 / 0 / 0; ρ = 0,05: **0,62 / 0,98 / 1,00 / 1,00**; ρ = 0,1: 1,00 überall; wo es gelingt, ist das CV²-Verhältnis zum Optimum **1,01 bis 1,02** (eine Ausnahme: 1,25 bei N = 100 und ρ = 0,01) |
| Noch seltener, q = 0,001 | erste Stufe nur 0,0080: ρ = 0,01 gibt keine Elite (Niveau −2, alle Ziehungen); N = 300, ρ = 0,01: **56 %** festgefahren; N = 3000, ρ = 0,005: **2 %**, q' = 0,119 (Optimum 0,118), Verhältnis 1,00 |
| Hantel (kleinster Schnitt eine Kante), q = 0,01, N = 1000 | ρ = 0,01: **1,4 Stufen**, q' = 0,158 (Optimum 0,144), Verhältnis 1,01, nie festgefahren; ρ = 0,05: 2 Stufen |
| Gleiches Budget B = 2000 / 6000 / 20 000, q = 0,01, relativer RMSE | einfach **1,09 / 0,62 / 0,36**; Orakel (festes Optimum, nicht lernbar ohne Wissen) **0,20 / 0,11 / 0,06**; **Kreuzentropie 0,25 / 0,15 / 0,07**; adaptives Splitting **0,55 / 0,32 / 0,18** |
| Dasselbe: Läufe mit mehr als 50 % Fehler, B = 2000 | einfach 65 %, Orakel 2 %, Kreuzentropie 2 %, Splitting 37 % |
| Gleiches Budget, q = 0,001, B = 20 000 | einfach **3,11** (93 % Nullschätzungen); Orakel 0,07; Kreuzentropie **0,22** (4 % festgefahren, als Null gezählt); Splitting **0,62** (5 % festgefahren) |
| Vorhersage des Splittings (in guter Näherung aus den Stufen, q = 0,01) | relativer Fehler **0,54 / 0,31 / 0,17** bei Budget 2000 / 6000 / 20 000 gegen gemessen 0,55 / 0,32 / 0,18; q = 0,001: Stufen 0,0080 und 0,0005, vorhergesagt 1,78 / 1,03 / 0,56 |
| Vorhersage der Kreuzentropie (exakt, Lernen mit einem Sechstel des Budgets je Stufe) | q = 0,01: **0,25 / 0,14 / 0,08**; q = 0,001, B = 20 000: **0,09** (gemessen 0,22 wegen 4 % festgefahrener Läufe, die als Null zählen) |

## Ehrliche Grenzen

- **Nur eine globale Kippung** für alle Kanten; die Mehrparameter-Familie (eine Kippung je Kante) ist nicht gebaut.
- **Eine Bewertungsfunktion** (kleinster Grad); mit einer feineren Bewertung hätte das Splitting mehr Stufen und würde besser. Der Vergleich gilt für **diese** Bewertung.
- **Kosten sind Ziehungen:** ein Gibbs-Durchlauf eines Teilchens zählt wie eine Ziehung; das ist eine Annahme, keine Messung.
- **Festgefahrene Läufe zählen im Budgetvergleich als Schätzung null**; im Anwendungsfall würde man ρ oder N ändern.
- **Nur das Gitter 3×4 und die Hantel sind gemessen**; andere Netze haben andere Niveaus.
- **Messungen mit wenigen Treffern** (kleines q, kleine Stufe) rauschen; die exakten Stufenfolgen und Vorhersagen sind belastbar.
- **Literaturangaben im Mathematik-Expander** stammen aus dem Gedächtnis und sind nicht eingesehen.

## Verifikation

- **Kern** (`tests/test_algorithm.py`): Zufallsstrom gegen Ganzzahl-Referenz, Bitpackung, Bewertung am Pfad von Hand, Update von Hand mit Brüchen (2/3 bei Gewichten 1, 5/8 bei q = ½, q' = ¾), Niveaus und Update-Ziele am Pfad (1/(2−q)), Stufenfolge, Gibbs-Schritt von Hand, Splitting und Lernen am Pfad.
- **Orakel** (`tests/test_oracle_ken.py`): Bewertung gegen `networkx` auf allen Kantenmengen von Hantel, 2×3 und 3×3; Niveauwahrscheinlichkeiten und Update-Ziele gegen eine Summation mit Brüchen; **gelerntes q' im Mittel gleich dem Ziel der exakten Stufenfolge**; Fixpunkt in der Simulation gleich dem exakten; **Gibbs-Durchlauf lässt die bedingte Originalverteilung exakt invariant** (Übergangsrechnung auf den Verteilungen der Hantel) und der Code gleicht der Übergangsrechnung im Dreieck; Splitting: Stufenwahrscheinlichkeiten und relative Varianz gegen Σ (1 − p_t)/(p_t N), erwartungstreu; Schätzer mit gelerntem q' erwartungstreu und mit der exakten Varianz.
- **Eigenschaften** (`tests/test_properties.py`): Update-Ziele wachsen mit dem Niveau, das Ziel des Ereignisniveaus liegt bei CV² höchstens 5 % über dem Optimum, nichts schlägt das Optimum, ρ-Grenze und Treffer-Bedingung der ersten Stufe, Festfahren, Splitting mit zwei Stufen, Kreuzentropie schlägt einfaches Ziehen.
- **Randfälle** (`tests/test_edge_cases.py`): Eingabefehler der Spielwiese, Pfad ohne Zwischenniveau, Reglergrenzen, unmögliches Ereignis, ρ·N unter eins, Hantel in einer Stufe.
- **Verdikte** (`tests/test_verdicts.py`): jede Schwelle und jede genannte Zahl an von Hand gesetzten Werten, deutsches Zahlenformat.
- **Presets, Layout, Oberfläche:** Presets vollständig und jedes mit dem erwarteten Verdikt; Schnellstart oben, fester Kopf, aktives Preset hervorgehoben; AppTest für jeden Button, jeden Regler an Minimum und Maximum, die Experimente, die Spielwiese, den PDF-Export und den Permalink.
- **Laufzeit-Importe, Footer, Claims.**

Zahl der Tests: siehe `pytest --collect-only -q tests`.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `ken_algorithm.py` | Kern: Strom, Bewertung, exakte Rechnung und Stufenfolgen, Kreuzentropie, Splitting |
| `ken_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `ken_evaluation.py` | Kennzahlen, Verdikt-Logik, Experimente über feste Ströme |
| `ken_layout.py` | Fester Kopf (Stylesheet) |
| `ken_pdf_export.py` | PDF der aktuellen Einstellung |
| `ken_presets.py` | Permalink (Begrenzen), Presets, Zufallsstrom-Knopf |
| `ken_scenario.py` | Gitter, Hantel und eigene Netze |
| `ken_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest, networkx und scipy |
| `requirements.txt` | Laufzeit-Bibliotheken |
| `tests/` | Kern von Hand, Orakel, Eigenschaften, Randfälle, Verdikte, Presets, Layout, AppTest, Footer, Laufzeit-Importe, Claims |

## Bewusst nicht umgesetzt

| Annahme | Verweis |
|---|---|
| Eine Kippung je Kante (Mehrparameter-Familie) | nicht gebaut |
| Glättung des Updates, Varianz-Minimierung statt Kreuzentropie | nicht gebaut |
| Feste Stufen für das Splitting | [splitting-demo](https://github.com/sebastian-hanisch/splitting-demo) (Warteschlangen 9) |
| Feste Kippung, deren Gewinn und optimale Lage | [importance-sampling-demo](https://github.com/sebastian-hanisch/importance-sampling-demo) (Stück 7) |
| Einen Teil exakt ausrechnen statt kippen | [bedingtes-monte-carlo-demo](https://github.com/sebastian-hanisch/bedingtes-monte-carlo-demo) (Stück 6) |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Monte-Carlo-Schätzung: vom Schätzer zur Wahl](https://sebastianhanisch.net/konzepte-monte-carlo.html).
