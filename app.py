"""Kreuzentropie: die Hilfsverteilung lernen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 8 der Linie "Monte-Carlo-Schätzung und Varianzreduktion": Die beste Kippung der Importance-Sampling-Hilfsverteilung (Stück 7) kennt man nur, wenn man die Antwort schon halb kennt.
Die Kreuzentropie-Methode lernt sie in wenigen Stufen aus einer Elite der Ziehungen; sie fährt sich fest, wenn die Elite zu groß oder die erste Stufe ohne Treffer ist. Vergleich bei gleichem
Budget mit einfachem Ziehen, dem unbekannten Optimum und adaptivem Splitting auf derselben Bewertungsfunktion. Vehikel ist der Ausfall eines Gitternetzes, die Wahrheit ist exakt aufgezählt.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import ken_constants as C
import ken_evaluation as ev
from ken_evaluation import Settings, de, de_e, dn, dz
from ken_layout import dunkles_design, kopf_css
from ken_pdf_export import build_pdf
from ken_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    options,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from ken_scenario import named_netz, parse_edges
from ken_visualization import build_budget, build_fixpoint, build_grid, build_netz, build_prediction, build_trajectories

st.set_page_config(page_title="Kreuzentropie: die Hilfsverteilung lernen – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Alle Kantenmengen aufzählen …")
def _table(netz_key):
    return ev.netz_table(named_netz(netz_key))


@st.cache_data(show_spinner="Lernläufe wiederholen …")
def _analysis(settings, table):
    return ev.analyse(settings, table=table)


@st.cache_data(show_spinner="ρ und N durchrechnen …")
def _grid(settings, table):
    return ev.rho_n_grid(settings, named_netz(settings.netz), table)


@st.cache_data(show_spinner="Verfahren bei gleichem Budget vergleichen …")
def _budget(settings, table):
    return ev.budget_compare(settings, named_netz(settings.netz), table)


@st.cache_data(show_spinner=False)
def _prediction(settings, table):
    return ev.splitting_prediction(settings, named_netz(settings.netz), table)


@st.cache_data(show_spinner=False)
def _fixpoint(settings, table):
    return ev.fixpoint_table(settings, named_netz(settings.netz), table)


@st.cache_data(show_spinner=False)
def _custom(text, settings):
    netz = parse_edges(text)
    return netz, ev.custom_result(settings, netz)


def _netz_label(key):
    netz = named_netz(key)
    return f"{netz.name} ({netz.m} Kanten)"


def _q_label(q):
    return f"{q:g}".replace(".", ",")


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in ken_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧭 Kreuzentropie: die Hilfsverteilung lernen")
st.markdown(
    """
Importance Sampling (Stück 7) braucht eine **Hilfsverteilung**; die beste Kippung kennt man nur, wenn man den kleinsten Schnitt des Netzes schon kennt. Die **Kreuzentropie-Methode** lernt sie:
man zieht mit der aktuellen Kippung q', behält die **Elite** (die besten Ziehungen nach einer Bewertungsfunktion), setzt q' auf den **gewichteten Ausfallanteil der Elite** und wiederholt, bis die Elite das Ereignis selbst ist.
Mit **einer** Zahl als Familie genügen **zwei Stufen**, und das gelernte q' ist beim Varianzquadrat fast so gut wie das Optimum. Aber das Lernen kann **scheitern**: ist die Elite zu groß, läuft q' in einen **Zwischen-Fixpunkt**;
sieht die erste Stufe keinen Treffer, kommt es gar nicht in Gang. Bei gleichem Budget liegt es dicht hinter dem unbekannten Optimum, weit vor einfachem Ziehen und vor **adaptivem Splitting** auf derselben Bewertung.

Weiter unten: **🔬 fünf Experimente**, **🧪 Spielwiese**, **🚧 Wo die Annahmen enden** und die Expander **⚠️ Annahmen, Grenzen und bekannte Probleme** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die Kreuzentropie-Methode", expanded=True):
    st.markdown(
        """
1. **Ereignis und Bewertung:** Jede Kante fällt unabhängig mit q aus; gesucht ist die Wahrscheinlichkeit, dass das Netz nicht mehr zusammenhängt. Die Bewertung s ist 0 beim Ausfall und sonst minus dem kleinsten Grad der Knoten: je näher ein Knoten an der Isolation ist, desto höher s.
2. **Stufe:** N Ziehungen mit der aktuellen Kippung q' (zuerst q). Die Schwelle ist die Bewertung der ⌈ρN⌉-besten Ziehung (höchstens 0); die **Elite** sind alle Ziehungen mit s ≥ Schwelle.
3. **Update:** q'_neu = Σ W f / (m Σ W) über die Elite, mit f = Zahl der Ausfälle, W = Dichtequotient Original zu q'. Das ist die Maximum-Likelihood-Anpassung der Familie an die Elite. Der Erwartungswert ist **E_q[f | s ≥ Schwelle] / m**, unabhängig vom aktuellen q'.
4. **Ende:** Sobald die Schwelle 0 ist (die Elite ist der Ausfall selbst), schätzt man mit frischen Ziehungen und dem gelernten q'. **Festgefahren:** alle Ziehungen sind Elite, oder nach 8 Stufen ist die Schwelle noch unter 0.
5. **Fixpunkt-Falle:** Bleibt der Ausfall unter dem aktuellen q' seltener als ρ, wählt die nächste Stufe wieder dieselbe Schwelle und q' ändert sich nicht mehr. ρ muss unter der Ausfallwahrscheinlichkeit der Zwischenstufe liegen und die erste Stufe braucht mindestens ρN Treffer.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Netz und Ereignis**")
    netz_key = st.selectbox("Netz", options("netz_select"), format_func=_netz_label, key="netz_select",
                            help="Gitter (Knoten in Zeilen und Spalten) oder die Hantel: zwei Dreiecke, durch eine einzige Brücke verbunden. Ereignis: das Netz hängt nicht mehr zusammen.")
    q = st.select_slider("Ausfallwahrscheinlichkeit q je Kante", options("q_select")[::-1], key="q_select", format_func=_q_label, help="Je kleiner q, desto seltener der Netzausfall.")
    st.markdown("**Lernen**")
    n = st.select_slider("Ziehungen N je Stufe", options("n_select"), key="n_select", help="Ziehungen je Lernstufe.")
    rho = st.select_slider("Anteil der Elite ρ", options("rho_select"), key="rho_select", format_func=_q_label,
                           help="Welcher Anteil der Ziehungen die Elite bildet (mit Gleichstand: alle bis zur Schwelle). Kleiner als die Ausfallwahrscheinlichkeit der Zwischenstufe, aber mit mindestens einigen Treffern.")
    st.markdown("**Vergleich**")
    budget = st.select_slider("Gesamtbudget (Ziehungen)", options("budget_select"), key="budget_select", help="Budget für den Vergleich der vier Verfahren in Experiment 3 (Lernen und Schätzen eingerechnet).")
    seed = st.number_input("Zufallsstrom", *bounds("seed_input"), key="seed_input", step=1, help="Welcher feste Strom von Zufallszahlen benutzt wird (SplitMix64).")
    st.button("🎲 Neuen Strom wählen", width="stretch", on_click=randomize_seed)

sync_query_params({"netz_select": netz_key, "q_select": float(q), "n_select": int(n), "rho_select": float(rho), "budget_select": int(budget), "seed_input": int(seed)})

settings = Settings(str(netz_key), float(q), int(n), float(rho), int(budget), int(seed))
table = _table(settings.netz)
a = _analysis(settings, table)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
if a.P > 0:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ausfallwahrscheinlichkeit P", de_e(a.P), help=f"Exakte Wahrscheinlichkeit, dass das {a.netz.name} bei q = {_q_label(settings.q)} je Kante nicht mehr zusammenhängt (Aufzählung aller Kantenmengen).")
    m2.metric("Optimale Kippung q' (exakt)", de(a.qp_opt, 3), help="Das q' mit dem kleinsten Varianzquadrat aus Stück 7 (exakte Aufzählung); das Lernen kennt es nicht.")
    m3.metric("Gelernte Kippung q'", dn(a.qp_med, 3), delta=f"erwartet {de(a.qp_ladder, 3)}", delta_color="off", help="Median über die erfolgreichen Lernläufe; erwartet ist das Ende der exakten Stufenfolge aus Erwartungswerten.")
    m4.metric("Festgefahren", f"{de(a.stuck * 100, 0)} %", help=f"Anteil der {C.REPS} Lernläufe, die das Ereignisniveau nicht erreichen.")
    stufen = " → ".join(de(x[3], 3) for x in a.ladder)
    st.caption(f"Exakte Stufenfolge: q' = {de(a.ladder[0][0], 3)} → {stufen} ({'erreicht das Ereignis' if a.reached else 'erreicht das Ereignis nicht'}). Größtes ρ, das das Ereignis erreicht: etwa {de(a.rho_hi, 3)}. "
               f"Gemessen: {dn(a.stages, 1)} Stufen, {dz(a.evals) if a.evals == a.evals else 'nicht bestimmbar'} Ziehungen, CV² des gelernten q' das {dn(a.ratio_med, 2)}-Fache des Optimums.")
    st.plotly_chart(build_trajectories(a), width="stretch", key="main_chart")
    st.caption("Dünn: die ersten Lernläufe; gestrichelt: die exakte Stufenfolge aus Erwartungswerten; punktiert: das exakte Optimum (logarithmische Achse).")
else:
    st.info("Bei diesen Einstellungen fällt das Netz nie aus.")

if st.button("📄 PDF der Schätzung erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(settings, a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="kreuzentropie.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wie lernt man die Hilfsverteilung aus den Ziehungen, und wann gelingt es nicht?")
kind, text = ev.verdict_main(a, settings)
getattr(st, kind)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Lernverlauf")
st.caption("Die ersten Lernläufe gegen die exakte Stufenfolge (Bild oben). Der Erwartungswert des Updates ist E_q[f | Elite] / m und hängt nur vom Niveau der Elite ab, nicht vom aktuellen q'.")
kind, text = ev.verdict_trajectories(a)
getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 2: Wie klein muss die Elite sein, wie groß die Stufe?")
st.caption(f"Anteil festgefahrener Läufe (Farbe) und Verhältnis CV² des gelernten q' zum Optimum (Zahl) für jede Kombination von Ziehungen je Stufe und ρ, je {C.GRID_REPS} Lernläufe. Kostet einige Sekunden.")
if st.button("▶ Messung starten", key="run_grid"):
    st.session_state["grid_done"] = True
if st.session_state.get("grid_done"):
    ex2 = _grid(settings, table)
    st.plotly_chart(build_grid(ex2), width="stretch", key="grid_chart")
    kind, text = ev.verdict_grid(ex2, settings)
    getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 3: Gleiches Budget")
st.caption(f"Vier Verfahren bei Gesamtbudget {de(settings.budget, 0)} Ziehungen (Lernen eingerechnet; ein Gibbs-Durchlauf eines Teilchens zählt wie eine Ziehung): einfach, Orakel mit bekanntem q', Kreuzentropie (ein Sechstel des Budgets je Lernstufe), adaptives Splitting. "
           f"{C.REPS} Wiederholungen; festgefahrene Läufe zählen als Schätzung null. Kostet einige Sekunden.")
if st.button("▶ Messung starten", key="run_budget"):
    st.session_state["budget_done"] = True
if st.session_state.get("budget_done"):
    ex3 = _budget(settings, table)
    if ex3["P"] > 0:
        st.plotly_chart(build_budget(ex3), width="stretch", key="budget_chart")
        st.dataframe(
            {"Verfahren": [r["method"] for r in ex3["rows"]], "festgefahren": [f"{de(r['stuck'] * 100, 0)} %" for r in ex3["rows"]], "Nullschätzung": [f"{de(r['zero'] * 100, 0)} %" for r in ex3["rows"]],
             "Verzerrung": [de(r["bias"], 3) for r in ex3["rows"]], "relativer Fehler (RMSE)": [de(r["rmse"], 2) for r in ex3["rows"]], "mediane Abweichung": [de(r["med"], 2) for r in ex3["rows"]],
             "mehr als 50 % Fehler": [f"{de(r['gt50'] * 100, 0)} %" for r in ex3["rows"]]},
            hide_index=True, width="stretch", key="budget_table",
        )
    kind, text = ev.verdict_budget(ex3)
    getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 4: Warum Splitting hier zurückliegt")
st.caption("Exakt (kein Zufall): die Stufenwahrscheinlichkeiten des Splittings auf derselben Bewertung und der daraus vorhergesagte relative Fehler, Σ (1 − p_t)/(p_t N), neben den Vorhersagen der anderen Verfahren.")
ex4 = _prediction(settings, table)
if ex4["P"] > 0:
    st.plotly_chart(build_prediction(ex4), width="stretch", key="pred_chart")
    st.dataframe(
        {"Budget": [r["budget"] for r in ex4["rows"]], "einfach": [de(r["plain"], 2) for r in ex4["rows"]], "Orakel": [de(r["oracle"], 2) for r in ex4["rows"]], "Kreuzentropie": [dn(r["ce"], 2) for r in ex4["rows"]],
         "adaptives Splitting": [de(r["split"], 2) for r in ex4["rows"]]},
        hide_index=True, width="stretch", key="pred_table",
    )
kind, text = ev.verdict_prediction(ex4)
getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 5: Die Fixpunkt-Falle")
st.caption("Exakt (kein Zufall): die Stufenfolge der Erwartungswerte für jedes ρ. Ist das Ereignis unter dem aktuellen q' seltener als ρ, wählt die Stufe wieder dasselbe Niveau und q' bleibt stehen.")
ex5 = _fixpoint(settings, table)
if a.P > 0:
    st.plotly_chart(build_fixpoint(ex5), width="stretch", key="fix_chart")
    st.dataframe(
        {"ρ": [de(r["rho"], 3) for r in ex5["rows"]], "Stufenfolge q'": [" → ".join([de(r["stages"][0][0], 3)] + [de(x[3], 3) for x in r["stages"]]) for r in ex5["rows"]],
         "Niveau je Stufe": [", ".join(str(x[1]) for x in r["stages"]) for r in ex5["rows"]], "Ergebnis": ["erreicht das Ereignis" if r["reached"] else "Fixpunkt" if r["fixpoint"] else "keine Elite" for r in ex5["rows"]]},
        hide_index=True, width="stretch", key="fix_table",
    )
kind, text = ev.verdict_fixpoint(ex5, settings)
getattr(st, kind)(text)

st.markdown("---")

# --- Spielwiese -----------------------------------------------------------------------------------------------------------------------------

with st.expander("🧪 Spielwiese: ein eigenes Netz"):
    st.caption(
        f"Eine Kante je Zeile, zwei ganze Zahlen (»0-1«). Höchstens {C.CUSTOM_EDGE_LIMIT} Kanten und {C.CUSTOM_NODE_LIMIT} Knoten; q, N, ρ und Strom stammen aus der Seitenleiste. "
        "Es zeigt die exakte Stufenfolge und die Lernläufe; Ereignis ist wieder: das Netz hängt nicht zusammen."
    )
    text_in = st.text_area("Kanten", "0-1\n1-2\n2-0\n3-4\n4-5\n5-3\n2-3", height=140, key="eigene_kanten")
    try:
        cnetz, cres = _custom(text_in, settings)
    except ValueError as err:
        st.error(str(err))
    else:
        ca = cres["analysis"]
        s1, s2, s3 = st.columns(3)
        s1.metric("Ausfallwahrscheinlichkeit P", de_e(ca.P) if ca.P > 0 else "0")
        s2.metric("optimale Kippung q'", dn(ca.qp_opt, 3))
        s3.metric("Festgefahren", f"{de(ca.stuck * 100, 0)} %")
        st.plotly_chart(build_netz(cnetz), width="stretch", key="custom_net")
        kind, text = ev.verdict_main(ca, settings)
        getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Eine Zahl genügt als Familie** | Eine globale Kippung ist die einfachste Familie; je Kante verschieden gekippt wäre besser und braucht mehr Elite je Parameter. | – |
| **Die Bewertungsfunktion trägt** | Hier sind es zwei Niveaus (ein Knoten mit nur noch einer Kante, dann der Ausfall). Eine grobe Bewertung schadet dem Splitting stärker als der Kreuzentropie. | – |
| **ρ ist klein genug, N groß genug** | Zu große Elite ⇒ Zwischen-Fixpunkt; zu wenige Treffer in der ersten Stufe ⇒ das Lernen kommt nicht in Gang (Experimente 2 und 5). | – |
| **Lernen und Schätzen sind getrennt** | Die Schätzung mit frischen Ziehungen ist unverzerrt; würde man die Lernziehungen wiederverwenden, entstünde eine Verzerrung. | – |
| **Das Ereignis ist ein Ausfall des Netzes** | Für andere Ereignisse braucht man eine andere Bewertungsfunktion. | – |
| **Splitting gegen Kreuzentropie** | Splitting braucht feinere Stufen; bei Warteschlangen sind sie naheliegend. | `splitting-demo` (Warteschlangen 9) |
"""
)

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(
        """
**Annahmen.** Alle Netze und Ausfälle sind **erfunden**; die Wahrheit ist die exakte Aufzählung. Die Kanten fallen unabhängig und mit gleicher Wahrscheinlichkeit aus. Die Zufallszahlen kommen aus festen Strömen (SplitMix64), nie aus einem einzelnen Seed.

**Grenzen.** Aufgezählt wird live bis 17 Kanten, Ereignis ist nur der Netzausfall. Es gibt **eine globale Kippung**, eine Bewertungsfunktion (kleinster Grad) und keine Glättung des Updates. Das Splitting benutzt feste zwei Gibbs-Durchläufe und zählt jeden Durchlauf eines
Teilchens wie eine Ziehung (eine **Annahme** über die Kosten). Festgefahrene Läufe zählen im Budgetvergleich als Schätzung null.

**Bekannte Probleme und Unsicherheiten.** Messungen mit wenigen Treffern (kleines q, kleine Stufe) rauschen stark; die exakte Stufenfolge ist belastbar. Das Verhältnis des CV² zum Optimum hängt vom Netz ab. Die Literaturangaben im Mathematik-Expander stammen aus dem Gedächtnis
und sind nicht eingesehen. Die Oberfläche ist mit AppTest geprüft und im Browser angesehen, nicht mit echtem Mausklick auf jedem Gerät.
        """
    )

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Problem.** Gesucht ist $P = \mathbb P(A)$ für das seltene Ereignis $A$ = „das Netz hängt nicht zusammen“; jede der $m$ Kanten fällt unabhängig mit $q$ aus. Familie der Hilfsverteilungen: jede Kante fällt mit $q'$ aus.

**Kreuzentropie.** Man minimiert die Kreuzentropie zwischen der (unbekannten) optimalen Hilfsverteilung $\propto \mathbf 1_A\,\mathbb P$ und der Familie. Stufenweise mit Elite $E_t = \{x : s(x) \ge \gamma_t\}$:
$$q'_{t+1} = \frac{\sum_{i \in E_t} W_i f_i}{m \sum_{i \in E_t} W_i},\qquad W_i = \Bigl(\tfrac{q}{q'_t}\Bigr)^{f_i}\Bigl(\tfrac{1-q}{1-q'_t}\Bigr)^{m-f_i}.$$
Der Erwartungswert dieses Updates ist $\mathbb E_q[f \mid s \ge \gamma_t]/m$ und hängt nicht von $q'_t$ ab; $\gamma_t$ ist die Bewertung der $\lceil \rho N \rceil$-besten Ziehung, höchstens 0.

**Fixpunkt-Falle.** Unter $q'_t$ hat das Niveau $\ell$ die Wahrscheinlichkeit $\mathbb P_{q'_t}(s \ge \ell)$. Ist $\mathbb P_{q'_t}(A) < \rho$, bleibt $\gamma_{t+1} = \gamma_t < 0$ und $q'_{t+1} = q'_t$: ein Fixpunkt unterhalb des Ereignisniveaus.

**Schätzer.** Mit dem gelernten $q'$ und $N_f$ frischen Ziehungen: $\hat P = \frac1{N_f}\sum_i \mathbf 1_A(x_i)\,W(x_i)$, unverzerrt. Das exakte relative Varianzquadrat ist
$$\mathrm{CV}^2(q') = \frac{\sum_f N_f\,[q^f(1-q)^{m-f}]^2 / [q'^f(1-q')^{m-f}]}{P^2} - 1.$$

**Adaptives Splitting.** Stufen $\ell_1 < \dots < \ell_T = A$ aus dem Quantil der Bewertung; $\hat P = \prod_t \hat p_t$ mit $\hat p_t$ dem Anteil der Teilchen auf dem Niveau; Teilchen werden mit Zurücklegen aus den Überlebenden gezogen und durch Gibbs-Durchläufe verändert (jede Kante aus der Originalverteilung neu, angenommen bei gehaltener Schwelle). Die relative Varianz ist etwa $\sum_t (1-p_t)/(p_t N)$.

**Wahrheit.** Die Aufzählung aller $2^m$ Kantenmengen liefert $P$, alle $N_f$, die Niveaus der Bewertung und die Erwartungswerte des Updates exakt.

**Literatur (aus dem Gedächtnis, nicht eingesehen).** Rubinstein 1999, „The cross-entropy method for combinatorial and continuous optimization“ (Methodology and Computing in Applied Probability); Rubinstein und Kroese, *The Cross-Entropy Method* 2004; de Boer, Kroese, Mannor und Rubinstein 2005, „A tutorial on the cross-entropy method“ (Annals of Operations Research); Cérou und Guyader 2007, „Adaptive multilevel splitting for rare event analysis“ (Stochastic Analysis and Applications).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
