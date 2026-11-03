"""Kennzahlen, Verdikt-Logik und Experimente über feste Ströme (reine Funktionen, kein Streamlit).

Ereignis A = das Netz hängt nicht zusammen (seltener Ausfall). Die Kreuzentropie lernt die Kippung q' (eine Zahl für alle Kanten) in wenigen Stufen aus einer Elite nach der Bewertung s;
das Ergebnis lässt sich exakt vorhersagen (deterministische Stufenfolge aus Erwartungswerten) und an der Simulation prüfen. Daneben adaptives Splitting auf derselben Bewertung.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import ken_algorithm as A
import ken_constants as C
from ken_scenario import Netz, named_netz


def de(x: float, nd: int = 2) -> str:
    """Zahl im deutschen Format: Punkt als Tausendertrenner, Komma als Dezimalzeichen, echtes Minuszeichen."""
    return f"{x:,.{nd}f}".translate(str.maketrans(",.", ".,")).replace("-", "−")


def dn(x: float, nd: int = 2) -> str:
    """Wie `de`, aber „nicht bestimmbar“ für NaN und „unendlich“ für ∞."""
    if np.isnan(x):
        return "nicht bestimmbar"
    if np.isinf(x):
        return "unendlich"
    return de(x, nd)


def de_e(x: float) -> str:
    """Wissenschaftliche Schreibweise mit Dezimalkomma, z. B. 4,2e-10."""
    return f"{x:.1e}".replace(".", ",")


def dz(x: float) -> str:
    """Zahl über viele Zehnerpotenzen: ab 10 Millionen wissenschaftlich, sonst mit Tausenderpunkt und passend vielen Stellen."""
    if not np.isfinite(x):
        return dn(x)
    if abs(x) >= 1e7:
        return de_e(x)
    if abs(x) >= 100:
        return de(x, 0)
    return de(x, 1 if abs(x) >= 10 else 2)


def kanten(n: int) -> str:
    return f"{n} Kante" if n == 1 else f"{n} Kanten"


@dataclass(frozen=True)
class Settings:
    netz: str = C.DEFAULT_NETZ
    q: float = C.DEFAULT_Q
    n: int = C.DEFAULT_N
    rho: float = C.DEFAULT_RHO
    budget: int = C.DEFAULT_BUDGET
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Analysis:
    netz: Netz
    P: float                      # exakte Ausfallwahrscheinlichkeit
    cv2_plain: float
    qp_opt: float                 # optimale Kippung (exakt)
    cv2_opt: float
    ladder: tuple                 # deterministische Stufenfolge: Tupel von (q' vorher, Niveau, Wahrscheinlichkeit, q' nachher)
    reached: bool                 # erreicht die Stufenfolge das Ereignis?
    fixpoint: bool                # bleibt q' in einem Zwischen-Fixpunkt stehen?
    rho_hi: float                 # größtes rho (aus einem Raster), bei dem die Stufenfolge das Ereignis erreicht
    qp_ladder: float              # q' am Ende der Stufenfolge (Erwartungswert des Lernens)
    cv2_ladder: float             # exaktes CV^2 bei diesem q'
    p_event: float                # Wahrscheinlichkeit des Ereignisses unter diesem q'
    stuck: float                  # gemessen: Anteil festgefahrener Läufe
    stages: float                 # gemessen: mittlere Zahl der Stufen (nur erfolgreiche Läufe)
    qp_med: float                 # gemessen: Median des gelernten q'
    qp_lo: float
    qp_hi: float
    ratio_med: float              # gemessen: Median von CV^2(gelerntes q') / CV^2(Optimum)
    ratio_gt2: float              # Anteil der erfolgreichen Läufe mit Verhältnis über 2
    evals: float                  # mittlere Zahl der Ziehungen des Lernens (erfolgreiche Läufe)
    trajectories: tuple           # die ersten Läufe: Verlauf von q'


# --- Wahrheit -----------------------------------------------------------------------------------------------------------------------------

def netz_table(netz: Netz):
    """Aufzählung aller Kantenmengen (höchstens C.LIVE_EDGE_LIMIT Kanten): Bewertung s, X, Zahl haltender Kanten k und die Ausfallzählung N_f."""
    if netz.m > C.LIVE_EDGE_LIMIT:
        raise ValueError(f"{netz.m} Kanten: aufgezählt wird nur bis {C.LIVE_EDGE_LIMIT}")
    s, X, k = A.score_table(netz.n, netz.edges)
    return s, X, k, A.failure_counts(X, k, netz.m)


def rho_reach_limit(s_tab, k, m, q) -> float:
    """Größtes rho aus einem geometrischen Raster, bei dem die deterministische Stufenfolge das Ereignis erreicht (0, wenn keines)."""
    best = 0.0
    for rho in np.geomspace(1e-4, 0.9, 120):
        if A.ce_ladder(s_tab, k, m, q, float(rho))["reached"]:
            best = float(rho)
    return best


# --- Hauptanalyse -------------------------------------------------------------------------------------------------------------------------

def analyse(settings: Settings, netz: Netz | None = None, table=None) -> Analysis:
    """Exakte Stufenfolge der Kreuzentropie und REPS Lernläufe mit festen Strömen."""
    netz = netz if netz is not None else named_netz(settings.netz)
    s_tab, X, k, Nf = table if table is not None else netz_table(netz)
    m = netz.m
    base = A.exact_is(Nf, settings.q, settings.q)
    qo = A.optimal_qp(Nf, settings.q)
    cvo = A.exact_is(Nf, settings.q, qo)["cv2"] if np.isfinite(qo) else float("nan")
    lad = A.ce_ladder(s_tab, k, m, settings.q, settings.rho)
    qpl = lad["stages"][-1]["qp_after"]
    cv2l = A.exact_is(Nf, settings.q, qpl)["cv2"]
    rows = [(x["qp_before"], x["level"], x["prob"], x["qp_after"]) for x in lad["stages"]]
    qs, ratios, stages, evals, trajs, stuck = [], [], [], [], [], 0
    for rep in range(C.REPS):
        r = A.ce_learn(s_tab, m, settings.q, settings.n, settings.rho, (1, netz.n, m, int(round(settings.q * 1e4)), settings.n, int(round(settings.rho * 1e4)), settings.seed, rep))
        if len(trajs) < 6:
            trajs.append(tuple(r["traj"]))
        if r["stuck"]:
            stuck += 1
            continue
        qs.append(r["traj"][-1])
        ratios.append(A.exact_is(Nf, settings.q, r["traj"][-1])["cv2"] / cvo)
        stages.append(len(r["thrs"]))
        evals.append(r["evals"])
    ok = len(qs) > 0
    return Analysis(
        netz, base["P"], base["cv2"], qo, cvo, tuple(rows), lad["reached"], lad["fixpoint"], rho_reach_limit(s_tab, k, m, settings.q), qpl, cv2l, A.prob_at(s_tab, k, m, qpl, 0), stuck / C.REPS,
        float(np.mean(stages)) if ok else float("nan"), float(np.median(qs)) if ok else float("nan"), float(np.percentile(qs, 10)) if ok else float("nan"), float(np.percentile(qs, 90)) if ok else float("nan"),
        float(np.median(ratios)) if ok else float("nan"), float(np.mean(np.array(ratios) > 2)) if ok else float("nan"), float(np.mean(evals)) if ok else float("nan"), tuple(trajs))


def verdict_main(a: Analysis, settings: Settings) -> tuple[str, str]:
    """Lernt die Kreuzentropie eine brauchbare Kippung, oder fährt sie sich fest, und warum?"""
    if not (a.P > 0):
        return "info", "Das Netz fällt bei diesen Einstellungen nie aus: das Ereignis ist unmöglich, es gibt nichts zu lernen."
    kopf = (f"Die Ausfallwahrscheinlichkeit ist {de_e(a.P)}; die optimale Kippung wäre q' = {de(a.qp_opt, 3)} (CV² = {dz(a.cv2_opt)} gegen {dz(a.cv2_plain)} beim einfachen Ziehen). ")
    if not a.reached:
        if a.fixpoint and len(a.ladder) > 1:
            last = a.ladder[-1]
            return "warning", kopf + (f"Mit ρ = {de(settings.rho, 3)} fährt sich das Lernen fest: q' läuft auf {de(last[3], 3)} und bleibt dort, denn bei diesem q' hat das Ereignis nur die Wahrscheinlichkeit "
                                     f"{de(a.p_event, 3)} und liegt unter ρ. Die Elite bleibt eine Zwischenstufe; ρ müsste unter {de(a.rho_hi, 3)} liegen. Gemessen: {de(a.stuck * 100, 0)} % der Läufe festgefahren.")
        return "warning", kopf + (f"Mit ρ = {de(settings.rho, 3)} und N = {de(settings.n, 0)} je Stufe findet schon die erste Stufe keine Elite, die sich von allen Ziehungen unterscheidet (ρ liegt über der "
                                 f"Wahrscheinlichkeit der ersten Stufe): {de(a.stuck * 100, 0)} % der Läufe festgefahren. ρ müsste unter {de(a.rho_hi, 3)} liegen.")
    if a.stuck > 0.5:
        return "warning", kopf + (f"Die Erwartungswerte erreichen das Ereignis in {len(a.ladder)} Stufen, aber mit N = {de(settings.n, 0)} Ziehungen je Stufe und ρ = {de(settings.rho, 3)} bleiben {de(a.stuck * 100, 0)} % der Läufe "
                                 f"stecken: die erste Stufe sieht zu oft zu wenige Treffer (erwartet {dz(settings.n * a.ladder[0][2])}, nötig mindestens {dz(settings.rho * settings.n)}).")
    text = kopf + (f"Das Lernen braucht {de(a.stages, 1)} Stufen ({dz(a.evals)} Ziehungen) und liefert q' = {de(a.qp_med, 3)} (10 bis 90 %: {de(a.qp_lo, 3)} bis {de(a.qp_hi, 3)}); der exakte CV² dort ist "
                   f"{de(a.ratio_med, 2)}-mal so groß wie beim Optimum. Festgefahren: {de(a.stuck * 100, 0)} % der Läufe.")
    return "success", text


# --- Experiment 1: Lernverlauf ------------------------------------------------------------------------------------------------------------------

def verdict_trajectories(a: Analysis) -> tuple[str, str]:
    if not (a.P > 0):
        return "info", "Das Ereignis ist unmöglich, es gibt keinen Verlauf."
    first = a.trajectories[0]
    erwartet = " → ".join(de(x[3], 3) for x in a.ladder)
    return "info", (f"Erwartet (exakt): q' = {de(a.ladder[0][0], 3)} → {erwartet} nach {len(a.ladder)} Stufen; Zielwert je Niveau ist der mittlere Ausfallanteil unter dem Original, bedingt auf die Elite. "
                    f"Der erste Lauf: {' → '.join(de(x, 3) for x in first)}.")


# --- Experiment 2: ρ und N je Stufe -------------------------------------------------------------------------------------------------------------

def rho_n_grid(settings: Settings, netz: Netz, table) -> dict:
    """Je Zelle (N je Stufe, ρ): Anteil festgefahrener Läufe und Median des Verhältnisses CV²(gelerntes q') / CV²(Optimum) (aus GRID_REPS Läufen)."""
    s_tab, X, k, Nf = table
    m = netz.m
    qo = A.optimal_qp(Nf, settings.q)
    cvo = A.exact_is(Nf, settings.q, qo)["cv2"] if np.isfinite(qo) else float("nan")
    cells = []
    for N in C.N_CHOICES:
        for rho in C.RHO_CHOICES:
            stuck, ratios = 0, []
            for rep in range(C.GRID_REPS):
                r = A.ce_learn(s_tab, m, settings.q, N, rho, (2, netz.n, m, int(round(settings.q * 1e4)), N, int(round(rho * 1e4)), settings.seed, rep))
                if r["stuck"]:
                    stuck += 1
                else:
                    ratios.append(A.exact_is(Nf, settings.q, r["traj"][-1])["cv2"] / cvo)
            cells.append({"n": N, "rho": rho, "stuck": stuck / C.GRID_REPS, "ratio": float(np.median(ratios)) if ratios else float("nan")})
    return {"cells": cells}


def verdict_grid(exp: dict, settings: Settings) -> tuple[str, str]:
    cells = exp["cells"]
    ok = [c for c in cells if c["stuck"] < 0.05]
    bad = [c for c in cells if c["stuck"] > 0.95]
    best = ", ".join(f"ρ = {de(c['rho'], 3)}" for c in ok if c["n"] == max(C.N_CHOICES)) or "keines"
    return "info", (f"Zellen ohne Festfahren (unter 5 %): {len(ok)} von {len(cells)}; immer festgefahren: {len(bad)}. Bei N = {de(max(C.N_CHOICES), 0)} je Stufe gelingt das Lernen für {best}. "
                    f"Wo es gelingt, ist das Verhältnis zum Optimum nahe 1: das Lernen scheitert selten an der Güte, meist an ρ und an zu wenigen Treffern in der ersten Stufe.")


# --- Experiment 3: gleiches Budget ------------------------------------------------------------------------------------------------------------

METHODS = ("einfach", "Orakel (festes Optimum)", "Kreuzentropie", "adaptives Splitting")


def budget_compare(settings: Settings, netz: Netz, table) -> dict:
    """Bei Gesamtbudget B (Ziehungen; ein Gibbs-Durchlauf je Teilchen zählt wie eine Ziehung): einfach, Orakel (bekanntes q'), Kreuzentropie (Lernen plus Schätzen), adaptives Splitting."""
    s_tab, X, k, Nf = table
    m = netz.m
    q, B = settings.q, settings.budget
    P = A.exact_is(Nf, q, q)["P"]
    qo = A.optimal_qp(Nf, q)
    res = {name: [] for name in METHODS}
    rho_sp = min(settings.rho, C.SPLIT_RHO_MAX)
    for rep in range(C.REPS):
        key = (3, netz.n, m, int(round(q * 1e4)), B, int(round(settings.rho * 1e4)), settings.seed, rep)
        mask, _ = A.draw_masks(A.stream_seed(*key, 1), B, m, q)
        res["einfach"].append(float(np.mean(s_tab[mask] == 0)))
        res["Orakel (festes Optimum)"].append(A.is_estimate(s_tab, m, q, qo, B, (*key, 2)))
        L = A.ce_learn(s_tab, m, q, max(B // 6, 30), settings.rho, (*key, 3))
        res["Kreuzentropie"].append(float("nan") if L["stuck"] else A.is_estimate(s_tab, m, q, L["traj"][-1], max(B - L["evals"], 30), (*key, 4)))
        res["adaptives Splitting"].append(A.splitting(s_tab, m, q, max(int(B / (1 + C.SWEEPS)), 30), rho_sp, C.SWEEPS, (*key, 5))["est"])
    rows = []
    for name in METHODS:
        est = np.asarray(res[name], dtype=float)
        ok = ~np.isnan(est)
        e = np.where(ok, est, 0.0)
        rel = e / P - 1.0
        rows.append({"method": name, "stuck": float(1 - ok.mean()), "zero": float(np.mean(e == 0)), "bias": float(np.mean(rel)), "rmse": float(np.sqrt(np.mean(rel**2))),
                     "med": float(np.median(np.abs(rel))), "gt50": float(np.mean(np.abs(rel) > 0.5))})
    return {"rows": rows, "P": P, "budget": B}


def verdict_budget(exp: dict) -> tuple[str, str]:
    by = {r["method"]: r for r in exp["rows"]}
    if not (exp["P"] > 0):
        return "info", "Das Ereignis ist unmöglich, es gibt nichts zu vergleichen."
    ce, pl, sp, orc = by["Kreuzentropie"], by["einfach"], by["adaptives Splitting"], by["Orakel (festes Optimum)"]
    return "info", (f"Bei Budget {de(exp['budget'], 0)}: relativer Fehler (RMSE) einfach {de(pl['rmse'], 2)} ({de(pl['zero'] * 100, 0)} % Nullschätzungen), Orakel mit bekanntem q' {de(orc['rmse'], 2)}, "
                    f"Kreuzentropie {de(ce['rmse'], 2)} ({de(ce['stuck'] * 100, 0)} % festgefahren, als Null gezählt), adaptives Splitting {de(sp['rmse'], 2)} ({de(sp['stuck'] * 100, 0)} % festgefahren). "
                    f"Das Lernen kostet gegen das Orakel das {de(ce['rmse'] / orc['rmse'], 2)}-Fache im Fehler.")


# --- Experiment 4: warum Splitting zurückliegt (exakt) ------------------------------------------------------------------------------------------

def splitting_prediction(settings: Settings, netz: Netz, table) -> dict:
    """Exakte Vorhersage des relativen Fehlers je Gesamtbudget: einfach sqrt(CV^2 / B), Orakel sqrt(CV^2_opt / B), Kreuzentropie (Lernen mit B/6 je Stufe, dann Schätzen mit dem Rest und dem
    q' der Stufenfolge), Splitting sqrt(Summe (1 - p_t) / p_t / N) mit N = B / (1 + SWEEPS)."""
    s_tab, X, k, Nf = table
    m = netz.m
    q = settings.q
    base = A.exact_is(Nf, q, q)
    qo = A.optimal_qp(Nf, q)
    opt = A.exact_is(Nf, q, qo)
    lad = A.ce_ladder(s_tab, k, m, q, settings.rho)
    sp = A.splitting_ladder(s_tab, k, m, q, min(settings.rho, C.SPLIT_RHO_MAX))
    cv2l = A.exact_is(Nf, q, lad["stages"][-1]["qp_after"])["cv2"]
    rows = []
    for B in C.BUDGET_CHOICES:
        pilot = max(B // 6, 30) * len(lad["stages"])
        rows.append({"budget": B, "plain": float(np.sqrt(base["cv2"] / B)), "oracle": float(np.sqrt(opt["cv2"] / B)),
                     "ce": float(np.sqrt(cv2l / max(B - pilot, 30))) if lad["reached"] else float("nan"), "split": float(np.sqrt(sp["factor"] / max(int(B / (1 + C.SWEEPS)), 30)))})
    return {"rows": rows, "probs": sp["probs"], "levels": sp["levels"], "factor": sp["factor"], "reached": lad["reached"], "P": base["P"]}


def verdict_prediction(exp: dict) -> tuple[str, str]:
    if not (exp["P"] > 0):
        return "info", "Das Ereignis ist unmöglich, es gibt nichts vorherzusagen."
    probs = " und ".join(de(p, 4) for p in exp["probs"])
    r = exp["rows"][1]
    return "info", (f"Splitting hat hier {len(exp['probs'])} Stufen mit den Wahrscheinlichkeiten {probs}; jede Stufe kostet einen Faktor (1 − p)/p in der relativen Varianz je Teilchen, zusammen {dz(exp['factor'])}. "
                    f"Vorhergesagter relativer Fehler bei Budget {de(r['budget'], 0)}: einfach {de(r['plain'], 2)}, Orakel {de(r['oracle'], 2)}, Kreuzentropie {dn(r['ce'], 2)}, Splitting {de(r['split'], 2)}. "
                    f"Die seltene Stufe (ein einzelner Kantenausfall) macht Splitting hier teuer; die Kreuzentropie braucht sie nicht.")


# --- Experiment 5: Fixpunkt-Falle (exakt) ---------------------------------------------------------------------------------------------------

def fixpoint_table(settings: Settings, netz: Netz, table) -> dict:
    """Exakte Stufenfolge der Kreuzentropie für jedes ρ aus RHO_CHOICES: Niveau, Wahrscheinlichkeit und q' je Stufe, erreicht oder Fixpunkt."""
    s_tab, X, k, Nf = table
    rows = []
    for rho in C.RHO_CHOICES:
        lad = A.ce_ladder(s_tab, k, netz.m, settings.q, rho)
        rows.append({"rho": rho, "stages": [(x["qp_before"], x["level"], x["prob"], x["qp_after"]) for x in lad["stages"]], "reached": lad["reached"], "fixpoint": lad["fixpoint"]})
    return {"rows": rows, "rho_hi": rho_reach_limit(s_tab, k, netz.m, settings.q)}


def verdict_fixpoint(exp: dict, settings: Settings) -> tuple[str, str]:
    ok = [r["rho"] for r in exp["rows"] if r["reached"]]
    bad = [r for r in exp["rows"] if not r["reached"]]
    text = (f"Das Ereignis wird für ρ = {', '.join(de(x, 3) for x in ok) or 'keines'} erreicht; die Grenze liegt (aus einem Raster) bei ρ ≈ {de(exp['rho_hi'], 3)}. ")
    if bad:
        b = bad[0]
        last = b["stages"][-1]
        text += f"Bei ρ = {de(b['rho'], 3)} bleibt q' bei {de(last[3], 3)} stehen (Niveau {last[1]}, Wahrscheinlichkeit {de(last[2], 3)})." if b["fixpoint"] else f"Bei ρ = {de(b['rho'], 3)} reicht die Elite nicht."
    return "info", text


# --- Spielwiese ---------------------------------------------------------------------------------------------------------------------------

def custom_result(settings: Settings, netz: Netz) -> dict:
    """Ein eigenes Netz: Analyse (exakte Stufenfolge und Lernläufe)."""
    table = netz_table(netz)
    return {"analysis": analyse(settings, netz=netz, table=table)}
