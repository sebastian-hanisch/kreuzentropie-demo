"""Kern der Demo (nur numpy).

Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b):
  Zufallsstrom    stream_seed, uniforms, draw_masks                     SplitMix64, zählerbasiert
  Ziel, Bewertung enumerate_x, score_table, failure_counts              Ereignis A = das Netz hängt NICHT zusammen; Bewertung s = 0 bei Ausfall, sonst -(kleinster Grad)
  Exakt           exact_is, optimal_qp, ce_ladder, splitting_ladder      CV^2 der Kippung, optimales q', deterministische Stufenfolge der Kreuzentropie und des Splittings
  Kreuzentropie   likelihood_ratio, ce_update, ce_learn, is_estimate     Elite, gewichtetes Maximum-Likelihood-Update, Schätzer mit dem gelernten q'
  Splitting       splitting                                              adaptives Fixed-Effort-Splitting mit Quantil-Stufen und Gibbs-Veränderung
Zufall kommt nur über einen übergebenen Strom, nie aus einem globalen Generator.

Ereignis A: jede Kante fällt unabhängig mit q aus. Bewertungsfunktion s(x) = 0, wenn das Netz nicht zusammenhängt, sonst s(x) = -delta(x) (kleinster Grad der Knoten im Netz der haltenden Kanten);
die Niveaumengen {s >= l} enthalten A und sind geschachtelt, {s >= 0} = A.
Kreuzentropie (eine Kippung q' für alle Kanten): Stufe t: N Ziehungen mit q'_t, Elite = alle Ziehungen mit s >= Schwelle, Schwelle = s der ceil(rho N)-besten Ziehung (höchstens 0);
q'_{t+1} = Summe_elite W f / (m Summe_elite W) mit W = (q/q'_t)^f ((1-q)/(1-q'_t))^(m-f) (Maximum Likelihood der Familie). Der Erwartungswert dieses Updates ist E_q[f | s >= Schwelle] / m,
unabhängig vom aktuellen q'. Ende, wenn die Schwelle 0 erreicht (die Elite ist das Ereignis); festgefahren, wenn alle Ziehungen Elite sind oder nach MAX_STAGES Stufen die Schwelle unter 0 bleibt.
"""

from __future__ import annotations

import numpy as np

import ken_constants as C

GAMMA = 0x9E3779B97F4A7C15
MASK = (1 << 64) - 1


# --- Zufallsstrom -------------------------------------------------------------------------------------------------------------------------

def mix_int(z: int) -> int:
    """SplitMix64-Mischfunktion mit Python-Ganzzahlen."""
    z &= MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return (z ^ (z >> 31)) & MASK


def _mix_arr(z: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def stream_seed(*keys: int) -> int:
    """Seed eines Stroms aus einem Schlüsseltupel (Einstellung, Wiederholung, ...)."""
    s = 0x1234_5678_9ABC_DEF0
    for k in keys:
        s = mix_int(s ^ mix_int(int(k) + GAMMA))
    return s


def uniforms(seed: int, count: int) -> np.ndarray:
    """`count` gleichverteilte Zahlen in [0, 1) (53 Bit) aus dem Strom `seed`; Ausgabe i hängt nur von (seed, i) ab."""
    idx = np.arange(1, count + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = _mix_arr(np.uint64(seed) + idx * np.uint64(GAMMA))
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


def draw_masks(seed: int, N: int, m: int, qp: float) -> tuple[np.ndarray, np.ndarray]:
    """N Ziehungen, jede Kante fällt mit qp aus: Maske der haltenden Kanten (Bit e = Kante e hält) und Zahl der Ausfälle f."""
    fail = uniforms(seed, N * m).reshape(N, m) < qp
    mask = ((~fail).astype(np.int64) << np.arange(m, dtype=np.int64)).sum(axis=1)
    return mask, fail.sum(axis=1)


# --- Ziel und Bewertung -------------------------------------------------------------------------------------------------------------------

def connected(present: np.ndarray, n: int, edges) -> np.ndarray:
    """Zusammenhang je Zeile (`present` ist (B, m) bool): Kleinste-Marke-Ausbreitung, zusammenhängend genau dann, wenn alle Marken 0 sind."""
    B = present.shape[0]
    if n <= 1:
        return np.ones(B, dtype=bool)
    lab = np.tile(np.arange(n, dtype=np.int16), (B, 1))
    while True:
        before = int(lab.sum())
        for e, (u, v) in enumerate(edges):
            low = np.minimum(lab[:, u], lab[:, v])
            pe = present[:, e]
            lab[:, u] = np.where(pe, low, lab[:, u])
            lab[:, v] = np.where(pe, low, lab[:, v])
        if int(lab.sum()) == before:
            break
    return lab.max(axis=1) == 0


def enumerate_x(n: int, edges, chunk: int = 1 << 15) -> tuple[np.ndarray, np.ndarray]:
    """Alle 2^m Kantenmengen in Maskenreihenfolge (Kante e = Bit e): X (1 = zusammenhängend) und Zahl haltender Kanten k."""
    m = len(edges)
    shifts = np.arange(m, dtype=np.uint32)
    xs, ks = [], []
    for start in range(0, 1 << m, chunk):
        masks = np.arange(start, min(start + chunk, 1 << m), dtype=np.uint32)
        present = ((masks[:, None] >> shifts) & 1).astype(bool)
        xs.append(connected(present, n, edges).astype(np.float64))
        ks.append(present.sum(axis=1))
    return np.concatenate(xs), np.concatenate(ks)


def score_table(n: int, edges) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bewertung s für alle 2^m Kantenmengen (0 bei Ausfall, sonst -(kleinster Grad der Knoten)), dazu X und k."""
    X, k = enumerate_x(n, edges)
    m = len(edges)
    masks = np.arange(1 << m, dtype=np.int64)
    deg = np.zeros((1 << m, n), dtype=np.int16)
    for e, (u, v) in enumerate(edges):
        up = ((masks >> e) & 1).astype(np.int16)
        deg[:, u] += up
        deg[:, v] += up
    s = np.where(X == 0, 0, -deg.min(axis=1)).astype(np.int16)
    return s, X, k


def failure_counts(X: np.ndarray, k: np.ndarray, m: int) -> np.ndarray:
    """N_f (f = 0..m): Zahl der Kantenmengen mit f ausgefallenen Kanten, bei denen das Netz NICHT zusammenhängt."""
    return np.bincount((m - k)[X == 0], minlength=m + 1).astype(float)


# --- Exakte Rechnung ----------------------------------------------------------------------------------------------------------------------

def exact_is(Nf: np.ndarray, q: float, qp: float) -> dict:
    """Exakt: P, zweites Moment M2 und CV^2 = (M2 - P^2) / P^2 je Ziehung bei Kippung q' (q' = q: einfaches Ziehen); unendlich bei q' = 0 oder 1, NaN bei P = 0."""
    m = len(Nf) - 1
    f = np.arange(m + 1)
    pq = q**f * (1.0 - q) ** (m - f)
    P = float(np.sum(Nf * pq))
    if P <= 0:
        return {"P": 0.0, "M2": float("nan"), "cv2": float("nan")}
    if qp <= 0 or qp >= 1:
        return {"P": P, "M2": float("inf"), "cv2": float("inf")}
    pqp = qp**f * (1.0 - qp) ** (m - f)
    M2 = float(np.sum(Nf * pq**2 / pqp))
    return {"P": P, "M2": M2, "cv2": (M2 - P**2) / P**2}


def optimal_qp(Nf: np.ndarray, q: float) -> float:
    """Das q' mit kleinstem CV^2 (Gittersuche in zwei Stufen); NaN, wenn der Ausfall unmöglich ist."""
    if exact_is(Nf, q, q)["P"] <= 0:
        return float("nan")
    grid = np.concatenate([np.geomspace(1e-4, 0.5, 600), np.linspace(0.5, 0.999, 300)])
    best = min(grid, key=lambda x: exact_is(Nf, q, float(x))["cv2"])
    fine = np.linspace(best * 0.7, min(best * 1.4, 0.999), 400)
    return float(min(fine, key=lambda x: exact_is(Nf, q, float(x))["cv2"]))


_LEVEL_CACHE: dict = {}


def level_counts(s_tab: np.ndarray, k: np.ndarray, m: int) -> dict[int, np.ndarray]:
    """Je Niveau l (jede ganze Zahl von der kleinsten Bewertung bis 0): die Zahl N_f(l) der Kantenmengen mit s >= l und f ausgefallenen Kanten (f = 0..m). Wird je Tabelle einmal berechnet und gemerkt."""
    key = (s_tab.ctypes.data, s_tab.shape[0], m)
    hit = _LEVEL_CACHE.get(key)
    if hit is not None and hit[0] is s_tab:
        return hit[2]
    f = (m - k).astype(np.int64)
    counts = {lvl: np.bincount(f[s_tab >= lvl], minlength=m + 1).astype(float) for lvl in range(int(s_tab.min()), 1)}
    if len(_LEVEL_CACHE) > 8:
        _LEVEL_CACHE.clear()
    _LEVEL_CACHE[key] = (s_tab, k, counts)
    return counts


def _level_prob(cnt: np.ndarray, q: float) -> tuple[float, float]:
    """P_q(Niveau) und E_q[f 1(Niveau)] aus den Zählungen N_f(l)."""
    m = len(cnt) - 1
    f = np.arange(m + 1)
    w = cnt * q**f * (1.0 - q) ** (m - f)
    return float(w.sum()), float((w * f).sum())


def level_stats(s_tab: np.ndarray, k: np.ndarray, m: int, q: float) -> list[dict]:
    """Je Niveau l <= 0 (aufsteigend): P_q(s >= l) und E_q[f | s >= l] / m (= das Ziel des Updates, wenn die Elite {s >= l} ist). Nur Niveaus, die vorkommen."""
    out = []
    for lvl, cnt in sorted(level_counts(s_tab, k, m).items()):
        P, Ef = _level_prob(cnt, q)
        if P > 0:
            out.append({"level": lvl, "P": P, "target": Ef / (m * P)})
    return out


def prob_at(s_tab: np.ndarray, k: np.ndarray, m: int, qp: float, level: int) -> float:
    """P_{q'}(s >= level): Wahrscheinlichkeit des Niveaus unter der Kippung q'."""
    return _level_prob(level_counts(s_tab, k, m)[level], qp)[0]


def ce_ladder(s_tab: np.ndarray, k: np.ndarray, m: int, q: float, rho: float, max_stages: int = C.MAX_STAGES) -> dict:
    """Deterministische Stufenfolge der Kreuzentropie (Erwartungswerte statt Stichprobe): je Stufe das höchste Niveau, dessen Wahrscheinlichkeit unter dem aktuellen q' mindestens rho beträgt,
    und das Update-Ziel E_q[f | s >= Niveau] / m. Rückgabe: Stufen (q' vorher, Niveau, Wahrscheinlichkeit des Niveaus, q' nachher), erreicht = Niveau 0 erreicht, Fixpunkt = q' bleibt stehen."""
    stats = {d["level"]: d for d in level_stats(s_tab, k, m, q)}
    levels = sorted(stats)
    qp = q
    stages = []
    for _ in range(max_stages):
        ok = [lv for lv in levels if prob_at(s_tab, k, m, qp, lv) >= rho]
        lvl = max(ok) if ok else levels[0]
        new = stats[lvl]["target"]
        stages.append({"qp_before": qp, "level": lvl, "prob": prob_at(s_tab, k, m, qp, lvl), "qp_after": new})
        if lvl >= 0:
            return {"stages": stages, "reached": True, "fixpoint": False}
        if abs(new - qp) < 1e-9 or (len(stages) > 1 and abs(new - stages[-2]["qp_after"]) < 1e-9 and lvl == stages[-2]["level"]):
            return {"stages": stages, "reached": False, "fixpoint": True}
        qp = new
    return {"stages": stages, "reached": False, "fixpoint": False}


def splitting_ladder(s_tab: np.ndarray, k: np.ndarray, m: int, q: float, rho: float) -> dict:
    """Deterministische Stufenfolge des Splittings: Stufe 1 aus dem Original, dann bedingte Wahrscheinlichkeiten p_t = P(s >= l_t | s >= l_{t-1}). Je Stufe das höchste Niveau mit
    bedingter Wahrscheinlichkeit >= rho; liegt keines über dem aktuellen Niveau, springt die Stufe zum nächsthöheren vorhandenen. Rückgabe: Stufenwahrscheinlichkeiten und die relative Varianz
    Summe (1 - p_t) / p_t je Teilchen (mal 1/N): rel_var(N) = factor / N."""
    stats = {d["level"]: d for d in level_stats(s_tab, k, m, q)}
    levels = sorted(stats)
    cur = levels[0]
    probs, chosen = [], []
    while cur < 0:
        higher = [lv for lv in levels if lv > cur]
        ok = [lv for lv in higher if stats[lv]["P"] / stats[cur]["P"] >= rho]
        nxt = max(ok) if ok else higher[0]
        probs.append(stats[nxt]["P"] / stats[cur]["P"])
        chosen.append(nxt)
        cur = nxt
    if levels[0] >= 0 or not probs:
        probs, chosen = [stats[0]["P"]], [0]
    # Stufe 1 aus dem Original: ihre Wahrscheinlichkeit ist die des ersten gewählten Niveaus
    factor = float(sum((1 - p) / p for p in probs))
    return {"probs": probs, "levels": chosen, "factor": factor}


# --- Kreuzentropie ------------------------------------------------------------------------------------------------------------------------

def likelihood_ratio(f: np.ndarray, m: int, q: float, qp: float) -> np.ndarray:
    """Dichtequotient Original / Hilfsverteilung für Ziehungen mit f Ausfällen."""
    return (q / qp) ** f * ((1.0 - q) / (1.0 - qp)) ** (m - f)


def ce_update(f_elite: np.ndarray, m: int, q: float, qp: float) -> float:
    """Gewichtetes Maximum-Likelihood-Update: q'_neu = Summe W f / (m Summe W) über die Elite."""
    w = likelihood_ratio(f_elite, m, q, qp)
    return float(np.sum(w * f_elite) / (m * np.sum(w)))


def ce_learn(s_tab: np.ndarray, m: int, q: float, N: int, rho: float, key: tuple, max_stages: int = C.MAX_STAGES) -> dict:
    """Lernt q' mit der Kreuzentropie. Rückgabe: Verlauf der q' (erstes = q), Schwellen je Stufe, festgefahren (Ereignisniveau nicht erreicht), Zahl der Ziehungen."""
    qp = q
    traj = [q]
    thrs: list[int] = []
    evals = 0
    stuck = False
    for t in range(max_stages):
        mask, f = draw_masks(stream_seed(*key, t), N, m, qp)
        evals += N
        s = s_tab[mask]
        kth = int(np.ceil(rho * N))
        thr = min(int(np.sort(s)[::-1][max(kth, 1) - 1]), 0)
        elite = s >= thr
        if elite.all() and thr < 0:
            stuck = True                                          # alle Ziehungen sind Elite: die Stufe unterscheidet nichts, q' bliebe gleich
            break
        qp = ce_update(f[elite].astype(float), m, q, qp)
        traj.append(qp)
        thrs.append(thr)
        if thr >= 0:
            break
    reached = bool(thrs) and thrs[-1] >= 0
    return {"traj": traj, "thrs": thrs, "stuck": stuck or not reached, "evals": evals}


def is_estimate(s_tab: np.ndarray, m: int, q: float, qp: float, N: int, key: tuple) -> float:
    """Importance-Sampling-Schätzer mit N frischen Ziehungen bei Kippung q' (unverzerrt)."""
    mask, f = draw_masks(stream_seed(*key, 999), N, m, qp)
    w = np.where(s_tab[mask] == 0, likelihood_ratio(f, m, q, qp), 0.0)
    return float(w.mean())


def gibbs_sweep(mask: np.ndarray, uu: np.ndarray, s_tab: np.ndarray, thr: int, q: float) -> np.ndarray:
    """Ein Gibbs-Durchlauf über alle Kanten (in der Reihenfolge 0..m-1): jede Kante wird aus ihrer Originalverteilung neu gezogen (hält, wenn die Zufallszahl uu[:, e] mindestens q ist) und die neue
    Menge angenommen, wenn die Bewertung die Schwelle hält. Lässt die bedingte Originalverteilung auf {s >= thr} unverändert."""
    m = uu.shape[1]
    for e in range(m):
        newbit = (uu[:, e] >= q).astype(np.int64)
        new = (mask & ~(1 << e)) | (newbit << e)
        mask = np.where(s_tab[new] >= thr, new, mask)
    return mask


# --- Adaptives Splitting ------------------------------------------------------------------------------------------------------------------

def splitting(s_tab: np.ndarray, m: int, q: float, N: int, rho: float, sweeps: int, key: tuple, max_stages: int = C.MAX_STAGES) -> dict:
    """Adaptives Fixed-Effort-Splitting. N Teilchen aus dem Original; Schwelle = s der ceil(rho N)-besten (höchstens 0), bei diskreter Bewertung zum nächsthöheren vorhandenen Niveau,
    wenn alle überleben; p_t = Anteil der Überlebenden; N Teilchen mit Zurücklegen; `sweeps` Gibbs-Durchläufe je Teilchen (jede Kante aus der Originalverteilung neu ziehen, annehmen,
    wenn s >= Schwelle). Schätzung = Produkt der p_t. Ziehungen: N plus N je Gibbs-Durchlauf."""
    mask, _ = draw_masks(stream_seed(*key, 0), N, m, q)
    p = 1.0
    evals = N
    for t in range(max_stages):
        s = s_tab[mask]
        kth = int(np.ceil(rho * N))
        thr = min(int(np.sort(s)[::-1][max(kth, 1) - 1]), 0)
        if (s >= thr).all() and thr < 0:
            higher = np.unique(s[s > thr])
            if len(higher) == 0:
                return {"est": float("nan"), "stages": t, "evals": evals, "stuck": True}
            thr = int(higher[0])
        surv = mask[s >= thr]
        p *= len(surv) / N
        if thr >= 0:
            return {"est": p, "stages": t + 1, "evals": evals, "stuck": False}
        u = uniforms(stream_seed(*key, 100 + t), N)
        mask = surv[np.minimum((u * len(surv)).astype(np.int64), len(surv) - 1)]
        for sw in range(sweeps):
            uu = uniforms(stream_seed(*key, 1000 + 10 * t + sw), N * m).reshape(N, m)
            mask = gibbs_sweep(mask, uu, s_tab, thr, q)
            evals += N
    return {"est": float("nan"), "stages": max_stages, "evals": evals, "stuck": True}
