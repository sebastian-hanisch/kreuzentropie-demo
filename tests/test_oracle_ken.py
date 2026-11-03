"""Orakel-Tests: andere Rechenwege als die der Demo.

- Bewertung (kleinster Grad, Ausfall) gegen direkte Schleifen und networkx auf allen Kantenmengen kleiner Netze.
- Niveauwahrscheinlichkeiten und Update-Ziele gegen eine Summation mit Brüchen über alle Kantenmengen.
- Exakte Stufenfolge gegen die Simulation des Lernens (gelerntes q' im Mittel gleich dem Ziel des Niveaus).
- Gibbs-Durchlauf lässt die bedingte Originalverteilung auf der Niveaumenge invariant (exakte Verteilungen auf einem Mini-Netz).
- Splitting: Stufenwahrscheinlichkeiten und relative Varianz gegen Σ (1 − p_t)/(p_t N); Erwartungstreue.
"""

from fractions import Fraction

import numpy as np
import pytest

import ken_algorithm as A
import ken_evaluation as ev
from ken_scenario import grid_netz, hantel_netz

nx = pytest.importorskip("networkx")


def score_direkt(netz, mask):
    g = nx.Graph()
    g.add_nodes_from(range(netz.n))
    g.add_edges_from(e for i, e in enumerate(netz.edges) if (mask >> i) & 1)
    if not nx.is_connected(g):
        return 0
    return -min(d for _, d in g.degree())


@pytest.mark.parametrize("netz", [hantel_netz(), grid_netz(2, 3), grid_netz(3, 3)], ids=["hantel", "2x3", "3x3"])
def test_bewertung_gegen_networkx_auf_allen_kantenmengen(netz):
    s, X, k, Nf = ev.netz_table(netz)
    for mask in range(1 << netz.m):
        assert s[mask] == score_direkt(netz, mask), mask


@pytest.mark.parametrize("netz", [hantel_netz(), grid_netz(2, 3)], ids=["hantel", "2x3"])
@pytest.mark.parametrize("q", [Fraction(1, 10), Fraction(1, 100)])
def test_niveaus_und_updateziele_gegen_summation_mit_bruechen(netz, q):
    s, X, k, Nf = ev.netz_table(netz)
    m = netz.m
    for st in A.level_stats(s, k, m, float(q)):
        P = Fraction(0)
        F = Fraction(0)
        for mask in range(1 << m):
            if score_direkt(netz, mask) >= st["level"]:
                f = m - bin(mask).count("1")
                pr = q**f * (1 - q) ** (m - f)
                P += pr
                F += pr * f
        assert st["P"] == pytest.approx(float(P), rel=1e-12)
        assert st["target"] == pytest.approx(float(F / (m * P)), rel=1e-12)


def test_gelerntes_qp_im_mittel_gleich_dem_ziel_der_stufenfolge(gitter34):
    netz, (s, X, k, Nf) = gitter34
    q, N, rho = 0.01, 1000, 0.01
    lad = A.ce_ladder(s, k, netz.m, q, rho)
    qs = [A.ce_learn(s, netz.m, q, N, rho, (31, rep))["traj"] for rep in range(120)]
    qs = [t for t in qs if len(t) == len(lad["stages"]) + 1]
    assert len(qs) > 100 and lad["reached"]
    for stufe, st in enumerate(lad["stages"], start=1):
        assert np.mean([t[stufe] for t in qs]) == pytest.approx(st["qp_after"], abs=0.006), stufe


def test_fixpunkt_in_der_simulation_gleich_dem_exakten(gitter34):
    netz, (s, X, k, Nf) = gitter34
    q, N, rho = 0.01, 1000, 0.05
    lad = A.ce_ladder(s, k, netz.m, q, rho)
    assert lad["fixpoint"] and not lad["reached"]
    r = A.ce_learn(s, netz.m, q, N, rho, (32, 1))
    assert r["stuck"] and r["thrs"] and all(th == -1 for th in r["thrs"])
    assert np.mean(r["traj"][2:]) == pytest.approx(lad["stages"][-1]["qp_after"], abs=0.004)


def test_exakte_stufenfolge_stimmt_mit_den_zahlen_aus_der_vormessung(gitter34):
    netz, (s, X, k, Nf) = gitter34
    lad = A.ce_ladder(s, k, netz.m, 0.01, 0.01)
    assert [x["qp_after"] for x in lad["stages"]] == pytest.approx([0.0665, 0.1288], abs=2e-4)
    assert [x["level"] for x in lad["stages"]] == [-1, 0] and lad["stages"][1]["prob"] == pytest.approx(0.0225, abs=5e-4)


def exakte_verteilung_nach_einem_durchlauf(netz, s, q, thr):
    """Exakte Verteilung auf den 2^m Kantenmengen nach einem Gibbs-Durchlauf (Kante 0..m-1, jeweils neu aus der Originalverteilung, Annahme bei s >= thr), gestartet in der bedingten Verteilung."""
    m = netz.m
    pi = np.zeros(1 << m)
    for mask in range(1 << m):
        if s[mask] >= thr:
            k = bin(mask).count("1")
            pi[mask] = (1 - q) ** k * q ** (m - k)
    pi /= pi.sum()
    start = pi.copy()
    for e in range(m):
        nxt = np.zeros_like(pi)
        for mask in range(1 << m):
            if pi[mask] == 0:
                continue
            for bit, pb in ((1, 1 - q), (0, q)):
                new = (mask & ~(1 << e)) | (bit << e)
                target = new if s[new] >= thr else mask
                nxt[target] += pi[mask] * pb
        pi = nxt
    return start, pi


@pytest.mark.parametrize("thr", [-1, 0])
def test_gibbs_durchlauf_laesst_die_bedingte_verteilung_invariant(thr):
    netz = hantel_netz()
    s, X, k, Nf = ev.netz_table(netz)
    start, ende = exakte_verteilung_nach_einem_durchlauf(netz, s, 0.2, thr)
    assert np.allclose(start, ende, atol=1e-12)


def test_gibbs_code_stimmt_mit_der_exakten_uebergangsrechnung_ueberein(dreieck):
    netz, (s, X, k, Nf) = dreieck
    q, thr, R = 0.3, 0, 40000
    start, ende = exakte_verteilung_nach_einem_durchlauf(netz, s, q, thr)
    rng_masks = np.array([m_ for m_ in range(1 << netz.m) if s[m_] >= thr])
    mask0 = rng_masks[np.minimum((A.uniforms(5, R) * len(rng_masks)).astype(int), len(rng_masks) - 1)]
    # Startverteilung: gleichverteilt über die Niveaumenge, die Exaktrechnung startet bei der bedingten Originalverteilung: zum Vergleich dieselbe Startverteilung nehmen
    uni = np.zeros(1 << netz.m)
    uni[rng_masks] = 1.0 / len(rng_masks)
    pi = uni.copy()
    for e in range(netz.m):
        nxt = np.zeros_like(pi)
        for mask in range(1 << netz.m):
            if pi[mask] == 0:
                continue
            for bit, pb in ((1, 1 - q), (0, q)):
                new = (mask & ~(1 << e)) | (bit << e)
                nxt[new if s[new] >= thr else mask] += pi[mask] * pb
        pi = nxt
    uu = A.uniforms(6, R * netz.m).reshape(R, netz.m)
    out = A.gibbs_sweep(mask0, uu, s, thr, q)
    emp = np.bincount(out, minlength=1 << netz.m) / R
    assert emp == pytest.approx(pi, abs=0.015)


def test_splitting_stufenwahrscheinlichkeiten_und_relative_varianz(gitter34):
    netz, (s, X, k, Nf) = gitter34
    q, N, R = 0.01, 2000, 200
    sp = A.splitting_ladder(s, k, netz.m, q, 0.05)
    assert sp["probs"] == pytest.approx([0.0776, 0.0054], abs=6e-4) and sp["factor"] == pytest.approx(197.1, rel=0.01)
    P = A.exact_is(Nf, q, q)["P"]
    est = np.array([A.splitting(s, netz.m, q, N, 0.05, 2, (33, rep))["est"] for rep in range(R)])
    assert not np.isnan(est).any()
    assert abs((est.mean() - P) / (est.std(ddof=1) / np.sqrt(R))) < 4.0
    assert (est.std(ddof=1) / P) ** 2 == pytest.approx(sp["factor"] / N, rel=0.35)


def test_is_schaetzer_mit_gelerntem_qp_ist_erwartungstreu(gitter34):
    netz, (s, X, k, Nf) = gitter34
    q = 0.01
    P = A.exact_is(Nf, q, q)["P"]
    est = np.array([A.is_estimate(s, netz.m, q, 0.129, 1000, (34, rep)) for rep in range(400)])
    assert abs((est.mean() - P) / (est.std(ddof=1) / np.sqrt(400))) < 4.0
    assert est.var(ddof=1) == pytest.approx(P**2 * A.exact_is(Nf, q, 0.129)["cv2"] / 1000, rel=0.25)
