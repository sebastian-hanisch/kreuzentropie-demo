"""Kern von Hand geprüft: Zufallsstrom, Bewertung am Pfad, Update mit Brüchen, Niveaus und Stufenfolge, Gibbs-Schritt, Splitting an kleinen Fällen."""

from fractions import Fraction

import numpy as np
import pytest

import ken_algorithm as A


def test_uniforms_gegen_die_ganzzahl_referenz():
    seed = A.stream_seed(3, 7)
    soll = [(A.mix_int((seed + i * A.GAMMA) & A.MASK) >> 11) * (1.0 / (1 << 53)) for i in range(1, 6)]
    assert A.uniforms(seed, 5).tolist() == soll


def test_ziehungen_bitpackung_von_hand():
    seed, N, m, qp = 11, 3, 4, 0.3
    u = A.uniforms(seed, N * m).reshape(N, m)
    mask, f = A.draw_masks(seed, N, m, qp)
    for i in range(N):
        assert f[i] == int(np.sum(u[i] < qp))
        assert mask[i] == sum(int(u[i, e] >= qp) << e for e in range(m))                 # Bit e = Kante e hält


def test_bewertung_am_pfad_von_hand(pfad):
    netz, (s, X, k, Nf) = pfad
    # Masken: Bit 0 = Kante 0-1, Bit 1 = Kante 1-2. Beide halten (3): Grad (1,2,1), Minimum 1, Bewertung -1; alle anderen: Ausfall, Bewertung 0.
    assert s.tolist() == [0, 0, 0, -1] and X.tolist() == [0, 0, 0, 1] and Nf.tolist() == [0.0, 2.0, 1.0]


def test_niveaumengen_sind_geschachtelt_und_das_ereignis_ist_s_gleich_null(gitter34, hantel):
    for netz, (s, X, k, Nf) in (gitter34, hantel):
        assert ((s == 0) == (X == 0)).all() and set(np.unique(s)) <= {-3, -2, -1, 0}
        assert (s >= -1).sum() >= (s >= 0).sum() > 0


def test_ce_update_von_hand_mit_bruechen():
    # Gewichte 1 (q' = q): N = 4 Ziehungen mit f = 0, 1, 1, 2; Elite f >= 1: (1 + 1 + 2) / (2 * 3) = 2/3
    assert A.ce_update(np.array([1.0, 1.0, 2.0]), 2, 0.5, 0.5) == pytest.approx(2 / 3)
    # q = 1/2, q' = 3/4: W(1) = 4/3, W(2) = 4/9; Elite f = 1 und 2: (4/3 + 2 * 4/9) / (2 (4/3 + 4/9)) = 5/8
    q, qp, m = Fraction(1, 2), Fraction(3, 4), 2
    w = lambda f: (q / qp) ** f * ((1 - q) / (1 - qp)) ** (m - f)
    assert float((w(1) * 1 + w(2) * 2) / (m * (w(1) + w(2)))) == pytest.approx(5 / 8)
    assert A.ce_update(np.array([1.0, 2.0]), 2, 0.5, 0.75) == pytest.approx(5 / 8)
    assert A.likelihood_ratio(np.array([1, 2]), 2, 0.5, 0.75).tolist() == pytest.approx([4 / 3, 4 / 9])


def test_niveaus_am_pfad_von_hand(pfad):
    netz, (s, X, k, Nf) = pfad
    q = 0.5
    st = {d["level"]: d for d in A.level_stats(s, k, netz.m, q)}
    assert set(st) == {-1, 0}
    assert st[-1]["P"] == pytest.approx(1.0) and st[0]["P"] == pytest.approx(0.75)
    assert st[0]["target"] == pytest.approx(1 / (2 - q))                                  # E[f | Ausfall] / m = 2/3
    assert st[-1]["target"] == pytest.approx(q)                                            # alle Ziehungen: der Originalanteil q
    assert A.prob_at(s, k, netz.m, 0.9, 0) == pytest.approx(1 - 0.1**2)


def test_stufenfolge_am_pfad_von_hand(pfad):
    netz, (s, X, k, Nf) = pfad
    lad = A.ce_ladder(s, k, netz.m, 0.5, 0.5)
    assert lad["reached"] and len(lad["stages"]) == 1 and lad["stages"][0]["level"] == 0 and lad["stages"][0]["qp_after"] == pytest.approx(2 / 3)
    lad2 = A.ce_ladder(s, k, netz.m, 0.5, 0.9)                                             # rho über P(Ausfall) = 0,75: Elite = alle: Fixpunkt bei q
    assert not lad2["reached"] and lad2["fixpoint"] and lad2["stages"][0]["qp_after"] == pytest.approx(0.5)
    sp = A.splitting_ladder(s, k, netz.m, 0.5, 0.5)
    assert sp["probs"] == pytest.approx([0.75]) and sp["factor"] == pytest.approx(1 / 3)


def test_zweites_moment_am_pfad_wie_in_stueck_sieben(pfad):
    netz, (s, X, k, Nf) = pfad
    e = A.exact_is(Nf, 0.5, 0.75)
    assert e["P"] == pytest.approx(0.75) and e["M2"] == pytest.approx(7 / 9)
    assert A.optimal_qp(Nf, 0.5) == pytest.approx(0.5, abs=0.2)


def test_gibbs_schritt_von_hand(dreieck):
    netz, (s, X, k, Nf) = dreieck
    # Dreieck (3 Kanten): Ausfall (s = 0) bei höchstens einer haltenden Kante. Schwelle 0: nur Ausfallmengen sind erlaubt.
    mask = np.array([0b001])                                                               # genau eine Kante hält (Ausfall)
    q = 0.4
    uu = np.array([[0.1, 0.9, 0.9]])                                                      # Kante 0: Zufallszahl < q -> fällt aus; neue Menge 0b000 (Ausfall) angenommen
    # Schritt für Schritt: Kante 0 -> 0b000; Kante 1: Zufallszahl 0,9 >= q -> hält: 0b010 (eine Kante hält: Ausfall, angenommen); Kante 2: hält: 0b110 (zwei halten: zusammenhängend: abgelehnt)
    assert A.gibbs_sweep(mask, uu, s, 0, q).tolist() == [0b010]


def test_splitting_beim_pfad_gibt_stufenwahrscheinlichkeit_und_erwartungstreue(pfad):
    netz, (s, X, k, Nf) = pfad
    q = 0.3
    ests = [A.splitting(s, netz.m, q, 200, 0.5, 2, (1, rep))["est"] for rep in range(300)]
    P = 1 - (1 - q) ** 2
    z = (np.mean(ests) - P) / (np.std(ests, ddof=1) / np.sqrt(len(ests)))
    assert abs(z) < 4.0 and A.splitting(s, netz.m, q, 50, 0.5, 2, (1, 0))["stages"] >= 1


def test_ce_lernen_am_pfad_erreicht_das_ereignis_in_einer_stufe(pfad):
    netz, (s, X, k, Nf) = pfad
    r = A.ce_learn(s, netz.m, 0.5, 400, 0.5, (2, 1))
    assert not r["stuck"] and len(r["thrs"]) == 1 and r["thrs"] == [0] and r["traj"][-1] == pytest.approx(2 / 3, abs=0.06) and r["evals"] == 400
    r2 = A.ce_learn(s, netz.m, 0.5, 400, 0.9, (2, 1))
    assert r2["stuck"]                                                                      # Elite = alle Ziehungen
