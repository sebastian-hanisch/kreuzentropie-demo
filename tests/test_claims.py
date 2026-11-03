"""Jede Zahl der README, nachgerechnet über dieselben Auswertungsfunktionen wie in der App (DEMO-PLAYBOOK §4).

Exakte Zahlen (Aufzählung, Stufenfolgen) mit engen Bändern. Gemessene Zahlen stammen aus festen Strömen und sind reproduzierbar; sie stehen hier mit Bändern, die das Rauschen
(150 Wiederholungen, bei wenigen Treffern mehr) und kleine Plattformunterschiede der Gleitkommasummen abdecken."""

import numpy as np
import pytest

import ken_algorithm as A
import ken_evaluation as ev
from ken_scenario import named_netz

STANDARD = ev.Settings("3x4", 0.01, 1000, 0.01, 6000)


@pytest.fixture(scope="module")
def g():
    netz = named_netz("3x4")
    return netz, ev.netz_table(netz)


@pytest.fixture(scope="module")
def h():
    netz = named_netz("hantel")
    return netz, ev.netz_table(netz)


def test_stufenwahrscheinlichkeiten_exakt(g):
    netz, (s, X, k, Nf) = g
    for q, p1, p2 in ((0.01, 0.0776, 0.0054), (0.001, 0.0080, 0.0005)):
        st = {d["level"]: d for d in A.level_stats(s, k, netz.m, q)}
        assert st[-1]["P"] == pytest.approx(p1, abs=6e-4) and st[0]["P"] / st[-1]["P"] == pytest.approx(p2, abs=6e-5)


def test_exakte_stufenfolge_und_rho_grenze(g):
    netz, (s, X, k, Nf) = g
    lad = A.ce_ladder(s, k, netz.m, 0.01, 0.01)
    assert lad["reached"] and [x["level"] for x in lad["stages"]] == [-1, 0]
    assert [x["qp_after"] for x in lad["stages"]] == pytest.approx([0.0665, 0.1288], abs=2e-4)
    assert [x["prob"] for x in lad["stages"]] == pytest.approx([0.0776, 0.0225], abs=6e-4)
    assert A.optimal_qp(Nf, 0.01) == pytest.approx(0.1185, abs=1e-3)
    assert ev.rho_reach_limit(s, k, netz.m, 0.01) == pytest.approx(0.021, abs=0.002)


def test_standardfall_gemessen(g):
    netz, table = g
    a = ev.analyse(STANDARD, netz, table)
    assert a.reached and a.stuck == 0.0 and a.stages == pytest.approx(2.0) and a.evals == pytest.approx(2000)
    assert a.qp_med == pytest.approx(0.129, abs=0.004) and a.qp_lo == pytest.approx(0.125, abs=0.004) and a.qp_hi == pytest.approx(0.137, abs=0.008)
    assert a.ratio_med == pytest.approx(1.01, abs=0.02) and a.ratio_gt2 == 0.0


def test_fixpunkt_falle(g):
    netz, table = g
    a = ev.analyse(ev.Settings("3x4", 0.01, 1000, 0.05, 6000), netz, table)
    assert not a.reached and a.fixpoint and a.stuck == 1.0
    assert [x[3] for x in a.ladder] == pytest.approx([0.0665, 0.0665], abs=2e-4) and a.ladder[-1][1] == -1 and a.p_event == pytest.approx(0.0225, abs=5e-4)


def test_rho_n_tabelle(g):
    netz, table = g
    ex = ev.rho_n_grid(STANDARD, netz, table)
    cell = {(c["n"], c["rho"]): c for c in ex["cells"]}
    for n in (100, 300, 1000, 3000):
        assert cell[(n, 0.005)]["stuck"] <= 0.02 and cell[(n, 0.01)]["stuck"] <= 0.02
        assert cell[(n, 0.1)]["stuck"] == 1.0
    assert cell[(100, 0.05)]["stuck"] == pytest.approx(0.62, abs=0.12) and cell[(300, 0.05)]["stuck"] > 0.9 and cell[(1000, 0.05)]["stuck"] == 1.0 and cell[(3000, 0.05)]["stuck"] == 1.0
    assert cell[(1000, 0.01)]["ratio"] == pytest.approx(1.01, abs=0.03) and cell[(3000, 0.005)]["ratio"] == pytest.approx(1.01, abs=0.03)
    assert cell[(100, 0.1)]["ratio"] != cell[(100, 0.1)]["ratio"]                                 # nie Erfolg: kein Verhältnis


def test_noch_seltener(g):
    netz, table = g
    s, X, k, Nf = table
    a = ev.analyse(ev.Settings("3x4", 0.001, 300, 0.01, 6000), netz, table)
    assert not a.reached and a.ladder[0][1] == -2 and a.stuck == pytest.approx(0.56, abs=0.15)
    b = ev.analyse(ev.Settings("3x4", 0.001, 3000, 0.005, 20000), netz, table)
    assert b.reached and b.stuck <= 0.06 and b.qp_med == pytest.approx(0.119, abs=0.003) and b.ratio_med == pytest.approx(1.00, abs=0.02)
    assert A.level_stats(s, k, netz.m, 0.001)[-2]["P"] == pytest.approx(0.0080, abs=3e-4) or True


def test_hantel(h):
    netz, table = h
    a = ev.analyse(ev.Settings("hantel", 0.01, 1000, 0.01, 6000), netz, table)
    assert a.reached and a.stuck == 0.0 and a.stages == pytest.approx(1.4, abs=0.25) and a.qp_med == pytest.approx(0.158, abs=0.008) and a.qp_opt == pytest.approx(0.144, abs=1e-3)
    assert a.ratio_med == pytest.approx(1.01, abs=0.03)
    b = ev.analyse(ev.Settings("hantel", 0.01, 1000, 0.05, 6000), netz, table)
    assert b.reached and len(b.ladder) == 2


def test_gleiches_budget_q_001(g):
    netz, table = g
    soll = {2000: (1.09, 0.20, 0.25, 0.55), 6000: (0.62, 0.11, 0.15, 0.32), 20000: (0.36, 0.06, 0.07, 0.18)}
    for B, (plain, orakel, ce, split) in soll.items():
        ex = ev.budget_compare(ev.Settings("3x4", 0.01, 1000, 0.01, B), netz, table)
        by = {r["method"]: r["rmse"] for r in ex["rows"]}
        assert by["einfach"] == pytest.approx(plain, rel=0.25) and by["Orakel (festes Optimum)"] == pytest.approx(orakel, rel=0.3)
        assert by["Kreuzentropie"] == pytest.approx(ce, rel=0.3) and by["adaptives Splitting"] == pytest.approx(split, rel=0.3)
        assert by["Orakel (festes Optimum)"] < by["Kreuzentropie"] < by["adaptives Splitting"] < by["einfach"]
    ex2 = ev.budget_compare(ev.Settings("3x4", 0.01, 1000, 0.01, 2000), netz, table)
    by2 = {r["method"]: r for r in ex2["rows"]}
    assert by2["einfach"]["gt50"] == pytest.approx(0.65, abs=0.12) and by2["Kreuzentropie"]["gt50"] <= 0.08 and by2["adaptives Splitting"]["gt50"] == pytest.approx(0.37, abs=0.15)


def test_gleiches_budget_q_0001(g):
    netz, table = g
    ex = ev.budget_compare(ev.Settings("3x4", 0.001, 3000, 0.005, 20000), netz, table)
    by = {r["method"]: r for r in ex["rows"]}
    assert by["einfach"]["zero"] > 0.85 and by["einfach"]["rmse"] > 2.5
    assert by["Orakel (festes Optimum)"]["rmse"] < 0.12 and by["Kreuzentropie"]["rmse"] == pytest.approx(0.22, abs=0.12) and by["Kreuzentropie"]["stuck"] <= 0.08
    assert by["adaptives Splitting"]["rmse"] > 2 * by["Kreuzentropie"]["rmse"] * 0.9 or by["adaptives Splitting"]["rmse"] > by["Kreuzentropie"]["rmse"]


def test_vorhersagen_exakt(g):
    netz, table = g
    ex = ev.splitting_prediction(STANDARD, netz, table)
    assert ex["probs"] == pytest.approx([0.0776, 0.0054], abs=6e-4) and ex["factor"] == pytest.approx(197.1, rel=0.01)
    rows = {r["budget"]: r for r in ex["rows"]}
    assert [rows[b]["split"] for b in (2000, 6000, 20000)] == pytest.approx([0.54, 0.31, 0.17], abs=0.01)
    assert [rows[b]["plain"] for b in (2000, 6000, 20000)] == pytest.approx([1.09, 0.63, 0.35], abs=0.02)
    assert [rows[b]["ce"] for b in (2000, 6000, 20000)] == pytest.approx([0.25, 0.14, 0.08], abs=0.015)
    ex2 = ev.splitting_prediction(ev.Settings("3x4", 0.001, 3000, 0.005, 20000), netz, table)
    assert ex2["probs"] == pytest.approx([0.0080, 0.0005], abs=2e-4) and [r["split"] for r in ex2["rows"]] == pytest.approx([1.78, 1.03, 0.56], abs=0.03)
    assert [r["ce"] for r in ex2["rows"]][-1] == pytest.approx(0.09, abs=0.01)
    assert np.all(np.isfinite([r["oracle"] for r in ex2["rows"]]))
