"""Randfälle: eigene Netze, unmögliche Ereignisse, ρ und N an den Grenzen, kleinste Größen, Eingabefehler."""

import math

import numpy as np
import pytest

import ken_algorithm as A
import ken_constants as C
import ken_evaluation as ev
from ken_scenario import grid_netz, parse_edges


def _s(**kw):
    return ev.Settings(**kw)


def test_eingabefehler_der_spielwiese():
    with pytest.raises(ValueError, match="Schleife"):
        parse_edges("0-0")
    with pytest.raises(ValueError, match="Keine Kanten"):
        parse_edges("\n# nur Kommentar\n")
    with pytest.raises(ValueError, match="Doppelte"):
        parse_edges("0-1\n1-0")
    with pytest.raises(ValueError, match="erwartet zwei ganze Zahlen"):
        parse_edges("0-a")
    with pytest.raises(ValueError, match="höchstens"):
        parse_edges("\n".join(f"{i}-{i + 1}" for i in range(C.CUSTOM_EDGE_LIMIT + 1)))
    with pytest.raises(ValueError, match="Knoten"):
        parse_edges("\n".join(f"{2 * i}-{2 * i + 1}" for i in range(8)))


def test_pfad_hat_kein_zwischenniveau_und_lernt_in_einer_stufe():
    netz = parse_edges("0-1\n1-2\n2-3")
    a = ev.analyse(_s(q=0.01, n=300, rho=0.01), netz=netz)
    assert a.reached and len(a.ladder) == 1 and a.stuck == 0.0
    assert a.qp_ladder == pytest.approx(a.ladder[0][3]) and a.qp_ladder > 0.3                 # Ausfall durch eine von drei Kanten: Ziel nahe 1/3


def test_alle_netze_und_die_reglergrenzen_laufen_ohne_fehler():
    for netz_key in C.NETZE:
        for q, n, rho in ((C.Q_MIN, C.N_MAX, C.RHO_MIN), (C.Q_MAX, C.N_MIN, C.RHO_MAX)):
            a = ev.analyse(_s(netz=netz_key, q=q, n=n, rho=rho))
            assert math.isfinite(a.P) and 0.0 <= a.stuck <= 1.0 and len(a.ladder) >= 1


def test_unmoegliches_ereignis_gibt_ein_ehrliches_verdikt():
    netz = grid_netz(2, 3)
    X = np.ones(1 << netz.m)
    k = np.array([bin(x).count("1") for x in range(1 << netz.m)])
    s = np.where(X == 0, 0, -1).astype(np.int16)
    Nf = A.failure_counts(X, k, netz.m)
    a = ev.analyse(_s(n=100), netz=netz, table=(s, X, k, Nf))
    assert a.P == 0.0
    kind, text = ev.verdict_main(a, _s())
    assert kind == "info" and "nie aus" in text


def test_kleinste_stufe_und_groesstes_rho_fahren_fest_ohne_fehler():
    netz = grid_netz(3, 4)
    s, X, k, Nf = ev.netz_table(netz)
    r = A.ce_learn(s, netz.m, 0.001, C.N_MIN, C.RHO_MAX, (51, 1))
    assert r["stuck"] and r["traj"] == [0.001]
    sp = A.splitting(s, netz.m, 0.001, 30, 0.5, 2, (51, 2))
    assert sp["stuck"] or math.isfinite(sp["est"])


def test_rho_mal_n_unter_eins_nimmt_die_beste_ziehung():
    netz = grid_netz(3, 4)
    s, X, k, Nf = ev.netz_table(netz)
    r = A.ce_learn(s, netz.m, 0.01, 30, 0.001, (52, 1))                                  # ceil(0,03) = 1: die einzige beste Ziehung bildet die Elite
    assert len(r["traj"]) >= 1 and all(0 < x < 1 for x in r["traj"])


def test_aufzaehlung_ueber_dem_limit_wird_abgelehnt():
    with pytest.raises(ValueError, match="aufgezählt"):
        ev.netz_table(grid_netz(3, 5))


def test_hantel_erreicht_das_ereignis_schon_in_einer_stufe_bei_kleinem_rho():
    a = ev.analyse(_s(netz="hantel", q=0.01, n=1000, rho=0.01))
    assert a.reached and len(a.ladder) == 1 and a.stuck == 0.0
    b = ev.analyse(_s(netz="hantel", q=0.01, n=1000, rho=0.05))
    assert b.reached and len(b.ladder) == 2


def test_experimente_laufen_auf_allen_netzen():
    for netz_key in C.NETZE:
        netz = ev.named_netz(netz_key)
        table = ev.netz_table(netz)
        s = _s(netz=netz_key, n=100, budget=2000)
        assert ev.splitting_prediction(s, netz, table)["rows"] and ev.fixpoint_table(s, netz, table)["rows"]
