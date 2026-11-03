"""Eigenschaftstests (ORACLE-PLAYBOOK §5): Aussagen, die für jede zulässige Eingabe gelten müssen."""

import numpy as np
import pytest

import ken_algorithm as A
import ken_evaluation as ev


def test_update_ziele_wachsen_mit_dem_niveau(gitter34, hantel):
    for netz, (s, X, k, Nf) in (gitter34, hantel):
        for q in (0.1, 0.01, 0.001):
            stats = A.level_stats(s, k, netz.m, q)
            targets = [d["target"] for d in stats]
            assert all(a <= b + 1e-12 for a, b in zip(targets, targets[1:]))              # höheres Niveau: mehr Ausfälle in der Elite
            assert targets[0] == pytest.approx(q, abs=1e-9) if stats[0]["P"] > 0.999999 else True
            assert stats[-1]["level"] == 0


def test_ziel_des_ereignisniveaus_ist_nahe_dem_optimalen_qp(gitter34, hantel):
    for netz, (s, X, k, Nf) in (gitter34, hantel):
        q = 0.001
        ziel = A.level_stats(s, k, netz.m, q)[-1]["target"]
        qo = A.optimal_qp(Nf, q)
        assert ziel == pytest.approx(qo, rel=0.15)
        assert A.exact_is(Nf, q, ziel)["cv2"] <= 1.05 * A.exact_is(Nf, q, qo)["cv2"]      # das gelernte q' ist bis auf wenige Prozent so gut wie das Optimum


def test_gelerntes_qp_ist_nie_besser_als_das_optimum(gitter34):
    netz, (s, X, k, Nf) = gitter34
    q = 0.01
    opt = A.exact_is(Nf, q, A.optimal_qp(Nf, q))["cv2"]
    for qp in np.linspace(0.03, 0.4, 25):
        assert A.exact_is(Nf, q, float(qp))["cv2"] >= opt - 1e-9


def test_stufenfolge_erreicht_das_ereignis_genau_unter_der_rho_grenze(gitter34):
    netz, (s, X, k, Nf) = gitter34
    q = 0.01
    grenze = ev.rho_reach_limit(s, k, netz.m, q)
    assert grenze == pytest.approx(0.021, abs=0.003)
    q1 = A.ce_ladder(s, k, netz.m, q, 0.02)["stages"][0]["qp_after"]
    assert A.prob_at(s, k, netz.m, q1, 0) == pytest.approx(0.0225, abs=5e-4)
    assert A.ce_ladder(s, k, netz.m, q, 0.02)["reached"] and not A.ce_ladder(s, k, netz.m, q, 0.03)["reached"]


def test_erste_stufe_braucht_rho_unter_der_wahrscheinlichkeit_des_ersten_niveaus(gitter34):
    netz, (s, X, k, Nf) = gitter34
    for q, p1 in ((0.01, 0.0776), (0.001, 0.0080)):
        assert A.ce_ladder(s, k, netz.m, q, p1 * 0.9)["reached"] or A.ce_ladder(s, k, netz.m, q, p1 * 0.9)["fixpoint"]
        lad = A.ce_ladder(s, k, netz.m, q, p1 * 1.2)
        assert not lad["reached"] and lad["stages"][0]["level"] == -2                       # keine Elite, alle Ziehungen


def test_festgefahrene_laeufe_haben_eine_elite_aus_allen_ziehungen_oder_bleiben_unter_null(gitter34):
    netz, (s, X, k, Nf) = gitter34
    r = A.ce_learn(s, netz.m, 0.001, 300, 0.1, (41, 1))
    assert r["stuck"] and r["traj"] == [0.001] and r["thrs"] == [] and r["evals"] == 300
    r2 = A.ce_learn(s, netz.m, 0.01, 1000, 0.05, (41, 2))
    assert r2["stuck"] and len(r2["thrs"]) == 8 and all(t == -1 for t in r2["thrs"])


def test_splitting_ist_bei_feiner_stufe_besser_als_bei_grober(gitter34):
    netz, (s, X, k, Nf) = gitter34
    q = 0.01
    P = A.exact_is(Nf, q, q)["P"]
    fein = A.splitting_ladder(s, k, netz.m, q, 0.05)["factor"]
    assert fein > 100                                                                   # nur zwei Stufen, die zweite mit p = 0,0054: Faktor (1 - p) / p ~ 185
    est = np.array([A.splitting(s, netz.m, q, 1000, 0.05, 2, (42, rep))["est"] for rep in range(80)])
    assert abs(np.nanmean(est) / P - 1) < 0.25


def test_kreuzentropie_schlaegt_einfaches_ziehen_bei_gleichem_budget(gitter34):
    netz, (s, X, k, Nf) = gitter34
    table = (s, X, k, Nf)
    settings = ev.Settings("3x4", 0.01, 1000, 0.01, 6000, 35)
    ex = ev.budget_compare(settings, netz, table)
    by = {r["method"]: r for r in ex["rows"]}
    assert by["Kreuzentropie"]["rmse"] < 0.4 * by["einfach"]["rmse"] and by["Orakel (festes Optimum)"]["rmse"] < by["Kreuzentropie"]["rmse"] * 1.1
    assert by["Kreuzentropie"]["rmse"] < by["adaptives Splitting"]["rmse"]


def test_gleiche_staerke_gleiche_schaetzung_und_anderer_strom_andere(gitter34):
    netz, (s, X, k, Nf) = gitter34
    a = A.ce_learn(s, netz.m, 0.01, 500, 0.01, (43, 1))
    b = A.ce_learn(s, netz.m, 0.01, 500, 0.01, (43, 1))
    c = A.ce_learn(s, netz.m, 0.01, 500, 0.01, (43, 2))
    assert a["traj"] == b["traj"] and a["traj"] != c["traj"]
