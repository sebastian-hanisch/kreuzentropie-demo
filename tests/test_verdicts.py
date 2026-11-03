"""Verdikte (reine Funktionen in ken_evaluation.py) an handgesetzten Werten: Schwellen und jede Zahl im Text."""

import dataclasses

import ken_evaluation as ev
from ken_scenario import grid_netz

S = ev.Settings()


def _a(**kw) -> ev.Analysis:
    basis = ev.Analysis(
        netz=grid_netz(3, 4), P=4.17e-4, cv2_plain=2397.0, qp_opt=0.1188, cv2_opt=80.6, ladder=((0.01, -1, 0.0776, 0.0662), (0.0662, 0, 0.0225, 0.1288)), reached=True, fixpoint=False, rho_hi=0.021,
        qp_ladder=0.1288, cv2_ladder=81.4, p_event=0.0225, stuck=0.0, stages=2.0, qp_med=0.1285, qp_lo=0.125, qp_hi=0.137, ratio_med=1.01, ratio_gt2=0.0, evals=2000.0,
        trajectories=((0.01, 0.066, 0.127), (0.01, 0.067, 0.13)))
    return dataclasses.replace(basis, **kw)


def test_erfolg_mit_zahlen():
    kind, text = ev.verdict_main(_a(), S)
    assert kind == "success"
    for frag in ("4,2e-04", "q' = 0,119 (CV² = 80,6 gegen 2.397", "braucht 2,0 Stufen (2.000 Ziehungen)", "q' = 0,129 (10 bis 90 %: 0,125 bis 0,137)", "1,01-mal so groß", "Festgefahren: 0 %"):
        assert frag in text, frag


def test_fixpunkt_warnung_nennt_rho_grenze_und_wahrscheinlichkeit():
    a = _a(reached=False, fixpoint=True, ladder=((0.01, -1, 0.0776, 0.0662), (0.0662, -1, 0.4343, 0.0662)), p_event=0.0225, stuck=1.0)
    kind, text = ev.verdict_main(a, ev.Settings(rho=0.05))
    assert kind == "warning" and "q' läuft auf 0,066 und bleibt dort" in text and "Wahrscheinlichkeit 0,022" in text and "unter 0,021 liegen" in text and "100 % der Läufe festgefahren" in text


def test_warnung_ohne_elite_in_der_ersten_stufe():
    a = _a(reached=False, fixpoint=True, ladder=((0.001, -2, 1.0, 0.001),), stuck=0.51, rho_hi=0.008)
    kind, text = ev.verdict_main(a, ev.Settings(q=0.001, n=300, rho=0.01))
    assert kind == "warning" and "findet schon die erste Stufe keine Elite" in text and "51 % der Läufe festgefahren" in text and "unter 0,008 liegen" in text


def test_warnung_wenn_viele_laeufe_stecken_bleiben_obwohl_die_erwartung_das_ereignis_erreicht():
    a = _a(stuck=0.86)
    kind, text = ev.verdict_main(a, ev.Settings(q=0.001, n=3000, rho=0.01))
    assert kind == "warning" and "bleiben 86 % der Läufe stecken" in text and "nötig mindestens 30" in text


def test_unmoegliches_ereignis():
    kind, text = ev.verdict_main(_a(P=0.0), S)
    assert kind == "info" and "nie aus" in text


def test_verlaufs_verdikt_zahlen():
    kind, text = ev.verdict_trajectories(_a())
    assert kind == "info" and "q' = 0,010 → 0,066 → 0,129 nach 2 Stufen" in text and "Der erste Lauf: 0,010 → 0,066 → 0,127" in text
    assert "unmöglich" in ev.verdict_trajectories(_a(P=0.0))[1]


def test_gitter_verdikt_zahlen():
    cells = [{"n": n, "rho": r, "stuck": 0.0 if r <= 0.02 else 1.0, "ratio": 1.01} for n in (300, 3000) for r in (0.005, 0.01, 0.02, 0.05, 0.1)]
    kind, text = ev.verdict_grid({"cells": cells}, S)
    assert kind == "info" and "Zellen ohne Festfahren (unter 5 %): 6 von 10" in text and "immer festgefahren: 4" in text and "ρ = 0,005, ρ = 0,010, ρ = 0,020" in text


def test_budget_verdikt_zahlen():
    rows = [{"method": "einfach", "stuck": 0.0, "zero": 0.1, "rmse": 0.62}, {"method": "Orakel (festes Optimum)", "stuck": 0.0, "zero": 0.0, "rmse": 0.11},
            {"method": "Kreuzentropie", "stuck": 0.0, "zero": 0.0, "rmse": 0.14}, {"method": "adaptives Splitting", "stuck": 0.0, "zero": 0.0, "rmse": 0.32}]
    kind, text = ev.verdict_budget({"rows": rows, "P": 4e-4, "budget": 6000})
    assert kind == "info" and "Bei Budget 6.000" in text and "einfach 0,62 (10 % Nullschätzungen)" in text and "Orakel mit bekanntem q' 0,11" in text and "Kreuzentropie 0,14" in text and "Splitting 0,32" in text
    assert "1,27-Fache" in text
    assert "unmöglich" in ev.verdict_budget({"rows": rows, "P": 0.0, "budget": 6000})[1]


def test_vorhersage_verdikt_zahlen():
    rows = [{"budget": 2000, "plain": 1.1, "oracle": 0.2, "ce": 0.26, "split": 0.54}, {"budget": 6000, "plain": 0.63, "oracle": 0.12, "ce": 0.14, "split": 0.31}]
    kind, text = ev.verdict_prediction({"rows": rows, "probs": [0.0776, 0.0054], "factor": 197.1, "P": 4e-4})
    assert kind == "info" and "2 Stufen mit den Wahrscheinlichkeiten 0,0776 und 0,0054" in text and "zusammen 197" in text and "Budget 6.000: einfach 0,63, Orakel 0,12, Kreuzentropie 0,14, Splitting 0,31" in text


def test_fixpunkt_tabelle_verdikt():
    rows = [{"rho": 0.01, "stages": [(0.01, -1, 0.078, 0.0662), (0.0662, 0, 0.0225, 0.1288)], "reached": True, "fixpoint": False},
            {"rho": 0.05, "stages": [(0.01, -1, 0.078, 0.0662), (0.0662, -1, 0.434, 0.0662)], "reached": False, "fixpoint": True}]
    kind, text = ev.verdict_fixpoint({"rows": rows, "rho_hi": 0.021}, S)
    assert kind == "info" and "für ρ = 0,010 erreicht" in text and "ρ ≈ 0,021" in text and "Bei ρ = 0,050 bleibt q' bei 0,066 stehen (Niveau -1, Wahrscheinlichkeit 0,434)" in text


def test_formate():
    assert ev.de(1234.567, 1) == "1.234,6" and ev.de(-0.5, 2) == "−0,50" and ev.de_e(0.000042) == "4,2e-05" and ev.dn(float("nan")) == "nicht bestimmbar" and ev.dn(float("inf")) == "unendlich"
    assert ev.dz(2.4e8) == "2,4e+08" and ev.dz(239700) == "239.700" and ev.dz(29.74) == "29,7" and ev.kanten(1) == "1 Kante" and ev.kanten(2) == "2 Kanten"
