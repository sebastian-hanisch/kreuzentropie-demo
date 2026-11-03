"""AppTest der Experimente, der Spielwiese, des PDF-Exports und des Permalinks (Einzelheiten, die der allgemeine Rauchtest nicht trifft)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import ken_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texte(at) -> str:
    return " ".join(e.value for kind in (at.success, at.info, at.warning, at.error) for e in kind)


@pytest.mark.parametrize("key,fragment", [
    ("run_grid", "Zellen ohne Festfahren (unter 5 %)"),
    ("run_budget", "Das Lernen kostet gegen das Orakel"),
])
def test_experiment_auf_klick_zeigt_sein_verdikt(key, fragment):
    at = _app()
    assert fragment not in _texte(at)                       # vor dem Klick steht noch nichts da
    at.button(key=key).click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert fragment in _texte(at)


def test_exakte_experimente_stehen_ohne_klick_da():
    t = _texte(_app())
    assert "Erwartet (exakt): q' = 0,010 → 0,066 → 0,129 nach 2 Stufen" in t
    assert "Splitting hat hier 2 Stufen mit den Wahrscheinlichkeiten 0,0776 und 0,0054" in t
    assert "Das Ereignis wird für ρ = 0,005, 0,010, 0,020 erreicht" in t


def test_experiment_bleibt_nach_einer_aenderung_der_einstellungen_sichtbar():
    at = _app()
    at.button(key="run_budget").click().run(timeout=TIMEOUT)
    at.sidebar.select_slider(key="budget_select").set_value(2000).run(timeout=TIMEOUT)
    assert not at.exception
    assert "Das Lernen kostet gegen das Orakel" in _texte(at) and "Bei Budget 2.000" in _texte(at)


def test_kernfrage_verdikt_wechselt_mit_dem_preset():
    at = _app()
    assert "Das Lernen braucht 2,0 Stufen" in _texte(at)
    at.sidebar.button(key="preset_Zu große Elite (Fixpunkt-Falle)").click().run(timeout=TIMEOUT)
    assert "fährt sich das Lernen fest" in _texte(at)
    at.sidebar.button(key="preset_Noch seltener, zu wenig Treffer").click().run(timeout=TIMEOUT)
    assert "findet schon die erste Stufe keine Elite" in _texte(at)
    at.sidebar.button(key="preset_Hantel (ein einziger Schnitt)").click().run(timeout=TIMEOUT)
    assert "Das Lernen braucht 1,0 Stufen" in _texte(at) or "Das Lernen braucht 1," in _texte(at)


def test_jedes_netz_laeuft_durch():
    at = _app()
    for netz in C.NETZE:
        at.sidebar.selectbox(key="netz_select").set_value(netz).run(timeout=TIMEOUT)
        assert not at.exception, (netz, [str(e) for e in at.exception])


def test_spielwiese_gueltiges_netz_und_fehlermeldungen():
    at = _app()
    assert not at.error
    at.text_area(key="eigene_kanten").set_value("0-0").run(timeout=TIMEOUT)
    assert not at.exception and "Schleife" in " ".join(e.value for e in at.error)
    at.text_area(key="eigene_kanten").set_value("").run(timeout=TIMEOUT)
    assert not at.exception and "Keine Kanten" in " ".join(e.value for e in at.error)


def test_spielwiese_pfad_laeuft_durch():
    at = _app()
    at.text_area(key="eigene_kanten").set_value("0-1\n1-2").run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert not at.error


def test_pdf_wird_erst_nach_klick_erzeugt():
    at = _app()
    assert not at.get("download_button")
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert len(at.get("download_button")) == 1
    at.sidebar.select_slider(key="budget_select").set_value(2000).run(timeout=TIMEOUT)         # andere Einstellung: altes PDF nicht mehr anbieten
    assert not at.get("download_button")


def test_permalink_wird_begrenzt_und_eingerastet():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["q"] = "0.0007"
    at.query_params["n"] = "400"
    at.query_params["r"] = "0.03"
    at.query_params["b"] = "9000"
    at.query_params["g"] = "9x9"            # unbekanntes Netz: verworfen
    at.query_params["seed"] = "-4"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.sidebar.select_slider(key="q_select").value == 0.001                           # unterhalb des kleinsten Wertes: der kleinste
    assert at.sidebar.select_slider(key="n_select").value == 300                             # 400 liegt näher an 300 als an 1000
    assert at.sidebar.select_slider(key="rho_select").value == 0.02                          # 0,03 liegt näher an 0,02 als an 0,05
    assert at.sidebar.select_slider(key="budget_select").value == 6000                       # 9000 liegt näher an 6000 als an 20 000
    assert at.sidebar.selectbox(key="netz_select").value == C.DEFAULT_NETZ
    assert at.sidebar.number_input(key="seed_input").value == 0
