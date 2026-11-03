"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; jedes Preset rechnet und hat das erwartete Verdikt."""

import pytest

import ken_constants as C
import ken_evaluation as ev
import ken_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_options():
    for name, preset in C.PRESETS.items():
        assert 0 <= preset["seed"] <= C.SEED_MAX, name
        assert preset["q"] in C.Q_CHOICES and preset["n"] in C.N_CHOICES and preset["rho"] in C.RHO_CHOICES and preset["budget"] in C.BUDGET_CHOICES and preset["netz"] in C.NETZE, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall (Voreinstellung)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_jedes_preset_unterscheidet_sich_vom_standard():
    standard = C.PRESETS["Standardfall (Voreinstellung)"]
    for name, preset in C.PRESETS.items():
        if name != "Standardfall (Voreinstellung)":
            assert preset != standard, name


def test_standard_preset_lernt_in_zwei_stufen():
    s = ev.Settings(**C.PRESETS["Standardfall (Voreinstellung)"])
    a = ev.analyse(s)
    assert a.reached and len(a.ladder) == 2 and a.stuck < 0.05 and a.ratio_med < 1.1 and ev.verdict_main(a, s)[0] == "success"


def test_preset_fixpunkt_falle_faehrt_sich_fest():
    s = ev.Settings(**C.PRESETS["Zu große Elite (Fixpunkt-Falle)"])
    a = ev.analyse(s)
    assert not a.reached and a.fixpoint and a.stuck == 1.0 and abs(a.ladder[-1][3] - 0.0665) < 2e-3 and ev.verdict_main(a, s)[0] == "warning"


def test_preset_zu_wenig_treffer_hat_keine_elite_in_der_ersten_stufe():
    s = ev.Settings(**C.PRESETS["Noch seltener, zu wenig Treffer"])
    a = ev.analyse(s)
    assert not a.reached and a.ladder[0][1] == -2 and a.stuck > 0.3 and ev.verdict_main(a, s)[0] == "warning"


def test_preset_passende_elite_bei_noch_selteneren_ausfaellen_gelingt():
    s = ev.Settings(**C.PRESETS["Noch seltener, passende Elite"])
    a = ev.analyse(s)
    assert a.reached and a.stuck < 0.15 and a.ratio_med < 1.1 and ev.verdict_main(a, s)[0] == "success"


def test_preset_hantel_lernt_in_einer_stufe():
    s = ev.Settings(**C.PRESETS["Hantel (ein einziger Schnitt)"])
    a = ev.analyse(s)
    assert a.reached and len(a.ladder) == 1 and a.stuck == 0.0 and a.netz.name == "Hantel"


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_rechnet(name):
    a = ev.analyse(ev.Settings(**C.PRESETS[name]))
    assert 0.0 < a.P < 1.0
