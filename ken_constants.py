"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

NETZE = {"3x3": ("gitter", 3, 3), "3x4": ("gitter", 3, 4), "hantel": ("hantel", 0, 0)}
DEFAULT_NETZ = "3x4"
LIVE_EDGE_LIMIT = 17          # Aufzählung aller Kantenmengen live bis hierher (2^17 = 131 072, rund 1 s); eigenes Netz ebenso
CUSTOM_EDGE_LIMIT = 17
CUSTOM_NODE_LIMIT = 14

Q_CHOICES = (0.02, 0.01, 0.005, 0.002, 0.001)               # Ausfallwahrscheinlichkeit je Kante (Seltenheit)
Q_MIN, Q_MAX, DEFAULT_Q = Q_CHOICES[-1], Q_CHOICES[0], 0.01
N_CHOICES = (100, 300, 1000, 3000)                          # Ziehungen je Stufe des Lernens
N_MIN, N_MAX, DEFAULT_N = N_CHOICES[0], N_CHOICES[-1], 1000
RHO_CHOICES = (0.005, 0.01, 0.02, 0.05, 0.1)                # Anteil der Elite je Stufe
RHO_MIN, RHO_MAX, DEFAULT_RHO = RHO_CHOICES[0], RHO_CHOICES[-1], 0.01
BUDGET_CHOICES = (2000, 6000, 20000)                        # Gesamtbudget in Ziehungen für den Vergleich
BUDGET_MIN, BUDGET_MAX, DEFAULT_BUDGET = BUDGET_CHOICES[0], BUDGET_CHOICES[-1], 6000
SEED_MAX, DEFAULT_SEED = 999999, 35
REPS = 150                                # Wiederholungen je Messung (feste Ströme)
GRID_REPS = 150                           # Wiederholungen je Zelle der ρ-N-Tabelle
SWEEPS = 2                                # Gibbs-Durchläufe je Splitting-Stufe
MAX_STAGES = 8                            # Obergrenze der Stufen (danach gilt das Lernen als festgefahren)
SPLIT_RHO_MAX = 0.05                      # Anteil der Überlebenden je Splitting-Stufe, wenn kleiner als die erste Stufe

Z95 = 1.959963984540054

COLOR_PLAIN = "#4c78a8"
COLOR_ORACLE = "#E08E0B"
COLOR_CE = "#2F6B65"
COLOR_SPLIT = "#8e6bb0"
COLOR_BAD = "#C0392B"
COLOR_TRUTH = "#222222"
COLOR_OTHER = "rgba(150,150,150,0.55)"

PRESETS = {
    "Standardfall (Voreinstellung)": {"netz": DEFAULT_NETZ, "q": DEFAULT_Q, "n": DEFAULT_N, "rho": DEFAULT_RHO, "budget": DEFAULT_BUDGET, "seed": DEFAULT_SEED},
    "Zu große Elite (Fixpunkt-Falle)": {"netz": DEFAULT_NETZ, "q": DEFAULT_Q, "n": DEFAULT_N, "rho": 0.05, "budget": DEFAULT_BUDGET, "seed": DEFAULT_SEED},
    "Noch seltener, zu wenig Treffer": {"netz": DEFAULT_NETZ, "q": 0.001, "n": 300, "rho": 0.01, "budget": DEFAULT_BUDGET, "seed": DEFAULT_SEED},
    "Noch seltener, passende Elite": {"netz": DEFAULT_NETZ, "q": 0.001, "n": 3000, "rho": 0.005, "budget": 20000, "seed": DEFAULT_SEED},
    "Hantel (ein einziger Schnitt)": {"netz": "hantel", "q": DEFAULT_Q, "n": DEFAULT_N, "rho": DEFAULT_RHO, "budget": DEFAULT_BUDGET, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Gitter 3×4, jede Kante fällt mit 1 % aus, 1000 Ziehungen je Stufe, 1 % Elite: das Lernen findet in zwei Stufen eine Kippung, die fast so gut ist wie das unbekannte Optimum.",
    "Zu große Elite (Fixpunkt-Falle)": "Mit 5 % Elite bleibt die Stufe bei „ein Eckknoten hat nur noch eine Kante“ stehen: q' läuft in einen Zwischen-Fixpunkt und erreicht den Ausfall nie.",
    "Noch seltener, zu wenig Treffer": "Bei q = 0,001 und 300 Ziehungen je Stufe sieht die erste Stufe oft keinen Treffer: das Lernen kommt nicht in Gang.",
    "Noch seltener, passende Elite": "Bei q = 0,001 mit 3000 Ziehungen je Stufe und 0,5 % Elite gelingt das Lernen fast immer.",
    "Hantel (ein einziger Schnitt)": "Eine einzige Brücke trennt das Netz: die Bewertungsfunktion hat kaum Zwischenstufen, das Lernen wird kürzer.",
}
