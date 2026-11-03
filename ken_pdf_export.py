"""PDF-Export der aktuellen Einstellung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import ken_constants as C
from ken_evaluation import Analysis, Settings, de, de_e, dn, dz, verdict_main

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "·": "*", "€": "EUR", "„": '"', "“": '"', "²": "^2", "³": "3", "×": "x", "√": "Wurzel", "≈": "~",
            "σ": "Sigma", "ρ": "rho", "α": "alpha", "β": "beta", "θ": "theta", "∞": "unendlich", "Σ": "Summe", "δ": "delta", "Δ": "Delta", "½": "1/2", "̂": "", "ϱ": "rho"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(settings: Settings, a: Analysis) -> bytes:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Kreuzentropie: die Hilfsverteilung lernen"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(
        f"{a.netz.name}, {a.netz.n} Knoten, {a.netz.m} Kanten; jede Kante fällt mit Wahrscheinlichkeit q = {de(settings.q, 3)} aus; Ereignis: das Netz hängt nicht zusammen. "
        f"{de(settings.n, 0)} Ziehungen je Stufe, Elite-Anteil rho = {de(settings.rho, 3)}, {C.REPS} Lernläufe, Strom {settings.seed}. Alles simuliert, keine Messdaten."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 9)
    widths = (80, 60)
    pdf.cell(widths[0], 6, _s("Kennzahl"), border=1)
    pdf.cell(widths[1], 6, _s("Wert"), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    stufen = " -> ".join(de(x[3], 3) for x in a.ladder)
    for label, val in (("Ausfallwahrscheinlichkeit P (exakt)", de_e(a.P)), ("CV^2 einfaches Ziehen (exakt)", dz(a.cv2_plain)), ("optimales q' und CV^2 (exakt)", f"{dn(a.qp_opt, 3)} / {dz(a.cv2_opt)}"),
                       ("exakte Stufenfolge q'", f"{de(a.ladder[0][0], 3)} -> {stufen}" + ("" if a.reached else " (Ereignis nicht erreicht)")),
                       ("festgefahren (gemessen)", f"{de(a.stuck * 100, 0)} %"), ("Stufen (Mittel)", dn(a.stages, 1)), ("gelerntes q' (Median)", dn(a.qp_med, 3)),
                       ("CV^2-Verhältnis zum Optimum (Median)", dn(a.ratio_med, 2)), ("Ziehungen des Lernens", dn(a.evals, 0))):
        pdf.cell(widths[0], 6, _s(label), border=1)
        pdf.cell(widths[1], 6, _s(val), border=1)
        pdf.ln()
    pdf.ln(2)
    pdf.multi_cell(0, 5, _s(verdict_main(a, settings)[1]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: kreuzentropie-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
