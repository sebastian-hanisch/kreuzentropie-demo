"""Netze: Gitter, die Hantel und eigene Kantenlisten (Szenario der Demo). Reine Daten und Eingabeprüfung, keine Rechnung."""

import math
import re
from dataclasses import dataclass

import ken_constants as C


@dataclass(frozen=True)
class Netz:
    name: str
    n: int
    edges: tuple[tuple[int, int], ...]
    coords: tuple[tuple[float, float], ...]   # nur für die Zeichnung

    @property
    def m(self) -> int:
        return len(self.edges)


def grid_netz(rows: int, cols: int) -> Netz:
    """Gitter rows x cols: Knoten zeilenweise nummeriert, Kanten waagerecht und senkrecht je Knoten."""
    edges = []
    for r in range(rows):
        for c in range(cols):
            v = r * cols + c
            if c + 1 < cols:
                edges.append((v, v + 1))
            if r + 1 < rows:
                edges.append((v, v + cols))
    coords = tuple((float(c), float(-r)) for r in range(rows) for c in range(cols))
    return Netz(f"Gitter {rows}×{cols}", rows * cols, tuple(edges), coords)


def hantel_netz() -> Netz:
    """Zwei Dreiecke (Knoten 0 bis 2 und 3 bis 5), durch die Brücke 2-3 verbunden: 7 Kanten, die Brücke ist die wirksame Eingabe."""
    edges = ((0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3))
    coords = ((0.0, 0.6), (0.0, -0.6), (1.0, 0.0), (3.0, 0.0), (4.0, 0.6), (4.0, -0.6))
    return Netz("Hantel", 6, edges, coords)


def named_netz(key: str) -> Netz:
    art, rows, cols = C.NETZE[key]
    return hantel_netz() if art == "hantel" else grid_netz(rows, cols)


def _circle(n: int) -> tuple[tuple[float, float], ...]:
    return tuple((math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)) for i in range(n))


def _parse_pair(line: str, line_no: int) -> tuple[int, int]:
    parts = re.split(r"[\s,;-]+", line.strip())
    if len(parts) != 2 or not all(re.fullmatch(r"\d+", p) for p in parts):
        raise ValueError(f"Zeile {line_no}: erwartet zwei ganze Zahlen wie »0-1«, gefunden »{line.strip()}«")
    a, b = int(parts[0]), int(parts[1])
    if a == b:
        raise ValueError(f"Zeile {line_no}: Schleife {a}-{a} ist keine Kante")
    return min(a, b), max(a, b)


def parse_edges(text: str) -> Netz:
    """Eigenes Netz aus Zeilen "a-b" (auch "a b" oder "a,b"); Knotennamen sind ganze Zahlen und werden von 0 an umnummeriert.

    Fehler als ValueError mit einer Meldung für die App: Schleifen, doppelte Kanten, zu viele Kanten oder Knoten, leere Eingabe."""
    pairs = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        if raw.strip() and not raw.strip().startswith("#"):
            pairs.append(_parse_pair(raw, line_no))
    if not pairs:
        raise ValueError("Keine Kanten angegeben")
    if len(set(pairs)) != len(pairs):
        raise ValueError("Doppelte Kante: jede Kante höchstens einmal")
    if len(pairs) > C.CUSTOM_EDGE_LIMIT:
        raise ValueError(f"{len(pairs)} Kanten: höchstens {C.CUSTOM_EDGE_LIMIT} (die Aufzählung zählt 2^m Kantenmengen)")
    nodes = sorted({v for p in pairs for v in p})
    if len(nodes) > C.CUSTOM_NODE_LIMIT:
        raise ValueError(f"{len(nodes)} Knoten: höchstens {C.CUSTOM_NODE_LIMIT}")
    index = {v: i for i, v in enumerate(nodes)}
    return Netz("Eigenes Netz", len(nodes), tuple((index[a], index[b]) for a, b in pairs), _circle(len(nodes)))
