"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ken_evaluation as ev  # noqa: E402
from ken_scenario import grid_netz, hantel_netz, parse_edges  # noqa: E402


@pytest.fixture(scope="session")
def pfad():
    """Pfad aus zwei Kanten: das Netz fällt aus, wenn mindestens eine Kante ausfällt. Bewertung: beide halten -> -1 (kleinster Grad 1), sonst 0. Handrechnung: Update-Ziel bei q = 1/2 ist 2/3."""
    netz = parse_edges("0-1\n1-2")
    return netz, ev.netz_table(netz)


@pytest.fixture(scope="session")
def dreieck():
    netz = parse_edges("0-1\n1-2\n2-0")
    return netz, ev.netz_table(netz)


@pytest.fixture(scope="session")
def hantel():
    netz = hantel_netz()
    return netz, ev.netz_table(netz)


@pytest.fixture(scope="session")
def gitter34():
    netz = grid_netz(3, 4)
    return netz, ev.netz_table(netz)
