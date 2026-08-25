"""Every module must import.

This exists because a syntax error shipped in pipeline.py and the whole suite
still passed -- no test imported it. Four sweep runs failed silently before
anyone noticed. Cheap insurance against the same class of mistake.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import fitment_rag

MODULES = sorted(
    m.name for m in pkgutil.walk_packages(fitment_rag.__path__, "fitment_rag.")
)


def test_every_module_is_discovered():
    assert len(MODULES) >= 10, f"expected the full package, found {MODULES}"


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)
