"""Sibling-package experiment for package_b.

Try these from the project root:

    python package_b/test_b.py
    python -m package_b.test_b
"""

from __future__ import annotations

import importlib
import sys


def try_import(label: str, import_func) -> None:
    print(f"\n{label}")
    try:
        result = import_func()
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
    else:
        print(f"OK: {result!r}")


print(f"__name__      = {__name__!r}")
print(f"__package__   = {__package__!r}")
print(f"__file__      = {__file__!r}")
print(f"__spec__      = {__spec__!r}")
print(f"sys.path[0]   = {sys.path[0]!r}")

try_import(
    "absolute import: from package_b import module_b",
    lambda: importlib.import_module("package_b.module_b"),
)

try_import(
    "explicit relative import: from . import module_b",
    lambda: importlib.import_module(".module_b", package=__package__),
)

try_import(
    "cross-package absolute import: from package_a import module_a",
    lambda: importlib.import_module("package_a.module_a"),
)
