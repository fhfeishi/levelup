"""A deeper package entry-point experiment.

Try these from the project root:

    python package_a/sub_c/test_c.py
    python -m package_a.sub_c.test_c
"""

from __future__ import annotations

import sys
import importlib


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
    "absolute import: from package_a import module_a",
    lambda: __import__("package_a.module_a", fromlist=["module_a"]),
)

try_import(
    "explicit relative import: from .. import module_a",
    lambda: importlib.import_module("..module_a", package=__package__),
)

try_import(
    "explicit relative import: from . import sub_module_c",
    lambda: importlib.import_module(".sub_module_c", package=__package__),
)
