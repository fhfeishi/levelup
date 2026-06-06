"""Observe why execution style changes import behavior.

Try these from the project root:

    python package_a/test_a.py
    python -m package_a.test_a

The file-path form runs this file as an isolated script. The -m form runs it
as a module inside package_a, so explicit relative imports know their package.
"""

from __future__ import annotations

import sys
import importlib


def section(title: str) -> None:
    print(f"\n{'=' * 12} {title} {'=' * 12}")


def try_import(label: str, import_func) -> None:
    print(f"\n{label}")
    try:
        result = import_func()
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
    else:
        print(f"OK: {result!r}")


section("execution context")
print(f"__name__      = {__name__!r}")
print(f"__package__   = {__package__!r}")
print(f"__file__      = {__file__!r}")
print(f"__spec__      = {__spec__!r}")
print(f"sys.path[0]   = {sys.path[0]!r}")

section("imports")

try_import(
    "absolute import: import module_1",
    lambda: __import__("module_1"),
)

try_import(
    "absolute import: from package_a import module_a",
    lambda: __import__("package_a.module_a", fromlist=["module_a"]),
)

try_import(
    "explicit relative import: from . import module_a",
    lambda: importlib.import_module(".module_a", package=__package__),
)

try_import(
    "explicit relative import: from .sub_c import sub_module_c",
    lambda: importlib.import_module(".sub_c.sub_module_c", package=__package__),
)


if __name__ == "__main__":
    print("\nDone. Now compare this output with `python -m package_a.test_a`.")
