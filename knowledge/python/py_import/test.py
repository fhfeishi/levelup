"""Import mechanism playground.

Run from the project root:

    python test.py

This file focuses on import statements, module objects, package objects,
sys.modules cache, and introspection helpers such as dir() / vars().
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import pkgutil
import subprocess
import sys


def section(title: str) -> None:
    print(f"\n{'=' * 12} {title} {'=' * 12}")


def show_execution_context() -> None:
    section("current module context")
    print(f"__name__      = {__name__!r}")
    print(f"__package__   = {__package__!r}")
    print(f"__file__      = {__file__!r}")
    print(f"__spec__      = {__spec__!r}")
    print(f"sys.path[0]   = {sys.path[0]!r}")


def show_module_object(mod) -> None:
    print(f"module object      = {mod!r}")
    print(f"__name__           = {mod.__name__!r}")
    print(f"__package__        = {mod.__package__!r}")
    print(f"__file__           = {getattr(mod, '__file__', None)!r}")
    print(f"__cached__         = {getattr(mod, '__cached__', None)!r}")
    print(f"__spec__.name      = {mod.__spec__.name!r}")
    print(f"__spec__.origin    = {mod.__spec__.origin!r}")
    print(f"has __path__       = {hasattr(mod, '__path__')}")
    if hasattr(mod, "__path__"):
        print(f"__path__           = {list(mod.__path__)!r}")


def show_dir_and_vars(mod) -> None:
    names = dir(mod)
    public_names = [name for name in names if not name.startswith("_")]
    public_keys = [key for key in vars(mod) if not key.startswith("_")]
    print(f"dir({mod.__name__}) public names = {public_names!r}")
    print(f"vars({mod.__name__}) public keys  = {public_keys!r}")


def show_find_spec(name: str) -> None:
    spec = importlib.util.find_spec(name)
    print(f"find_spec({name!r}) = {spec!r}")
    # !r 等同于 repr()  # 返回一个对象的“官方”字符串表示，适合调试和日志记录。
    # print(f"find_spec({repr(name)}) = {repr(spec)}")  
    if spec is not None:
        print(f"  origin                    = {spec.origin!r}")
        print(f"  submodule_search_locations = {spec.submodule_search_locations!r}")


def show_cache(*names: str) -> None:
    for name in names:
        print(f"{name!r} in sys.modules -> {name in sys.modules}")


def demo_import_module() -> None:
    section("import module_1")
    show_cache("module_1")

    import module_1

    print("After `import module_1`, the bound name is `module_1`.")
    show_cache("module_1")
    show_module_object(module_1)
    show_dir_and_vars(module_1)
    print(f"id(module_1) == id(sys.modules['module_1']) -> {id(module_1) == id(sys.modules['module_1'])}")

    section("from module_1 import f1, c1 as c")
    from module_1 import c1 as c
    from module_1 import f1

    print("`from ... import ...` binds selected attributes, not the module name again.")
    print(f"f1                  = {f1!r}")
    print(f"c                   = {c!r}")
    print(f"callable(f1)        = {callable(f1)}")
    print(f"f1 is module_1.f1   = {f1 is module_1.f1}")


def demo_package_imports() -> None:
    section("import package_a")
    show_cache("package_a", "package_a.module_a")

    import package_a

    print("A package is also a module object, plus it usually has __path__.")
    show_module_object(package_a)
    show_dir_and_vars(package_a)
    print("Children are not automatically imported just because the package exists.")
    show_cache("package_a", "package_a.module_a")

    section("import package_a.module_a")
    import package_a.module_a

    print("`import package_a.module_a` binds the top-level name `package_a`.")
    print(f"hasattr(package_a, 'module_a') -> {hasattr(package_a, 'module_a')}")
    print(f"package_a.module_a = {package_a.module_a!r}")
    show_cache("package_a", "package_a.module_a")

    section("from package_a import module_a")
    from package_a import module_a

    print("`from package_a import module_a` binds the child object directly as `module_a`.")
    print(f"module_a is package_a.module_a -> {module_a is package_a.module_a}")

    section("import package_a.sub_c.sub_module_c")
    import package_a.sub_c.sub_module_c

    print("Each parent package is loaded and receives the imported child as an attribute.")
    print(f"package_a.sub_c = {package_a.sub_c!r}")
    print(f"package_a.sub_c.sub_module_c = {package_a.sub_c.sub_module_c!r}")
    show_cache("package_a", "package_a.sub_c", "package_a.sub_c.sub_module_c")


def demo_namespace_package() -> None:
    section("namespace package: package_b has no __init__.py")
    show_find_spec("package_b")
    show_cache("package_b", "package_b.module_b")

    import package_b

    print("package_b is importable even without package_b/__init__.py.")
    print("No __init__.py means there is no package initialization code to execute.")
    show_module_object(package_b)
    show_dir_and_vars(package_b)

    section("import package_b.module_b")
    import package_b.module_b

    print("A namespace package can still contain normal submodules.")
    print(f"package_b.module_b = {package_b.module_b!r}")
    show_cache("package_b", "package_b.module_b")


def demo_cross_package_imports() -> None:
    section("package_a imports sibling package_b and child subpackage")
    from package_a import module_a

    print("package_a.module_a -> package_b.module_b uses absolute import:")
    module_a.import_from_sibling_package()

    print("\npackage_a.module_a -> package_a.sub_c.sub_module_c uses relative import:")
    module_a.import_from_child_package_relative()

    print("\nThe same child import can also be written as absolute import:")
    module_a.import_from_child_package_absolute()

    print("\npackage_b.module_b -> package_a.module_a also uses absolute import:")
    from package_b import module_b

    print(module_b.import_from_package_a())


def demo_tools() -> None:
    section("useful import/introspection tools")

    for name in ["module_1", "package_a", "package_a.module_a", "package_a.sub_c", "package_b", "package_b.module_b"]:
        show_find_spec(name)

    print("\nTraditional modules/packages reported by pkgutil.iter_modules([sys.path[0]]):")
    print([m.name for m in pkgutil.iter_modules([sys.path[0]])])
    print("Notice: namespace packages such as package_b may not appear in this list; find_spec is clearer.")

    import module_1

    print("\ninspect.getmembers(module_1, inspect.isfunction):")
    print(inspect.getmembers(module_1, inspect.isfunction))
    print("getattr(module_1, 'MODULE_VALUE'):")
    print(getattr(module_1, "MODULE_VALUE"))


def demo_importlib() -> None:
    section("importlib.import_module")
    mod = importlib.import_module("package_a.module_a")
    print(f"importlib.import_module('package_a.module_a') -> {mod!r}")
    print("This is the programmatic form of import; plugins/frameworks often use it.")


def demo_command_matrix() -> None:
    section("entry command matrix")
    commands = [
        [sys.executable, "test.py", "--context-only"],
        [sys.executable, "package_a/test_a.py"],
        [sys.executable, "-m", "package_a.test_a"],
        [sys.executable, "package_b/test_b.py"],
        [sys.executable, "-m", "package_b.test_b"],
    ]

    for command in commands:
        print(f"\n$ {' '.join(command)}")
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        output = result.stdout.strip() or result.stderr.strip()
        print(f"exit code = {result.returncode}")
        for line in output.splitlines()[:18]:
            print(f"  {line}")
        if len(output.splitlines()) > 18:
            print("  ...")


if __name__ == "__main__":
    show_execution_context()
    if "--context-only" in sys.argv:
        raise SystemExit

    demo_import_module()
    demo_package_imports()
    demo_namespace_package()
    demo_cross_package_imports()
    demo_tools()
    demo_importlib()
    demo_command_matrix()
