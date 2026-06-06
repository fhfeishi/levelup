
"""A submodule inside package_a."""

print("package_a.module_a.py executed")

MODULE_A_VALUE = "value from package_a.module_a"


class a:
    print("package_a.module_a.a class body executed")


def module_a_func():
    print("package_a.module_a.module_a_func() executed")


def import_from_sibling_package():
    """Import package_b.module_b from package_a.

    package_a and package_b are siblings under the project root, so this must
    be an absolute import. Relative import cannot cross from one top-level
    package to another top-level package.
    """
    from package_b import module_b

    return module_b.func_b()


def import_from_child_package_relative():
    """Import package_a.sub_c.sub_module_c with explicit relative import."""
    from .sub_c import sub_module_c

    return sub_module_c.sub_module_c_func()


def import_from_child_package_absolute():
    """Import package_a.sub_c.sub_module_c with absolute import."""
    from package_a.sub_c import sub_module_c

    return sub_module_c.sub_module_c_func()
