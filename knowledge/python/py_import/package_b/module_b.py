
"""A module in package_b."""

print("package_b.module_b.py executed")


class b:
    print("package_b.module_b.b class body executed")


def func_b():
    print("package_b.module_b.func_b() executed")


# absolute import 
def import_from_package_a():
    """Import a sibling top-level package with absolute import."""
    from package_a import module_a

    return module_a.MODULE_A_VALUE



# relative import 
def import_from_same_package():
    """Relative import inside namespace package package_b."""
    from . import module_b

    return module_b.func_b
