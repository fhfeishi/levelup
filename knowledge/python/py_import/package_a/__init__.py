"""package_a is a regular package because this file exists."""

print("package_a.__init__.py executed")

PACKAGE_VALUE = "value from package_a"

# __all__ affects `from package_a import *`, not ordinary import lookup.
__all__ = ["PACKAGE_VALUE"]
