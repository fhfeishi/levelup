# module_1.py
"""A plain module used by the import experiments."""

print("module_1.py executed")

MODULE_VALUE = "value from module_1"


def f1():
    print("module_1.f1() executed")


class c1:
    print("module_1.c1 class body executed")

    def method(self):
        return "method result from module_1.c1"


__all__ = ["MODULE_VALUE", "f1", "c1"]
    





