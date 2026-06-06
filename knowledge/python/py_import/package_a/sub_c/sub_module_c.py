# package_a.sub_c.sub_module_c.py
print("package_a.sub_c.sub_module_c.py executed")

# function
def sub_module_c_func():
    print("package_a.sub_c.sub_module_c.sub_module_c_func() executed")


# class
class c:
    print("package_a.sub_c.sub_module_c.c class body executed")


# absolute import 
def absolute_import_module_a():
    from package_a import module_a

    return module_a.MODULE_A_VALUE




# relative import 
def relative_import_module_a():
    from .. import module_a

    return module_a.MODULE_A_VALUE





