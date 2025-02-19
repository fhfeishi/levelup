import test
# from package1 import module1
# from package1.subpackage11 import module11
from a1 import a1
from a1.aa11 import aa11
from package2 import module2
from package2.subpackage22 import module22

def func():
    print("========main.py============")
    print("Module Name:", __name__)
    print("Module File path:", __file__)
    print("Module Doc:", __doc__)
    print("Module Package:", __package__)
    print("Module Loader for import:", __loader__)
    print("Module Spec info:", __spec__)

func()

print(test.func())

# print(module1.func())
# print(module11.func())
print(a1.func())
print(aa11.func())

print(module2.func())
print(module22.func())












