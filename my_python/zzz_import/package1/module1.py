

import sys 
print(sys.path)
def func1():
    print("package1 module1 func1")
    print("Module Name:", __name__)
    print("Module File:", __file__)
    print("Module Doc:", __doc__)
    print("Module Package:", __package__)
    print("Module Loader:", __loader__)
    print("Module Spec:", __spec__)


