def func():
    print("========module2.py============")
    print("Module Name:", __name__)
    print("Module File path:", __file__)
    print("Module Doc:", __doc__)
    print("Module Package:", __package__)
    print("Module Loader for import:", __loader__)
    print("Module Spec info:", __spec__)

# from main import func as mfunc
# print(mfunc())