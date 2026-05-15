# import os
# current_dir = os.path.dirname(os.path.abspath(__file__))
# print(current_dir)

print("Module Name:", __name__)  # __main__   # package2.module3
print("Module File:", __file__)
print("Module Doc:", __doc__)
print("Module Package:", __package__)
print("Module Loader:", __loader__)
print("Module Spec:", __spec__)

# # 绝对路径
# import sys
# sys.path.append(r"D:\codespace\creations\levelup\my_python\zzz_import\package1")
# from subpackage.module2 import func2

# # 相对路径
# import sys
# sys.path.append('../')
# from package1.subpackage.module2 import func2

# 无法索引到module2.py文件，会error
from ..package1.subpackage.module2 import func2   
def call_sum():
    func2()
call_sum()  # attempted relative import beyond top-level package
