# check version


# new
from importlib.metadata import version, PackageNotFoundError
from packaging.version import Version, parse
# python > 3.8 
# 版本号字符串比大小，这里用官方的工具转一下再比较
def check_version(current: str = "0.0.0",
                  minimum: str = "0.0.0",
                  name: str = "version ",
                  pinned: bool = False) -> bool:
    try:
        # 使用 importlib.metadata 来获取当前版本
        current_version = version(name)
        current = current_version if current == "0.0.0" else current
    except PackageNotFoundError:
        # 如果找不到包，说明可能未安装
        print(f"{name} package not found")
        return False
    # 版本号比较
    result = (current == minimum) if pinned else (current >= minimum)
    return result


"""
# old   # 
import pkg_resources as pkg
def check_version(current: str='0.0.0',
                  minimum: str='0.0.0',
                  name: str='version',
                  pinned: bool=False) -> bool:
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)
    return result
"""
if  __name__ ==  '__main__':
    import PIL
    k = check_version(PIL.__version__, '9.2.0', "pillow")
    print(f"PIL.__version__ > '9.2.0' is {k}")
    
    print(f"{PIL.__version__ = }")     # 9.4.0
    print(f"{version('pillow') = }")   # 9.4.0

    # pip show pillow  -> pillow 10.4.0


