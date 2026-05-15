# check version

# python 3.8 以后
from importlib.metadata import version
from packaging.version import Version, parse

def check_version(current: None,
                  minimum: str='0.0.0',
                  pkg_name = None,
                  name = 'version',
                  pinned: bool = False) -> bool:
    """
    Args:
        current (str, None) : package_version str(version) or pkgname.__version__. Defaults to '0.0.0'.
        minimum (str)       : str(min_version). Defaults to '0.0.0'.
        pkg_name (str, None): str(pkg_name) or None . Defaults to None.
        pinned (bool)       : True current==minimum, False current>=minimum. Defaults to False.
    Returns:
        bool: 
            True if pinned: current==minimum else: False
            True if not pinned: current>=minimum else: False
    """
    assert current is not None or pkg_name is not None, f"current and pkg_name are None !!!"
    if not isinstance(current, str):
        current = str(current)
        
    if pkg_name is not None:
        current_b = version(str(pkg_name))
        if current != current_b:
            print(f"package{pkg_name} is unequal in diffierent ways, pls reinstall {pkg_name}")
        else:
            current = current_b

    if name == 'version':
        current, minimum = (Version(x) for x in (current, minimum))
    else:
        current, minimum = (parse(x) for x in (current, minimum))
    
    result = (current == minimum) if pinned else (current >= minimum)
    return result  
    
# # old   #   python 3.8  以前  可能不会报错deprecated error ?
# import pkg_resources as pkg
# def check_version_old(current: str='0.0.0',
#                   minimum: str='0.0.0',
#                   name: str='version',
#                   pinned: bool=False) -> bool:
#     current, minimum = (pkg.parse_version(x) for x in (current, minimum))
#     result = (current == minimum) if pinned else (current >= minimum)
#     return result

if  __name__ ==  '__main__':
    import PIL
    k = check_version(PIL.__version__, '9.2.0', "pillow")
    
    # k = check_version_old(PIL.__version__, "9.2.0")
    
    print(f"PIL.__version__ > '9.2.0' is {k}")
    
    print(f"{PIL.__version__ = }")     # 9.4.0
    print(f"{version('pillow') = }")   # 9.4.0

    # pip show pillow  -> pillow 10.4.0


