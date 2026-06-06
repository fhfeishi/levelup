# Python import 机制梳理

这个工程是一组小实验，用来观察 Python 从“文件系统里的 `.py` 文件”到“运行时里的 module 对象”之间做了什么。主要测试逻辑集中在根目录的 `test.py`，包内的 `test_a.py` / `test_b.py` / `test_c.py` 主要作为“被不同命令运行的样本模块”。

## 推荐运行顺序

在项目根目录执行：

```powershell
python test.py
python package_a/test_a.py
python -m package_a.test_a
python package_a/sub_c/test_c.py
python -m package_a.sub_c.test_c
python package_b/test_b.py
python -m package_b.test_b
```

也可以试试：

```powershell
python module_1.py
python -m module_1
python package_a/sub_c/sub_module_c.py
python -m package_a.sub_c.sub_module_c
```

## import 的大体流程

`import name` 不是简单的“复制代码”。更接近下面这条流水线：

1. 检查 `sys.modules`：如果之前已经导入过同名模块，直接复用缓存对象。
2. 根据模块名寻找 `ModuleSpec`：主要通过 `sys.meta_path` 上的 finder 完成。普通文件和包最终会落到 `sys.path` 里的目录搜索。
3. 创建 module 对象：模块也是对象，类型通常是 `module`。
4. 先放进 `sys.modules`：这样可以处理循环导入。
5. 执行模块代码：顶层代码会运行，函数体不会自动运行，类体会在定义类时运行。
6. 绑定名字：不同 import 语句最后绑定到当前命名空间的名字不同。

所以 import 同时做了三件事：

- 找到代码：通过 finder/spec/loader 和搜索路径。
- 执行代码：把文件顶层语句跑一遍，生成函数、类、变量。
- 绑定名字：把模块对象或模块里的属性绑定到当前作用域。

## 几种 import 语句绑定了什么

```python
import module_1
```

会导入并绑定名字 `module_1`。之后用 `module_1.f1` 访问模块里的属性。

```python
from module_1 import f1
from module_1 import c1 as c
```

会导入 `module_1`，然后把 `module_1.f1`、`module_1.c1` 这两个属性绑定到当前作用域里的 `f1` 和 `c`。它不一定会绑定 `module_1` 这个名字。

```python
import package_a.module_a
```

会导入包和子模块，但在当前作用域绑定的是顶层名字 `package_a`。访问时写 `package_a.module_a`。

```python
from package_a import module_a
```

会把 `package_a.module_a` 这个子模块对象直接绑定为当前作用域的 `module_a`。

```python
from . import module_a
from .sub_c import sub_module_c
```

这是显式相对导入。它依赖当前模块的 `__package__`，所以只有当前文件确实作为包内模块运行时才可靠。

## package 为什么也是 module

包本质上也是 module 对象。常规包的标志是目录里有 `__init__.py`。

导入 `package_a` 时，Python 执行的是：

```text
package_a/__init__.py
```

包对象相比普通模块多了一个关键属性：

```python
package_a.__path__
```

`__path__` 表示“查找这个包的子模块时，从哪些目录里找”。普通模块没有 `__path__`，因为普通模块不能再包含子模块。

## 没有 `__init__.py` 的 package_b

`package_b` 故意没有 `__init__.py`。在 Python 3 中，这种目录仍然可以作为 namespace package 被导入：

```python
import package_b
import package_b.module_b
from package_b import module_b
```

它和 `package_a` 这种常规包的差别是：

- 导入 `package_a` 会执行 `package_a/__init__.py`。
- 导入 `package_b` 没有任何 `__init__.py` 可执行，所以不会有包初始化代码。
- `package_b.__file__` 通常是 `None`。
- `package_b.__spec__.loader` 通常是 namespace package 的 loader。
- `package_b.__path__` 仍然存在，因为它仍然需要告诉 import 系统“从哪些目录找子模块”。

这说明“包对象”和“有没有初始化文件”是两个相关但不同的问题：没有 `__init__.py` 时，包仍然可以存在，但少了初始化代码这个钩子。

## package_a 如何导入 package_b 和 sub_c

`package_a` 和 `package_b` 是项目根目录下的两个顶层包。顶层兄弟包之间应该使用绝对导入：

```python
# package_a/module_a.py
from package_b import module_b
```

这要求“项目根目录”在 `sys.path` 中。推荐从项目根目录用 `-m` 运行包内模块：

```powershell
python -m package_a.test_a
python -m package_b.test_b
```

不要优先使用：

```powershell
python package_a/test_a.py
```

因为这种文件路径运行方式会把 `sys.path[0]` 设置为 `package_a` 目录本身，项目根目录反而不在最前面，`import package_b` 就可能找不到。

`package_a` 导入自己的子包 `sub_c` 有两种常见写法：

```python
# 相对导入：适合包内部表达“这是我的子模块”
from .sub_c import sub_module_c

# 绝对导入：适合全项目统一按完整包名读
from package_a.sub_c import sub_module_c
```

相对导入依赖当前模块的 `__package__`。如果 `package_a/module_a.py` 是被正常 import 进来的，它的 `__package__ == "package_a"`，相对导入成立。如果你直接运行包内文件，包语境容易丢失。

## 为什么 dir(module) 和 dir(package) 有信息

`dir(obj)` 是内置的 introspection 工具。对模块对象来说，它返回模块命名空间里的名字，接近于：

```python
sorted(obj.__dict__.keys())
```

模块被 import 时，顶层赋值、函数定义、类定义、导入进来的名字，以及 Python 自动放进去的内置属性，都会进入模块的 `__dict__`，因此 `dir(module)` 能看到它们。

包也是 module 对象，所以 `dir(package)` 同理能看到包的 `__dict__`。不过只导入 `package_a` 时，不会自动导入所有子模块；只有 `__init__.py` 里定义或导入过的名字，以及 import 系统自动设置的名字，会出现在 `dir(package_a)` 里。

常见自动属性：

| 属性 | 含义 |
| --- | --- |
| `__name__` | 模块的规范名称；入口脚本会是 `"__main__"` |
| `__package__` | 当前模块所属包名；显式相对导入靠它定位 |
| `__file__` | 源文件路径 |
| `__cached__` | `.pyc` 缓存路径 |
| `__loader__` | 负责加载模块的 loader |
| `__spec__` | 模块导入规范，包含 origin、loader、包搜索路径等 |
| `__path__` | 只有包通常有；用于查找子模块 |
| `__builtins__` | 模块可用的内置命名空间 |

## sys.path、脚本路径和 -m 的关键差异

`sys.path` 是当前 Python 进程的模块搜索路径，不是某个 `.py` 文件私有的一份配置。某个模块执行 import 时，看到的是当时这个进程里的同一个 `sys.path`。

不过入口命令会初始化 `sys.path[0]`，所以看起来像“某个文件有自己的 import 查找路径”。更准确地说：

- `sys.path` 是进程级、全局、可变的。
- `python some/file.py` 会把 `some/file.py` 所在目录放到 `sys.path[0]`。
- `python -m package.module` 通常会把当前工作目录放到 `sys.path[0]`。
- 导入包的子模块时，除了顶层的 `sys.path`，还会用父包的 `package.__path__` 去找子模块。

所以 `sys.path` 不是“该模块的查找路径”，而是“当前解释器后续 import 的全局查找路径”。模块当然可以读取甚至修改它，但修改会影响之后所有导入，这在工程里要非常克制。

### `python test.py`

当你运行根目录下的脚本：

```powershell
python test.py
```

`sys.path[0]` 通常是脚本所在目录，也就是项目根目录。因此：

```python
import module_1
import package_a.module_a
```

都能找到。

### `python -m module_1`

当你运行：

```powershell
python -m module_1
```

Python 会从当前工作目录所在的 import 搜索路径中查找 `module_1`，然后把它作为 `__main__` 执行。适合“按模块名运行”。

### `python package_a/test_a.py`

当你用文件路径运行包内文件：

```powershell
python package_a/test_a.py
```

`sys.path[0]` 会变成 `package_a` 目录本身，而不是项目根目录。此时容易出现两类问题：

- `import module_1` 可能失败，因为项目根目录不在 `sys.path[0]`。
- `from . import module_a` 会失败，因为这个文件是作为孤立脚本运行，`__package__` 通常是 `None`。

这就是很多“明明文件在那里，import 却失灵”的根源：代码文件位置、当前工作目录、启动命令共同决定了 import 语境。

### `python -m package_a.test_a`

当你运行：

```powershell
python -m package_a.test_a
```

Python 先从当前工作目录找 `package_a.test_a`，然后把它作为包内模块运行。此时：

```python
__name__ == "__main__"
__package__ == "package_a"
```

所以项目根目录通常在 `sys.path` 中，绝对导入能找到顶层模块；`__package__` 也正确，显式相对导入也能工作。

实践建议：包内模块如果需要相对导入，优先用 `python -m package.module` 从项目根目录运行。

## 好用的观察工具

- `dir(obj)`：看对象暴露了哪些名字。
- `vars(obj)`：看对象的 `__dict__`；对模块尤其直观。
- `type(obj)`：确认对象类型，例如模块、函数、类、实例。
- `id(obj)`：观察两个名字是否指向同一个对象。
- `hasattr(obj, name)` / `getattr(obj, name)`：动态检查和取属性。
- `callable(obj)`：判断对象是否可调用。
- `globals()` / `locals()`：观察当前作用域名字绑定。
- `sys.modules`：import 缓存；同名模块通常只执行一次。
- `importlib.util.find_spec(name)`：在不真正执行模块代码的前提下，观察 import 系统能否找到模块。
- `importlib.import_module(name)`：用字符串动态导入模块，框架和插件系统常用。
- `pkgutil.iter_modules(path)`：枚举某个路径下的传统模块和包；没有 `__init__.py` 的 namespace package 不一定会出现在这里。
- `inspect.getmembers(obj)`：按条件筛选成员，例如函数、类。

## 一个工业软件设计视角

Python import 的设计很精妙：它把“定位、加载、执行、缓存、名字绑定”拆成了可替换的层。

- `sys.path` / `sys.meta_path` 负责扩展搜索来源。
- `ModuleSpec` 描述“找到的东西是什么、怎么加载”。
- loader 负责实际创建和执行模块。
- `sys.modules` 作为全局缓存，保证模块单例感，并缓解循环导入。
- 普通语法 `import x` 和程序化 API `importlib.import_module()` 共享同一套机制。

这就是为什么 Python 可以同时支持普通 `.py` 文件、包、内置模块、扩展模块、zipimport、虚拟环境、插件系统等：语法简单，但底层协议是开放的。
