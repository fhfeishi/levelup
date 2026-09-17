# singleton pattern 单例模式
# 确保一个类只有一个实例，并提供一个全局访问点。
# 在Python中单例模式可以通过多种方式。

# -1-------singleton pattern __new__  ----------------
# # img_processor.py
class ImageProcessor(object) :
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__init_once(*args, **kwargs)
        return cls._instance
    
    def __init_once(self):
        self.model1 = self.load_model1()
        
    def load_model1(self):
        return "model1 loaded"

# -2-------import----------------
# singleton.py
class Singleton:
    def __init__(self):
        self.model1 = self.load_model()
    def load_model(self):
        return "model loaded"
instance = Singleton()
# # use  # from singleton import instance


# -3------decorator--------------
# singleton.py
def singleton(cls):
    instance = {}
    def get_instance(*args, **kwargs):
        if cls not in instance:
            instance[cls] = cls(*args, **kwargs)
        return instance[cls]
    return get_instance
@singleton
class Singleton:
    def __init__(self):
        self.model = self.load_model()
    def load_model(self):
        return "model loaded"
# use
# # other_module.py
"""
from singleton import Singleton

instance1 = Singleton()
instance2 = Singleton()
当 Singleton() 被首次调用时，singleton 装饰器会检查 Singleton 类是否已经有实例存在。如果没有，它会创建一个新的实例，并将其存储在 instances 字典中。
对于后续的 Singleton() 调用，装饰器会从 instances 字典中返回已经创建的实例，确保全局只有一个 Singleton 类的实例存在。
print(instance1 == instance2) # True
"""


# -4------使用类变量和同步-----------------------
# # 在多线程环境中确保单例模式的线程安全，可以使用同步机制。
import threading
class Singleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(Singleton, cls).__new__(cls)
                cls._instance.initialize(*args, **kwargs)
        return cls._instance
    def initialize(self, *args, **kwargs):
        self.model = self.load_model()
    def load_model(self):
        return "model loaded"
# 这种方法使用了线程锁来确保即使在多线程环境中，实例的创建也是线程安全的。



# -5-------使用元类------------------------------
# # 元类是 Python 的高级特性，可以控制类的创建过程。
# 使用元类的方法可以在更底层控制类的实例化过程，使其只能产生一个实例。
# 文件：singleton.py
class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class Singleton(metaclass=SingletonMeta):
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        return "Model loaded"

"""
# main.py
from singleton import Singleton

instance1 = Singleton()
print(instance1.model)  # 输出: Model loaded

instance2 = Singleton()
print(instance2.model)  # 输出: Model loaded
print(instance1 is instance2)  # 输出: True, 表明 instance1 和 instance2 是同一实例
"""
    
