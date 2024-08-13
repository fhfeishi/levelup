class ModelZooMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]
    
class ModelZoo(metaclass=ModelZooMeta):
    def __init__(self):
        self.model = self.load_model()
    
    def load_model(self):
        print("mdoel loading ... --meta method")
        return "model laoded --meta method"

models = ModelZoo()
print(models)