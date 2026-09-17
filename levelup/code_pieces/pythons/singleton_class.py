import threading

class model_zoo:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(model_zoo, cls).__new__(cls)
                    cls._instance.initialize(*args, **kwargs)
        return cls._instance
    
    def initialize(self, *args, **kwargs):
        self.model = self.load_model()
    
    def load_model(self):
        print("model loading ... --threding method")
        return "model loaded ==threading method"
    
models = model_zoo()
print(models)
    