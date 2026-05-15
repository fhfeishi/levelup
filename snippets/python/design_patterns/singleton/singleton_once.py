class model_zoo:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__init_once(*args, **kwargs)
        return cls._instance
    
    def __init_once(self):
        self.model = self.load_model()
    
    def load_model(self):
        print("model loading ... --init_once method")
        return "model loaded! --init_once method"
                
model = model_zoo()
print(model.model)

