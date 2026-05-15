
def model_zoo(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@model_zoo
class Model_zoo:
    def __init__(self):
        self.model = self.load_model()
    def load_model(self):
        print("model loading ... --decorator method")
        return "model loaded --decorator method"

models = Model_zoo()
print(models.model)