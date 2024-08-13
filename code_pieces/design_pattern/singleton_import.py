class model_zoo:
    def __init__(self):
        self.model = self.load_model()
    def load_model(self):
        print("model loading ... --module import method")
        return "model loaded --module import method"
module_ = model_zoo()

print(module_.model)
