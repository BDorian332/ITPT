from itpt.core import Model

class ExampleModel(Model):
    def __init__(self):
        super().__init__()
        self._metadata["name"] = "Example"
        self._metadata["description"] = "An example"
        self._metadata["version"] = 0

    def get_input_spec(self):
        print("get_input_spec called")
        return "get_input_spec"

    def load(self, weights_path=None):
        print(f"load called with weights_path={weights_path}")
        self._loaded = True

    def convert(self, image):
        self.ensure_loaded()
        print(f"convert called on image={image}")
        return "convert"
