from itpt.core import Model

class ExampleModel(Model):
    def __init__(self):
        super().__init__()
        self._metadata["name"] = "Example"
        self._metadata["description"] = "An example"
        self._metadata["version"] = 0

    def load(self):
        print("load called")
        self._loaded = True

    def convert(self, image_path):
        self.ensure_loaded()
        print(f"convert called on image_path={image_path}")
        return "newick"
