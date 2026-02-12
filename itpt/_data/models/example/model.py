from itpt.core import Model
from itpt.core.newick import Newick, NewickInternal

class ExampleModel(Model):
    def __init__(self):
        super().__init__()
        self._metadata["name"] = "Example"
        self._metadata["description"] = "An example"
        self._metadata["version"] = 0

    def load(self):
        print("load called")
        self._loaded = True

    def convert(self, path_or_array):
        self.ensure_loaded()
        print(f"convert called on path_or_array={path_or_array}")
        leaf1 = NewickInternal(name="A", length=1.0)
        leaf2 = NewickInternal(name="B", length=1.0)
        return Newick(internals=[leaf1, leaf2])
