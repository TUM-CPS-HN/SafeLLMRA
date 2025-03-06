import numpy as np

class ConZonotope():
    def __init__(self, *args):
        self.A = np.array([])
        self.b = np.array([])

        if(len(args) == 1):
            self.Z = args[0].Z
        else:
            raise Exception("Constraints are not yet implemented in a ConZonotope")
