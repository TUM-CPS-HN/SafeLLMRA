import numpy as np 
from pyzonotope.Interval import Interval


class IntervalMatrix():
    def __init__(self, *args):
        self.dim = 1
        self.Inf = []
        self.Sup = []
        self.int = []
        self.setting = 'sharpivmult'

        if(len(args) == 1):
            if(type(args[0]) == IntervalMatrix):
                self = args[0]
            else:
                self.dim = args[0].shape[0]
                self.Inf = args[0]
                self.Sup = args[0]
                self.int = Interval(args[0], args[0])
                self.setting = []
        elif(len(args) == 2):
            self.dim = args[0].shape[0]
            matrix_delta = np.abs(args[1])
            self.Inf = args[0] - matrix_delta
            self.Sup = args[0] + matrix_delta
            self.int = Interval(self.Inf, self.Sup)
            self.setting = []
        
        elif(len(args) == 3):
            self.dim = args[0].shape[0]
            matrix_delta = np.abs(args[1])
            self.Inf = args[0] - matrix_delta
            self.Sup = args[0] + matrix_delta
            self.int = Interval(self.Inf, self.Sup)
            self.setting = args[2]

        else:
            raise Exception("Invalid Number of inputs")