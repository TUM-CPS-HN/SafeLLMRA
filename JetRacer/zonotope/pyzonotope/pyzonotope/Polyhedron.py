import numpy as np 
from pyzonotope.Point import Point 
from scipy.spatial import ConvexHull 

class Ineq():
    # Class that denotes the parameters of the inequality A * x <= b
    def __init__(self, A = None, b = None):
        self.A = A 
        self.b = b 
        self.lin = np.array([])

class Polyhedron():
    def __init__(self, *args):
        self.V = None
        self.H = None 
        self.P = Ineq()
        self.R = None 
        self.lb = None 
        self.ub = None 
        self.Ae = None 
        self.be = None 
        self.He = None 
        self.data = None 
        self.has_vrep = True
        self.internal_pt = Point()
        if(len(args) == 1):
            if(isinstance(args[0], np.ndarray)):
                self.V = args[0] 
            else:
                raise Exception("Polyhedrons with vertices are the only implemented type at the moment")
        self.compute_HRep()

    def compute_HRep(self):
        # TODO: do VRep availability checks and do minimal representation for the VRep
        self.interior_point()
        x0 = self.internal_pt.x 
        k = ConvexHull(self.V).simplices 
        d = self.V.shape[1]
        dummy_P = Ineq()
        dummy_P.A = np.zeros((k.shape[0], d))
        dummy_P.b = np.ones((k.shape[0], 1))
        
        for i in range(k.shape[0]):
            P = self.V[k[i, :], :]
            
            W = np.hstack((P, -1 * np.ones((d, 1))))
            AB, _ = np.linalg.qr(W.T, 'complete')
            
            a = AB[:d, -1]
            b = AB[-1, -1]
            
            if(np.dot(a.T, x0) > b):
                a *= -1 
                b *= -1
            dummy_P.A[i, :] = a.T 
            dummy_P.b[i] = b 
        self.P = dummy_P
        self.has_hrep = True

    def interior_point(self):
        # TODO: add support for H-Representation
        result = Point()
        if(self.has_vrep):
            result.x = np.mean(self.V, axis=0).T 

        self.internal_pt = result
