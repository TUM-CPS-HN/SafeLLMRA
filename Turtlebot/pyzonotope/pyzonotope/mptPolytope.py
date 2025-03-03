import numpy as np
import matplotlib.pyplot as plt
from pyzonotope.Polyhedron import Polyhedron

from pyzonotope.ConZonotope import ConZonotope
from scipy.optimize import linprog 
from scipy.linalg import block_diag
import math 

class mptPolytope():
    def __init__(self, *args):
        self.p = np.array([])
        self.halfspace = np.array([])

        if(len(args) == 1):
            if(isinstance(args[0], mptPolytope)):
                self.p = np.copy(args[0].p)
                self.halfspace = np.copy(args[0].halfspace)
            else:
                self.p = Polyhedron(np.array(args[0]))
        elif(len(args) == 2):
            raise Exception ("More than one argument is not implemented yet.")
    
    def ConZonotope(self):
        pass
    
    def is_intersecting(self, obj, type = "exact"):
	#from .Zonotope import Zonotope
        if(type != "exact"):
            raise Exception("Approximate intersection methods are not implemented yet.")
        if(isinstance(obj, Zonotope.Zonotope)):
            return self.intersect_polyConZono(ConZonotope(obj))

    def intersect_polyConZono(self, obj):
        H = self.p.P.A 
        d = self.p.P.b
        #H = np.array([H[0], H[2], H[3], H[1]])
        #d = np.array([d[0], d[2], d[3], d[1]])
        c = obj.Z[:, 0:1]
        G = obj.Z[:, 1:]

        n = len(c)
        m = G.shape[1]
        p = H.shape[0]

        A = np.vstack((np.hstack((H, -1 * np.eye(p))), np.hstack((np.zeros((p, n)), -1 * np.eye(p)))))
        b = np.vstack((d, np.zeros((p, 1))))

        A = block_diag(A, np.vstack((np.eye(m), -1 * np.eye(m))))
        b = np.vstack((b, np.ones((m, 1)), np.ones((m, 1))))


        Aeq = np.hstack((np.eye(n), np.zeros((n, p)), -1 * G))
        beq = c 

        # Construct the objective function 
        f = np.vstack((np.zeros((n, 1)), np.ones((p, 1)), np.zeros((m, 1))))

        print(f, A, b)
        res = linprog(f.flatten(), A, b, Aeq, beq, bounds = (None, None)) 

        isIntersecting = True 

        print(res["status"], res["fun"], res["status"] > 0, res["fun"] > (math.pow(10, -5)))
        if(res["status"] > 0 or res["fun"] > math.pow(10, -5)):
            isIntersecting = False 

        return isIntersecting

