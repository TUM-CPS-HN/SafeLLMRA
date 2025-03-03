import numpy as np
import matplotlib.pyplot as plt
from pyzonotope.contSet import contSet
from pyzonotope.Zonotope import Zonotope
from pyzonotope.IntervalMatrix import IntervalMatrix

class MatZonotope():
    def __init__(self, *args):
        self.dim = 1
        self.gens = 0
        self.center = 0
        self.generators = np.array([])

        if(len(args) == 0):
            self.center = 0
            self.generators = np.array([])

        elif(len(args) == 1):
            raise Exception("One argument isn't supported yet")
            #if(type(args[0]) == Zonotope):
            #
            #    c = args[0].center()
            #    G = args[1].generators()
                
                
        elif(len(args) == 2):
            self.center = args[0]
            self.generators = args[1]

            self.dim = np.max(self.center.shape)
            self.gens = np.max(self.generators.shape)


    def __add__(self, operand):
        #print("GEEZ", isinstance(operand, np.ndarray), type(operand))
        #print("*"*80, "\n",self.center.shape)
        if(isinstance(operand, np.ndarray)):
            Z = MatZonotope(self.center, self.generators)
            #Z = Z.copy(self)
            Z.center = Z.center + operand
            #print("*"*80, "done")
            return Z
        else:
            #print(type(operand))
            print("MatZonotope addition operation is currently only implemented for addition with matrices")
            raise NotImplementedError


    __radd__ = __add__

    def __mul__(self, factor):
        if(type(factor) == np.ndarray):
            result = MatZonotope(self.center, self.generators)
            result.center = np.dot(result.center, factor)
            generators = []
            #print(result.generators.shape, factor.shape)
            result.generators = np.squeeze(result.generators)
            for i in range(self.gens):
                generators.append(np.dot(result.generators[i], factor))

            result.generators = np.array(generators)
            return result

        elif(isinstance(factor, Zonotope)):
            result = Zonotope()
            result.copy(factor)
            Znew = np.dot(self.get_centers(), factor.Z)
            for i in range(self.gens):
                Zadd = np.dot(self.generators[i], factor.Z)
                Znew = np.hstack((Znew, Zadd))
            result.Z = Znew 
            return result

        elif(isinstance(factor, float) or isinstance(factor, int)):
            result = MatZonotope(self.center, self.generators)
            result.center = result.center * factor
            #for i in range(result.gens):
            result.generators = result.generators * factor             
            return result

        else:
            raise Exception("Multiplication is only supported for matrices and zonotopes.")

    __rmul__ = __mul__


    def get_centers(self):
        return self.center

    def get_generators(self):
        return self.generators

    def __str__(self):
        S = "dimension: {} gens: {} Centers Shape: {} Generators Shape: {} \n Centers: \n {} \n Generators: \n {}".format(self.dim, 
        self.gens, self.center.shape, self.generators.shape, self.center, self.generators)
        return S


    def copy(self, matzono):
        result = MatZonotope()
        result.dim = matzono.dim
        result.gens = matzono.gens
        result.center = np.copy(matzono.center)
        result.generators = np.copy(matzono.generators)
        return result


    def interval_matrix(self, *args):
        matZ = MatZonotope(self.center, self.generators)
        #matZ = matZ.copy(self)
        setting = []
        if(len(args) == 1):
            setting = args[1]

        #print("-"*10, matZ.gens,matZ.generators.shape, self.generators.shape)
        delta = np.abs(matZ.generators[0])
        for i in range(1, matZ.gens, 1):
            delta = delta + np.abs(matZ.generators[i])


        return IntervalMatrix(matZ.center, delta, setting)
