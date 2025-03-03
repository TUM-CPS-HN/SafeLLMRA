import numpy as np
import matplotlib.pyplot as plt
from pyzonotope.contSet import contSet
from numpy import linalg as LA
from scipy.linalg import block_diag
from pyzonotope.Interval import Interval
from pyzonotope.mptPolytope import mptPolytope
from pyzonotope.ConZonotope import ConZonotope
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
#import ray 
#from multiprocessing import get_context

class Zonotope():
    def __init__(self, *args):
        self.Z = np.array([])
        self.half_space = np.array([])
        self.cont_set = np.array([])

        if(len(args) == 0):
            self.cont_set = contSet()

        elif(len(args) == 1):
            if(len(np.shape(args[0])) == 1 or isinstance(args[0], Interval)):
                self.cont_set = contSet(1)
            else:
                self.cont_set = contSet(np.shape(args[0])[0])

            if(type(args[0]) == Zonotope):
                self.copy(args[0])
            
            elif(type(args[0]) == Interval):
                center = 0.5 * (args[0].inf + args[0].sup)
                G = np.diag(0.5 * (np.array(args[0].sup) - np.array(args[0].inf)).flatten())
                #print("tataaa", center, G)
                self.Z = np.hstack((center.reshape(-1, 1), G))
                self.half_space = np.array([])

            else:
                self.Z = np.copy(args[0])
                self.half_space = np.array([])


        elif(len(args) == 2):
            if(len(np.shape(args[0])) == 1 or len(np.shape(args[0])) == 0):
                self.cont_set = contSet(1)
            else:
                self.cont_set = contSet(np.shape(args[0])[0])
            
            self.Z = np.hstack([args[0], args[1]])
            self.half_space = np.array([])
        
    def center(self):
        #print(self.Z)
        if(len(self.Z.shape) == 1):
            return np.array([self.Z[0]]).reshape(-1, 1)
        else:
            return self.Z[:, 0:1]

    def generators(self):
        if(len(self.Z.shape) == 1):
            return np.array([self.Z[1]]).reshape(-1, 1)
        else:
            return self.Z[:, 1:]


    def copy(self, zon):
        self.Z = np.copy(zon.Z) 
        self.half_space = np.copy(zon.half_space )
        self.cont_set = zon.cont_set
        return self

    def __add__(self, operand):
        Z = Zonotope(self.Z)
        #Z = Z.copy(self)

        if(type(operand) == Zonotope):
            Z.Z[:, 0:1] = Z.Z[:, 0:1] + operand.Z[:, 0:1]
            Z.Z = np.hstack((Z.Z, operand.Z[:, 1:]))
        
        elif(type(operand) == np.ndarray and (len(operand.shape) == 1 or operand.shape[1] == 1)):
            Z.Z[:, 0:1] = Z.Z[:, 0:1] + operand

        elif(type(operand) == np.ndarray):
            Z.Z = Z.Z + Zonotope(operand)

        else:
            raise Exception("Invalid argument for addidtion")


        return Z 

    def __sub__(self, operand):
        Z = Zonotope(self.Z)
        #Z = Z.copy(self)
        if(type(operand) == Zonotope):
            raise Exception("Zonotopes subtraction is not supported when both operands are zonotopes")
        elif(type(operand) == np.ndarray and (len(operand.shape) == 1 or operand.shape[1] == 1)):
            Z.Z[:, 0:1] = self.Z[:, 0:1] - operand

        return Z

    def __mul__(self, operand):
        #print("1")
        if(isinstance(operand, float) or isinstance(operand, int)):
            Z = Zonotope(self.Z)
            #Z = Z.copy(self)
            Z.Z = Z.Z * operand
            return Z
        else:
            Z = Zonotope(self.Z)
            #Z = Z.copy(self)
            #print("2", Z.Z)
            Z.Z = np.dot(operand, Z.Z.reshape((operand.shape[1], -1))) 
            #print(Z)
            #print("3")
            return Z
    
    __rmul__ = __mul__   # commutative operation

    def reduce(self, option, *args):
        if(len(args) == 0):
            order = 1
            filterLength = []

        elif(len(args) == 1):
            order = args[0]
            filterLength = []

        elif(len(args) == 2):
            order = args[0]
            filterLength = args[1]
        
        elif(len(args) == 3):
            order = args[0]
            filterLength = args[1]
            method = args[2]

        elif(len(args) == 4):
            order = args[0]
            filterLength = args[1]
            method = args[2]
            alg = args[3]

        if(option == "girard"):
            Zred = self.reduce_girard(order)
        else:
            raise Exception("Other Reduction methods are not implemented yet")

        return Zred
        

    def reduce_girard(self, order):
        Zred = Zonotope(self.Z)
        #Zred = Zred.copy(self)

        center, Gunred, Gred = Zred.picked_generators(order)
        #print('shapes', center.shape, Gunred.shape, Gred.shape)
        if(Gred.size == 0):
            Zred.Z = np.hstack((center, Gunred))
        else:
            d = np.sum(np.abs(Gred), axis=1)
            Gbox = np.diag(d)
            #print("YARAPPP", center.shape, Gunred.shape, Gbox.shape)
            center = center.reshape((center.shape[0], -1))
            Gunred = Gunred.reshape((center.shape[0], -1))
            Gbox = Gbox.reshape((center.shape[0], -1))
            #print("YARAPPP", center.shape, Gunred.shape, Gbox.shape)
            Zred.Z = np.hstack((center, Gunred, Gbox))#np.array([[center], [Gunred], [Gbox]])
        
        return Zred

    def picked_generators(self, order):
        Z = Zonotope(self.Z)
        #Z = Z.copy(self)

        c = Z.center()
        G = Z.generators()

        Gunred = np.array([])
        Gred = np.array([])

        if(np.sum(G.shape) != 0):
            G = self.nonzero_filter(G)
            d, nr_of_gens = G.shape
            if(nr_of_gens > d * order):
                #h = LA.norm(G, 1) - LA.norm(G, np.inf)
                h = np.apply_along_axis(lambda row:np.linalg.norm(row,ord=1), 0, G)  - \
                    np.apply_along_axis(lambda row:np.linalg.norm(row,ord=np.inf), 0, G)
                #print('hhh', h.shape)
                n_unreduced = np.floor(d * (order - 1))

                n_reduced = int(nr_of_gens - n_unreduced)
                #print("arg_partition")
                #print(h, h.shape)
                #print('*'*80)
                #print(n_reduced)
                #print('*'*80)
                idx = np.argpartition(h, n_reduced - 1)
                Gred   = G[:, idx[: n_reduced]]
                Gunred = G[:, idx[n_reduced: ]]

            else:
                Gunred = G 

        
        return c, Gunred, Gred


    def nonzero_filter(self, generators):
        idx = np.argwhere(np.all(generators[..., :] == 0, axis=0))
        return np.delete(generators, idx, axis=1)

    def cart_prod(self, other):
        """
        Cart Product Function. IMPORTANT NOTE: THIS function doesn't take into account order. It's somewhat messed up.
        However, it works fine with the current implementation of reachability. The part that needs modification is the 
        numpy.ndarray or list part. That is the concatenation of the array or the list should be reversed as it depends on the 
        order of multiplication
        """
        if(isinstance(other, Zonotope)):
            center = np.vstack((self.center(), other.center()))
            G = block_diag(self.generators(), other.generators())
            Z = Zonotope(center, G)
            return Z
        elif(isinstance(other, np.ndarray) or isinstance(other, list)):
            other = np.array(other)
            center = np.vstack(( other, self.center()))
            G = np.vstack((np.zeros((other.shape[0], self.generators().shape[1])), self.generators()))
            result = Zonotope(center, G)
            return result
        else:
            raise Exception("cart products are only implemented if the two arguments are zonotopes")

    def __str__(self):
        S = "id: {} dimension: {} \n Z: \n {}".format(self.cont_set._id, self.cont_set._dimension, self.Z)
        return S 



    def project(self, proj):
        Z = Zonotope(self.Z)
        Z.Z = Z.Z[proj[0] - 1: proj[1] - 1, :]
        return Z 

    def polygon(self):
        #print("Hi")
       # self.Z[:, 1:] = self.nonzero_filter(self.Z[:, 1:])
        c = self.center()
        G = self.generators()

        n = G.shape[1]

        xmax = np.sum(np.abs(G[0, :]))
        ymax = np.sum(np.abs(G[1, :]))

        Gnorm = np.copy(G)
        Gnorm[:, np.where(G[1, :] < 0)] = G[:, np.where(G[1, :] < 0)] * -1

        angles = np.arctan2(Gnorm[1, :], Gnorm[0, :])

        angles[np.where(angles < 0)] = angles[np.where(angles < 0)] + 2 * np.pi 

        IX = np.argsort(angles)
        #print(Gnorm)
        p = np.zeros((2, n + 1))

        for i in range(n):
            p[:, i + 1] = p[:, i] + 2 * Gnorm[:, IX[i]]
        #print(p)
        p[0, :] = p[0, :] + xmax - max(p[0, :])
        p[1, :] = p[1, :] - ymax
        #print(p)
        p = np.vstack((np.hstack((p[0, :], p[0, -1] + p[0, 0] - p[0, 1:])),
                       np.hstack((p[1, :], p[1, -1] + p[1, 0] - p[1, 1:]))))
        
        #consider center
        p[0, :] = c[0] + p[0, :]
        p[1, :] = c[1] + p[1, :]

        return p


    def to_interval(self):
        result = Zonotope(self.Z)
        #result = result.copy(self)

        c = result.center()
        delta = np.sum(np.abs(result.Z), axis=1).reshape((-1, 1)) - np.abs(c)

        left_limit  = c - delta
        right_limit = c + delta 

        I = Interval(left_limit, right_limit)

        return I


    def plot(self, *args):
        dims = [1,2]
        linespec = 'b'
        filled = False

        if(len(args) == 2):
            dims = args[1]

        elif(len(args) >= 3):
            pass
        
        #Z = self.project(dims)

        V = self.polygon()

        if(V.shape[0] == 2):
            #print("In")
            xs = V[0, 1:]
            ys = V[1, 1:]
            colors = ["b", "g", "r", "m", "y", "c", "k"]
            plt.fill(xs, ys, np.random.choice(colors))
            plt.show()  
        #pass

    def rand_point(self):
        G = self.generators()
        factors = -1 + 2 * np.random.random((1, G.shape[1]))
        p = self.center() + np.reshape(np.sum(factors * G, axis=1), (G.shape[0], 1))
        return p
        
    def quad_map(self, Q):
        Z_mat = self.Z 
        dimQ = len(Q)
        gens = len(Z_mat[0, :]) - 1
        #print(0.5 * ((np.power(gens, 2)) + gens) + gens)
        C = np.zeros((dimQ, 1))
        G = np.zeros((dimQ, int(0.5 * ((np.power(gens, 2)) + gens) + gens)))
        
        Qnonempty = np.zeros((dimQ, 1))

        for i in range(dimQ):
            Qnonempty[i] = np.any(Q[i].reshape((-1, 1)))

            if(Qnonempty[i]):
                QuadMat = np.dot(Z_mat.T, np.dot(Q[i], Z_mat))

                G[i, 0 : gens - 1] = 0.5 * np.diag(QuadMat[1 : gens, 1 : gens])

                C[i, 0] = QuadMat[0, 0] + np.sum(G[i, 0 : gens - 1])

                quadMatoffdiag = QuadMat + QuadMat.T

                quadMatoffdiag = quadMatoffdiag.flatten()

                kInd = np.tril(np.ones((gens + 1, gens + 1)), -1) 
                #print(kInd)
                G[i, gens :] = quadMatoffdiag[kInd.flatten() == 1]

        if(np.sum(Qnonempty) <= 1):
            Zquad = Zonotope(C, np.sum(np.abs(G),1))
        else:
            Zquad = Zonotope(C, self.nonzero_filter(G))

        return Zquad


    def ConZonotope(self):
        return ConZonotope(self.Z)


    def is_intersecting(self, obj, *args):
        if(isinstance(obj, Zonotope)):
            if(len(args) == 0):
                return self.check_zono_intersection(obj)
            else:
                return self.check_zono_intersection(obj, args)
        elif(isinstance(obj, mptPolytope.mptPolytope)):
            return obj.is_intersecting(self, "exact")
        #elif(isinstance(obj, Zonotope)):
        #    return self.check_zono_intersection(obj)
    

    def check_zono_intersection(self, obj):
        
        c_1 = self.center()
        G_1 = self.generators()

        c_2 = obj.center()
        G_2 = obj.generators()

        c, G, A, b = self.intersect_zonotopes(obj)
        
        c, G, A, b = self.intersect_zonotopes(obj)

        LP_f,LP_A_ineq,LP_b_ineq,LP_A_eq,LP_b_eq = self.make_con_zono_empty_check_LP(A, b)
        #print("{} \n {} \n {} \n {} \n {}".format(LP_f, LP_A_ineq,LP_b_ineq,LP_A_eq,LP_b_eq))
        """
        def eq_con(x):
            global LP_A_eq
            global LP_b_eq 
            return A * x - b

        def ineq_con(x):
            global LP_A_ineq
            global LP_b_ineq

            return LP_A_ineq * x - LP_b_ineq 
        """
        def cost(x, *args):
            A = args[0] 
            result = np.dot(x.reshape((1, len(x))), A)[0]
            return result
        

        eq_con = LinearConstraint(LP_A_eq, LP_b_eq.flatten(), LP_b_eq.flatten())
        ineq_con = LinearConstraint(LP_A_ineq, np.array([np.NINF] * 24), LP_b_ineq.flatten())
        #cons = [{'type':'eq', 'fun': eq_con},
        #        {'type':'ineq', 'fun': ineq_con}]
        cons = [eq_con, ineq_con]
        #print(cons)
        #print(LP_f.shape, LP_A_ineq.shape, LP_b_ineq.shape, LP_A_eq.shape, LP_b_eq.shape)
        #print(LP_A_eq)
        #start_t = time.time()
        
        res = minimize(cost, np.zeros((13)), constraints = cons, args=(LP_f), method='trust-constr', options={'maxiter': 10})
        #print("hi")
        return
        #print("After min", time.time() - start_t)
        z_opt, lagrange_multipliers = res["x"].reshape((-1, 1)), res["v"][1]

        if(z_opt[-1] > 1):
            return False, 0
        else:
            M_Q = np.zeros((LP_A_ineq.shape[1], LP_A_ineq.shape[1]))
            M_GT = LP_A_ineq.T
            M_AT = LP_A_eq.T 
            M_DlmG = np.dot(np.diag(lagrange_multipliers), LP_A_ineq) 
            M_DGzh = np.diag((np.dot(LP_A_ineq, z_opt) - LP_b_ineq).flatten()) 
            M_A = LP_A_eq
            #print(M_Q)
            row_1 = np.hstack((M_Q, M_GT, M_AT))

            #print("row 1 \n", row_1, "*"*80)
            row_2 = np.hstack((M_DlmG, M_DGzh, np.zeros((M_DGzh.shape[0], M_AT.shape[1]))))
            #print("row 2 \n", row_2, "*"*80)
            row_3 = np.hstack((M_A, np.zeros((M_A.shape[0], M_DGzh.shape[1] + M_AT.shape[1]))))
            LHS = np.vstack((row_1,
                                row_2,
                                row_3))
            #print(LHS)
            db = np.eye(LP_b_eq.shape[0])

            RHS = np.vstack((np.zeros((LHS.shape[0] - db.shape[0], db.shape[1])), db))
            
            J = np.dot(np.linalg.pinv(LHS), RHS) 
            #print(J)
            dz_opt_d_c_2 = J[: len(z_opt), :]
            if(len(args) != 0):
                #(args)
                dz_opt_d_c_2 = np.dot(dz_opt_d_c_2, args[0])
            #print("dz/dc2", dz_opt_d_c_2)
            con = 1 - z_opt[-1] * z_opt[-1]

            d_con = -2 * z_opt[-1] * dz_opt_d_c_2[-1, :]
            #print("con", con, np.linalg.pinv(d_con.reshape((1, -1))))
            delta_center = np.linalg.pinv(d_con.reshape((1, -1))) * con
            #print(delta_center)
            return True, delta_center


    def intersect_zonotopes(self, obj):
        
        c_1 = self.center()
        G_1 = self.generators()

        c_2 = obj.center()
        G_2 = obj.generators()

        d = c_1.shape[0]
        n = G_2.shape[1]
       
       
        G = np.hstack((G_1, np.zeros((d, n))))
        A = np.hstack((G_1, -1 * G_2))
        b = c_2 - c_1
      
        return c_1, G, A, b

    def make_con_zono_empty_check_LP(self, A, b):
        d = A.shape[1]

        f_cost = np.vstack((np.zeros((d, 1)), np.ones((1, 1))))

        A_ineq = np.vstack((np.hstack((-1 * np.eye(d), -1 * np.ones((d, 1)))), np.hstack((np.eye(d), -1 * np.ones((d, 1))))))

        b_ineq = np.zeros((2 * d, 1))


        A_eq = np.hstack((A, np.zeros((len(A), 1))))
        b_eq = b 

        return f_cost, A_ineq, b_ineq, A_eq, b_eq


    def conv_hull(self):
        pass 

    """def vertices(self):
        V = self.Z[:, 0]

        # generate further potential vertices in the loop
        for iVertex in range(len(self.Z[0: 1:])):
            translation = self.Z[:, iVertex + 1] * np.ones((1, len(V[0, :])))

            # remove inner points

            if(iVertex > len(self.Z[:, 0])):





        for iVertex = 1:length(obj.Z(1,2:end))

            translation = obj.Z(:,iVertex+1)*ones(1,length(V(1,:)));
            V = [V+translation,V-translation];

            % remove inner points
            if iVertex > length(obj.Z(:,1))
                try
                    K = convhulln(V');
                    indices = unique(K);
                    V = V(:,indices);
                catch
                    disp('Convex hull failed')
                    V = V;
                end
            else
                V = V;
    """
