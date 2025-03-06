from scipy.optimize import LinearConstraint
from scipy.optimize import linprog
import numpy as np

def check_zono_intersection(_input):
    zon1, obj = _input
    c_1 = zon1.center()
    G_1 = zon1.generators()

    c_2 = obj.center()
    G_2 = obj.generators()
    c, G, A, b = zon1.intersect_zonotopes(obj)


    LP_f,LP_A_ineq,LP_b_ineq,LP_A_eq,LP_b_eq = zon1.make_con_zono_empty_check_LP(A, b)
 
    def cost(x, *args):
        A = args[0] 
        result = np.dot(x.reshape((1, len(x))), A)[0]
        return result

   
    eq_con = LinearConstraint(LP_A_eq, LP_b_eq.flatten(), LP_b_eq.flatten())

    num_constraints = LP_A_ineq.shape[0]
    lb = np.array([np.NINF] * num_constraints)
    ub = LP_b_ineq.flatten()  # Ensure `ub` matches the number of rows
    ineq_con = LinearConstraint(LP_A_ineq, lb, ub)


    cons = [eq_con, ineq_con]
 
    
    res = linprog(LP_f, A_ub = LP_A_ineq, b_ub=LP_b_ineq.flatten(), \
                  A_eq=LP_A_eq, b_eq=LP_b_eq.flatten(), bounds=(None,None), \
                  options = {'maxiter':50}, method = 'highs-ipm')
    
    z_opt = res["x"]
    lm = res["ineqlin"]["marginals"]
    nu = res["eqlin"]["marginals"]
    if(z_opt[-1] > 1):
        return False, 0
    else:
        M_Q = np.zeros((LP_A_ineq.shape[1], LP_A_ineq.shape[1]))
        M_GT = LP_A_ineq.T
        M_AT = LP_A_eq.T 
        M_DlmG = np.dot(np.diag(lm), LP_A_ineq) 
        M_DGzh = np.diag((np.dot(LP_A_ineq, z_opt) - LP_b_ineq.flatten()))
        M_A = LP_A_eq
        row_1 = np.hstack((M_Q, M_GT, M_AT))

     
        row_2 = np.hstack((M_DlmG, M_DGzh, np.zeros((M_DGzh.shape[0], M_AT.shape[1]))))
        row_3 = np.hstack((M_A, np.zeros((M_A.shape[0], M_DGzh.shape[1] + M_AT.shape[1]))))
        LHS = np.vstack((row_1,
                            row_2,
                            row_3))
        db = np.eye(LP_b_eq.shape[0])

        RHS = np.vstack((np.zeros((LHS.shape[0] - db.shape[0], db.shape[1])), db))

        J = np.dot(np.linalg.pinv(LHS), RHS) 
        dz_opt_d_c_2 = J[: len(z_opt), :]

        con = 1 - z_opt[-1] * z_opt[-1]

        d_con = -2 * z_opt[-1] * dz_opt_d_c_2[-1, :]
        delta_center = np.linalg.pinv(d_con.reshape((1, -1))) * con

        return True, delta_center
    
