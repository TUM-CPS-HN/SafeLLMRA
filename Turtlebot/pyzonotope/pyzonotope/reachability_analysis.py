from pyzonotope.MatZonotope import MatZonotope
import numpy as np 
from numpy.linalg import pinv# as pinv


def concat_traj_linear(X0, x, utraj, dim_x, steps, initpoints):
    x_meas_vec_0 = np.reshape(x[:, 0], (dim_x, initpoints), order='F')
    x_meas_vec_1 = np.reshape(x[:, 1], (dim_x, initpoints), order='F')
    u_mean_vec_0 = np.reshape(utraj[utraj != 0], (-1, initpoints), order='F')
    U_full = u_mean_vec_0
    X_0t = x_meas_vec_0
    X_1t = x_meas_vec_1

    return U_full, X_0t, X_1t

def concat_traj(x, dim_x, steps, initpoints):
    x_meas_vec_0 = np.reshape(x[:, 0], (dim_x, -1), order='F')
    x_meas_vec_1 = np.reshape(x[:, 1], (dim_x, -1), order='F')
    #u_mean_vec_0 = np.reshape(utraj[utraj != 0], (-1, initpoints), order='F')
    #U_full = u_mean_vec_0
    X_0t = x_meas_vec_0
    X_1t = x_meas_vec_1

    return  X_0t, X_1t

def get_AB(U_full, X_0t, X_1t, Wmatzono):
    #print(Wmatzono.generators)
    X1W_cen =  X_1t - Wmatzono.center
    #print("X1W_cen", X1W_cen)
    #print()
    X1W = MatZonotope(X1W_cen,Wmatzono.generators)
    #print("X1W", X1W)
    AB = X1W * pinv(np.vstack((X_0t, U_full)))
    return AB