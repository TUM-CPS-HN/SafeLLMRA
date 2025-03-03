from pyzonotope.Zonotope import Zonotope
from pyzonotope.MatZonotope import MatZonotope
import numpy as np 
from pyzonotope.Options import Options
from pyzonotope.Params import Params
from pyzonotope.reachability_nonlinear import *
import time

class NonLinear_reachability:
    def __init__(self):
        self.dim_x = 6
        self.dt = 0.015
        self.params = Params(tFinal = self.dt * 5, dt = self.dt)
        self.options = Options()
        #print("dude", sys.path)

        self.options.params["dim_x"] = self.dim_x

        #Number of trajectories
        self.initpoints = 1000
        #Number of time steps
        self.steps = 10

        # Totoal number of samples
        self.totalsamples = 10000#steps * initpoints

        #noise zonotope
        self.wfac = 0.001

        self.W = Zonotope(np.array(np.zeros((self.options.params["dim_x"], 1))), self.wfac * np.ones((self.options.params["dim_x"], 1)))
        self.params.params["W"] = self.W
        self.GW = []
        for i in range(self.W.generators().shape[1]):
            vec = np.reshape(self.W.Z[:, i + 1], (self.dim_x, 1))
            dummy = []
            dummy.append(np.hstack((vec, np.zeros((self.dim_x, self.totalsamples - 1)))))
            #print(vec.shape, W.Z.shape, dummy[i][:, 2:].shape, dummy[0][:, 0].shape)
            for j in range(1, self.totalsamples, 1):
                right = np.reshape(dummy[i][:, 0:j], (self.dim_x, -1))
                left = dummy[i][:, j:]
                dummy.append(np.hstack((left, right)))
            self.GW.append(np.array(dummy))

        self.GW = np.array(self.GW[0])
        #print("->"*10, np.array(GW).shape)

        self.params.params["Wmatzono"] = MatZonotope(np.zeros((self.dim_x, self.totalsamples)), self.GW)

        self.options.params["zonotopeOrder"] = 100
        self.options.params["tensorOrder"] = 2
        self.options.params["errorOrder"] = 5

        self.u = np.load('/home/mahmoud/catkin_ws/U_azure_500.npy', allow_pickle=True) #read_matlab('D:\\KTH\\matlab reachability\Data-Driven-Reachability-Analysis\\u.mat', 'u')
        #utraj = read_matlab('D:\\KTH\\utraj.mat', 'utraj')
        self.x_meas_vec_0 = np.load('/home/mahmoud/catkin_ws/X0_azure_500.npy', allow_pickle=True)#read_matlab('D:\\KTH\\matlab reachability\Data-Driven-Reachability-Analysis\\x_meas_vec_0.mat', 'x_meas_vec_0')
        self.x_meas_vec_1 = np.load('/home/mahmoud/catkin_ws/X1_azure_500.npy', allow_pickle=True)#read_matlab('D:\\KTH\\matlab reachability\Data-Driven-Reachability-Analysis\\x_meas_vec_1.mat', 'x_meas_vec_1')
        #print(u.shape, x_meas_vec_0.shape, "xasd")
        self.X_0T = x_meas_vec_0
        self.X_1T = x_meas_vec_1


        
        #self.options.params["Zeps"] = Zonotope(np.array(np.zeros((self.dim_x, 1))), self.Z_eps_gen)
        #print(options.params["Zeps"])
        #print(z1)
        #print(f1)

        
        self.options.params["X_0T"] = self.x_meas_vec_0
        self.options.params["X_1T"] = self.x_meas_vec_1 

    def run_reachability(r = [0, 0, 0, 0], u = [0, 0]):
        U = Zonotope(np.array(np.array(u).reshape((2, 1))), np.diag([0.001, .001]))

        R0 = Zonotope(np.array(r).reshape((dim_x, 1)), np.diag([0.001, .001, 0.001, .001, 0.001, .001]))

        self.params.params["U"] = U
        self.params.params["R0"] = R0
        self.options.params["U_full"] = u

        L = 0
        for i in range(self.totalsamples):
            z1 = np.hstack((self.x_meas_vec_0[:, i].flatten(), u.flatten(order='F')[i]))
            f1 = self.x_meas_vec_1[:, i]
            #print("dude", z1)
            for j in range(self.totalsamples):
                z2 = np.hstack((self.x_meas_vec_0[:, j].flatten(), u.flatten(order='F')[j]))
                f2 = self.x_meas_vec_1[:, j]
                #print("TG", np.linalg.norm(z1 - z2))
                new_norm = np.linalg.norm(f1 - f2) / np.linalg.norm(z1 - z2)

                if (new_norm > L):
                    L = new_norm
                    eps = L * np.linalg.norm(z1 - z2)
        
        self.options.params["Zeps"] = Zonotope(np.array(np.zeros((self.dim_x, 1))),eps * np.diag(np.ones((self.options.params["dim_x"], 1)).T[0]))
        #Z_eps_gen = np.load("ZepsZ.npy")#np.save("ZepsZ", eps * np.diag(np.ones((options.params["dim_x"], 1)).T[0]))
        self.options.params["ZepsFlag"] = True


        R_data = reach_DT(self.params, self.options)
        #print("predicted pose", R_data[-1].reduce('girard', 1).Z[:2, 0])

        return R_data

def run_reachability(r = [0, 0, 0, 0], u = [0, 0]):
    dim_x = 6
    dim_u = 2
    dt = 0.015
    params = Params(tFinal = dt * 5, dt = dt)
    options = Options()
    #print("dude", sys.path)
    U = Zonotope(np.array(np.array(u).reshape((dim_u, 1))), np.diag([.001] * dim_u))

    R0 = Zonotope(np.array(r).reshape((dim_x, 1)), np.diag([.001] * dim_x))

    params.params["U"] = U
    params.params["R0"] = R0
    options.params["dim_x"] = dim_x

    #Number of trajectories
    initpoints = 500
    #Number of time steps
    steps = 10

    # Totoal number of samples
    totalsamples = 500#steps * initpoints

    #noise zonotope
    wfac = 0.001

    W = Zonotope(np.array(np.zeros((options.params["dim_x"], 1))), wfac * np.ones((options.params["dim_x"], 1)))
    params.params["W"] = W
    GW = []
    for i in range(W.generators().shape[1]):
        vec = np.reshape(W.Z[:, i + 1], (dim_x, 1))
        dummy = []
        dummy.append(np.hstack((vec, np.zeros((dim_x, totalsamples - 1)))))
        #print(vec.shape, W.Z.shape, dummy[i][:, 2:].shape, dummy[0][:, 0].shape)
        for j in range(1, totalsamples, 1):
            right = np.reshape(dummy[i][:, 0:j], (dim_x, -1))
            left = dummy[i][:, j:]
            dummy.append(np.hstack((left, right)))
        GW.append(np.array(dummy))

    GW = np.array(GW[0])
    #print("->"*10, np.array(GW).shape)

    params.params["Wmatzono"] = MatZonotope(np.zeros((dim_x, totalsamples)), GW)

    options.params["zonotopeOrder"] = 100
    options.params["tensorOrder"] = 2
    options.params["errorOrder"] = 5

    u = np.load('U_azure_500.npy', allow_pickle=True).T #read_matlab('D:\\KTH\\matlab reachability\Data-Driven-Reachability-Analysis\\u.mat', 'u')
    #utraj = read_matlab('D:\\KTH\\utraj.mat', 'utraj')
    x_meas_vec_0 = np.load('X0_azure_500.npy', allow_pickle=True).T#read_matlab('D:\\KTH\\matlab reachability\Data-Driven-Reachability-Analysis\\x_meas_vec_0.mat', 'x_meas_vec_0')
    x_meas_vec_1 = np.load('X1_azure_500.npy', allow_pickle=True).T#read_matlab('D:\\KTH\\matlab reachability\Data-Driven-Reachability-Analysis\\x_meas_vec_1.mat', 'x_meas_vec_1')
    #print(u.shape, x_meas_vec_0.shape, "xasd")
    X_0T = x_meas_vec_0
    X_1T = x_meas_vec_1


    L = 0
    for i in range(totalsamples):
        z1 = np.hstack((x_meas_vec_0[:, i].flatten(), u.flatten(order='F')[i]))
        f1 = x_meas_vec_1[:, i]
        #print("dude", z1)
        for j in range(totalsamples):
            z2 = np.hstack((x_meas_vec_0[:, j].flatten(), u.flatten(order='F')[j]))
            f2 = x_meas_vec_1[:, j]
            #print("TG", np.linalg.norm(z1 - z2))
            new_norm = np.linalg.norm(f1 - f2) / np.linalg.norm(z1 - z2)

            if (new_norm > L):
                L = new_norm
                eps = L * np.linalg.norm(z1 - z2)
    
    options.params["Zeps"] = Zonotope(np.array(np.zeros((dim_x, 1))),eps * np.diag(np.ones((options.params["dim_x"], 1)).T[0]))
    #Z_eps_gen = np.load("ZepsZ.npy")#np.save("ZepsZ", eps * np.diag(np.ones((options.params["dim_x"], 1)).T[0]))
    options.params["ZepsFlag"] = True
    #options.params["Zeps"] = Zonotope(np.array(np.zeros((dim_x, 1))), Z_eps_gen)
    #print(options.params["Zeps"])
    #print(z1)
    #print(f1)
    
    options.params["U_full"] = u
    options.params["X_0T"] = x_meas_vec_0
    options.params["X_1T"] = x_meas_vec_1

    start_t = time.time()
    R_data = reach_DT(params, options)
    print("Operation took {}".format(time.time() - start_t))
    #print("predicted pose", R_data[-1].reduce('girard', 1).Z[:2, 0])

    return R_data

if __name__ == "__main__":
    run_reachability()
