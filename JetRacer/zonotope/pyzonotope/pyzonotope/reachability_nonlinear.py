from pyzonotope.Zonotope import Zonotope
import numpy as np 
from pyzonotope.Options import Options
from pyzonotope.Params import Params
from numpy.linalg import pinv
from pyzonotope.Interval import Interval
import numpy.matlib as matlib
import time 

def params2options(params, options):
    for key, value in params.params.items():
        options.params[key] = value
    return options 


def set_inputSet(options, checkname):
    """
    This function is greatly modified and different from the one in CORA. 
    Mostly removed parts that are irrelevant to our reachability analysis techniques, that is the check for "checkOptionsSimulate"
    """
    uTrans = options.params["U"].center()
    #print(options.params["U"])
    options.params["U"] = options.params["U"]  - uTrans
    #print(options.params["U"])
    options.params["uTrans"] = uTrans

    return options


def checkOptionsReach(options, hyb):
    checkName = 'checkOptionsReach'
    options.params["tStart"] = 0
    options.params["reductionTechnique"] = 'girard'
    options.params["verbose"] = 0
    options = set_inputSet(options, checkName)
    return options


def linReach_DT(R_data ,options):
    """
    This function calculates teh next state for the reachability analysis of a non linear system.
    """
    #print("*" * 80)
    
   # print(options.params["U"] , "\n", options.params["uTrans"])
    options.params["Uorig"] = options.params["U"] + options.params["uTrans"]
    xStar = R_data.center()
    uStar = options.params["Uorig"].center()

    xStarMat = matlib.repmat(xStar, 1, options.params["X_0T"].shape[1])
    uStarMat  = matlib.repmat(uStar, 1, options.params["U_full"].shape[1])
    oneMat    = matlib.repmat(np.array([1]), 1, options.params["U_full"].shape[1])
    start_t = time.time()
    #options.params["X_0T"] + (-1 * xStarMat), options.params["U_full"] + -1 * uStarMat
    IAB = np.dot(options.params["X_1T"], pinv(np.vstack([oneMat, options.params["X_0T"] + (-1 * xStarMat), options.params["U_full"] + -1 * uStarMat])))
    start_t2 = time.time()
    #dummy = options.params["Wmatzono"] + np.dot(IAB, np.vstack([oneMat, options.params["X_0T"]+(-1*xStarMat), options.params["U_full"]+ -1 * uStarMat]))
    #print("GEEZ \n", np.dot(IAB, np.vstack([oneMat, options.params["X_0T"]+(-1*xStarMat), options.params["U_full"]+ -1 * uStarMat])).shape)
    V =  -1 * (options.params["Wmatzono"] + np.dot(IAB, np.vstack([oneMat, options.params["X_0T"]+(-1*xStarMat), options.params["U_full"] + \
    -1 * uStarMat]))) + options.params["X_1T"]
    #print("IAB is {}".format(IAB))
    #print("V is: \n", V)
    start_t3 = time.time()
    VInt = V.interval_matrix()
    leftLimit = VInt.Inf
    rightLimit = VInt.Sup
    
    #print(leftLimit.shape)
    V_one = Zonotope(Interval(leftLimit.min(axis=1).T, rightLimit.max(axis=1).T))

    #print("Vone is {}".format(V_one))
    x = R_data+(-1*xStar)
    #print("Move center:", np.dot(IAB[:, -2:], uStar), IAB[:, 3:])
    result = (x.cart_prod(options.params["Uorig"] + (-1 * uStar)).cart_prod([1]) * IAB) +  V_one + options.params["W"] + options.params["Zeps"]
    #print("wow {}".format(x.cart_prod(options.params["Uorig"] + (-1 * uStar)).cart_prod([1])))
    #print("derivative is {}".format(IAB[:2, -2:]))
    #print("done {} {} {}".format(time.time() - start_t, time.time() - start_t2, time.time() - start_t3))
    return result#, IAB[:2, -2:]

def reach_DT(params, options, *varargin):
    options = params2options(params,options)
    options = checkOptionsReach(options,0)

    spec = []
    if(len(varargin) > 0):
        spec = varargin[0]

    #t = np.linspace(options.params["tStart"], options.params["tFinal"], 
    #                num = (options.params["tFinal"] - options.params["tStart"]) // options.params["dt"]).tolist()


    #print("T", t)


    steps = 5#len(t) - 1

    R_data = [params.params["R0"]]

    for i in range(steps):
        if('uTransVec' in options.params):
            options.params['uTrans'] = options.params["uTransVec"][:, i]

        R_data[i] = R_data[i].reduce('girard', 100)
        start_t = time.time()
        new_state = linReach_DT(R_data[i] ,options)
        #print("Took {}".format(time.time() - start_t))
        R_data.append(new_state.reduce('girard', 1))
        #print("early info", R_data)
        #R_data.append(dc_du)
    return R_data
