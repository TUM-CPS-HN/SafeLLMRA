class Options():
    """
    Options Class. Acting as a structure to hold variables. 
    Used in Non-Linear Reachability
    """
    def __init__(self, dim_x = None, W = None, Wmatzono = None, zonotopeOrder = None, tensorOrder = None, errorOrder = None, Zeps = None, 
                 ZepsFlag = None, U_full = None, X_0T = None, X_1T = None):

        self.params = {"dim_x" : dim_x,
                       "W"     : W ,
                       "Wmatzono" : Wmatzono,
                       "zonotopeOrder" : zonotopeOrder,
                       "tensorOrder" : tensorOrder,
                       "errorOrder" : errorOrder,
                       "Zeps" : Zeps,
                       "ZepsFlag" : ZepsFlag,
                       "U_full" : U_full,
                       "X_0T" : X_0T,
                       "X_1T" : X_1T}

    