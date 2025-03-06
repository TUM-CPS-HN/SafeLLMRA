import numpy as np 

class Interval():
    def __init__(self, *args):
        if(len(args) == 1):
            if(type(args[0]) == Interval):
                self = Interval
            else:
                self.inf = args[0]
                self.sup = args[0]

        elif(len(args) == 2):
            if(args[0].shape != args[1].shape):
                raise Exception("Limits are of different dimensions!")
            
            if(np.all(args[0] <= args[1])):
                self.inf = args[0]
                self.sup = args[1]
            else:
                raise Exception("Lower limit is larger than the upper limit!")

        #print("inf {} \n sup {}".format(self.inf, self.sup))


    def __str__(self):
        S = "inf {} \n sup {}".format(self.inf, self.sup)
        return S