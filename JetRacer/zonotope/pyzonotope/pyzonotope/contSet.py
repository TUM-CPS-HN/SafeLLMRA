
class contSet():
    def __init__(self, *args):
        self._id = 0
        self._dimension = 0

        if(len(args) == 1):
            self._dimension = args[0]
            
        elif(len(args) == 2):
            self._id = args[0]
            self._dimension = args[1]

    def copy(self, other_set):
        self._id = other_set._id
        self._dimension = other_set._dimension

    def display(self):
        print("id {} \n dimension {}".format(self._id, self._dimension))