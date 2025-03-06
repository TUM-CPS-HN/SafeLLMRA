from scipy.io import loadmat

def read_matlab(filename, var):
    mat = loadmat(filename)
    return mat[var]