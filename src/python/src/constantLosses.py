import numpy as np

def getUpdateMatrices(m: np.ndarray, n: np.ndarray, dZ: float, dT: float):

    r1 = np.linalg.inv(dZ*m/dT + dZ*n/2)
    aux = dZ*m/dT - dZ*n/2
    r2 = np.matmul(r1,aux)
    
    return r1, r2


def getTerminalMatrices(m: np.ndarray, n: np.ndarray, r: np.ndarray, dZ: float, dT: float):
    
    dim = np.shape(m)[0]
    
    r1 = np.linalg.inv((dZ/dT)*np.matmul(r, m) + (dZ/2)*np.matmul(r, n) + np.identity(dim))
    r2 = (dZ/dT)*np.matmul(r, m) - (dZ/2)*np.matmul(r, n) - np.identity(dim)
    
    return r1, r2