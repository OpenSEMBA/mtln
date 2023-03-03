import numpy as np
import json

def getIUpdateMatrices(m: np.ndarray, n: np.ndarray, dZ: float, dT: float, vf: str):

    with open(vf, 'r') as f:
        data = json.load(f) 

    dim = len(data)
    A = np.zeros([dim,dim])
    B = np.zeros([dim,dim])
    
    for (k,v) in zip(data.keys(), data.values()):
        A[k][k] = v["constant"]
        B[k][k] = v["proportional"]

    #next, residues, poles
    r1 = np.linalg.inv(dZ*m/dT + dZ*n/2)
    aux = dZ*m/dT - dZ*n/2
    r2 = np.matmul(r1,aux)

    Q1 = 0
    Q2 = 0
    Q3 = 0

    
    return r1, r2

def getUpdateMatrices(m: np.ndarray, n: np.ndarray, dZ: float, dT: float):

    r1 = np.linalg.inv(dZ*m/dT + dZ*n/2)
    aux = dZ*m/dT - dZ*n/2
    r2 = np.matmul(r1,aux)

    Q1 = 0
    Q2 = 0
    Q3 = 0

    
    return r1, r2


def getTerminalMatrices(m: np.ndarray, n: np.ndarray, r: np.ndarray, dZ: float, dT: float):
    
    dim = np.shape(m)[0]
    
    r1 = np.linalg.inv((dZ/dT)*np.matmul(r, m) + (dZ/2)*np.matmul(r, n) + np.identity(dim))
    r2 = (dZ/dT)*np.matmul(r, m) - (dZ/2)*np.matmul(r, n) - np.identity(dim)
    
    return r1, r2