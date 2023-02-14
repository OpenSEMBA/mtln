import numpy as np
from os import getcwd

def TL_2conductors(zSteps: int, tSteps: int, l: float, c: float):

    v = np.zeros(zSteps+1)
    i = np.zeros(zSteps)

    dZ = 1e-3
    dT = 0.95*dZ*np.sqrt(l*c)

    vs = 25
    rs = np.sqrt(l/c)
    vl = 0
    rl = np.sqrt(l/c)

    for _ in range(1, tSteps + 1):

        v[0] = (dZ*rs*c/dT+1)**-1*((dZ*rs*c/dT-1)*v[0] - 2*rs*i[0] + (2*vs))
        
        for k in range(1, zSteps):
            v[k] = v[k] - dT/(dZ*c) * (i[k]-i[k-1])
        
        v[zSteps] = (dZ*rl*c/dT+1)**-1*((dZ*rl*c/dT-1)*v[zSteps] + 2*rl*i[zSteps-1] + (2*vl))

        for k in range(zSteps):
            i[k] = i[k] - dT/(dZ*l) * (v[k+1]-v[k])

    return v[:-1], i


def TL_Nconductors(zSteps: int, tSteps: int, l: np.ndarray, c: np.ndarray):
    
    dim = np.shape(l)[0]

    v = np.zeros([dim,zSteps+1])
    i = np.zeros([dim,zSteps])

    maxV = 0

    vs = 25*np.ones(dim)
    rs = np.identity(dim)
    vl = np.zeros(dim)
    rl = np.identity(dim)

    for idx in range(dim):
        lineVel = 1/np.sqrt(l[idx][idx]*c[idx][idx])
        if (lineVel > maxV):
            maxV = lineVel
            
        rs[idx][idx] = np.sqrt(l[idx][idx]/c[idx][idx])
        rl[idx][idx] = np.sqrt(l[idx][idx]/c[idx][idx])

    dZ = 1e-3
    dT = 0.95*dZ/maxV


    cInv = np.linalg.inv(c)
    lInv = np.linalg.inv(l)

    rSc = np.matmul(rs,c)
    rLc = np.matmul(rl,c)

    sourceEq = dZ*rSc/dT-np.identity(dim)
    loadEq   = dZ*rLc/dT-np.identity(dim)
    sourceEqInv = np.linalg.inv(dZ*rSc/dT+np.identity(dim))
    loadEqInv = np.linalg.inv(dZ*rLc/dT+np.identity(dim))

    for _ in range(1, tSteps + 1):

        v[:,0] = np.matmul(sourceEqInv,np.matmul(sourceEq,v[:,0]) - 2*np.matmul(rs,i[:,0]) + (2*vs))
        
        for k in range(1, zSteps):
            
            v[:,k] = v[:,k] - (dT/dZ)*np.matmul(cInv,i[:,k]-i[:,k-1])
        
        v[:,zSteps] = np.matmul(loadEqInv,np.matmul(loadEq,v[:,zSteps]) + 2*np.matmul(rl,i[:,zSteps-1]) + (2*vl))

        for k in range(zSteps):
            i[:,k] = i[:,k] - (dT/dZ)*np.matmul(lInv,v[:,k+1]-v[:,k])

    return v[:,:-1], i
    

    
