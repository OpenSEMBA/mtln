import numpy as np

class Probe:
    def __init__(self):
        return

class MTL:
    """
    Lossless Multiconductor Transmission Line
    """

def __init__(self, L, C, length=1.0, nx=100, Zs=0.0, Zl=0.0):
    self.x = np.linspace(0, length, nx+1)
    self.l = L
    self.c = C
    
    self.number_of_conductors = np.shape(self.l)[0]
    self.v = np.zeros([self.number_of_conductors, self.x.shape[0]  ])
    self.i = np.zeros([self.number_of_conductors, self.x.shape[0]-1])

    self.zs = Zs * np.eye(self.number_of_conductors)
    self.zl = Zl * np.eye(self.number_of_conductors)

def get_phase_velocities(self):
    return 1/np.sqrt(np.diag(self.l)*np.diag(self.c))

def get_max_timestep(self):
    dx = np.min(self.x[1:] - self.x[:-1])
    return dx / np.max(self.get_phase_velocities())

def step():

    
    

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

def add_voltage_source(self, position: float, conductor: int, magnitude):
    return

def add_voltage_probe(self, position: float):
    return

