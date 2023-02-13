import errno
import os
import numpy as np

def readVI(filename, dim, zSteps):

    V, I = np.zeros((dim, zSteps)), np.zeros((dim, zSteps))
    try:
        with open(filename) as values:
            for line in values:
                VI = line.split()
                for i in range(dim):
                    V[i] = float(VI[2*i])
                    I[i] = float(VI[2*i+1])
                    # V = np.append(V, float(VI[0]))
                    # I = np.append(I, float(VI[1]))
    except:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)
    
    return V,I
        
