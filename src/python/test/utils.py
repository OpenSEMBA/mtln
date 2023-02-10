import errno
import os
import numpy as np

def readVI(filename):

    V, I = np.array([]), np.array([])
    try:
        with open(filename) as values:
            for line in values:
                VI = line.split()
                V = np.append(V, float(VI[0]))
                I = np.append(I, float(VI[1]))
    except:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)
    
    return V,I
        
