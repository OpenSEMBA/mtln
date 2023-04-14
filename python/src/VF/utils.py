import json
import numpy as np
from numpy.random import default_rng



def Zw(f, f0, R, sigma):
    return (R*(1+2.5j)*np.exp(-((f-f0*np.ones(len(f)))**2)/(2*sigma**2)))

def Y_Zw(f, f0, R, sigma):
    return 1/(R*(1+2.5j)*np.exp(-((f-f0*np.ones(len(f)))**2)/(2*sigma**2)))
    # return 1/Zw(f, f0, R, sigma)

def RLseriesDisp(f, R, L, n):
    w = 2*np.pi*f
    return R + 1j*w**(n)*L

def Y_RLseriesDisp(f, R, L, n):
    w = 2*np.pi*f
    return 1/(R + 1j*w**(n)*L)
   

def RLpar(f, R, L):
    w = 2*np.pi*f
    return 1j*w*R/(1j*w+R/L)

def RCpar(f, R, C):
    w = 2*np.pi*f
    # return R/(1j*w*C*R+1)
    return (1/C)/(1j*w+1/(C*R))

def epsDebye(f,k,tau):
    out = 1
    for i in range(np.size(k)):
        out += k[i]/((1+2*np.pi*1j*f*tau[i]))

    return out

def epsDebyeErr(f,k,tau,p):
    out = 1
    rng = default_rng()
    ran = 1+(p*2/100)*rng.random(np.size(f))-p/100
    for i in range(np.size(k)):
        out += ran*k[i]/((1+2*np.pi*1j*f*tau[i]))

    return out

def vf(f,p,r):
    out = 0
    for i in range(np.size(p)):
        out += r[i]/((2*np.pi*1j*f)-p[i])

    return out

def vfi(f,p,r,i):
    return r[i]/((2*np.pi*1j*f)-p[i])

def z(f,f0, rdc):
    f0 = np.ones(np.size(f))*f0
    return rdc*(1+1j*f/f0)*(f<=f0)+rdc*(1+1j)*np.sqrt(f/f0)*(f>=f0)

def serializeJSON(nReal: int, nCmplx: int, constant: bool, proportional: bool,
                    a: np.ndarray, b: np.ndarray, p: np.ndarray, r:np.ndarray):
    
    # fitDict =  {
    #         "constant" : 0,
    #         "proportional" : 0,
    #         "poles" : [],
    #         "residuesR" : [],
    #         "residuesI" : []
    #     }
    
    zDict = {}
    
    nPoles = nReal + nCmplx*2
    dim = int(np.sqrt(np.shape(r)[0]))
    
    with open( "info.json" , "w" ) as x:
  
        for i in range(dim):
            
            zDict[i] = {
                "constant" : 0,
                "proportional" : 0,
                "polesR" : [], 
                "polesI" : [],
                "residuesR" : [], 
                "residuesI" : []
                }

            if (constant):
                zDict[i]["constant"] = a[i*nPoles]
            if (proportional):
                zDict[i]["proportional"] = b[i*nPoles]
            
            poles = []
            res = []
            for j in range(nPoles):
                poles.append(p[j])
                res.append(r[i*nPoles][j])
                
            zDict[i]["polesR"] = [np.real(v) for v in poles]
            zDict[i]["polesI"] = [np.imag(v) for v in poles]
            zDict[i]["residuesR"] = [np.real(v) for v in res]
            zDict[i]["residuesI"] = [np.imag(v) for v in res]
                        
        json.dump( zDict, x )
