from os import getcwd
import numpy as np
import matplotlib.pyplot as plt

import subprocess

from src.fd1d_TL import TL_2conductors
from src.fd1d_TL import TL_Nconductors
from .utils import readVI

from scipy.constants import epsilon_0, mu_0

def test_coaxial_dc():
    
    eps = epsilon_0
    mu = mu_0
    #pul L and C for a coaxial given shield and wire radius
    shieldR = 5e-3
    wireR = 1e-3
    l = (mu/(2*np.pi))*np.log(shieldR/wireR)
    c = 2*np.pi*eps/np.log(shieldR/wireR)
    
    zSteps = 200
    tSteps = 1000
    
    py_voltage, py_current = TL_2conductors(zSteps, tSteps, l, c)

    filename = "2conductors_coaxial_output.txt"
    outfile = open(getcwd()+"\\python\\logs\\"+filename, 'w')
    
    for k in range(zSteps):
        outfile.write(str(py_voltage[k])+" "+str(py_current[k])+"\n")
    outfile.close()

    subprocess.run(getcwd()+"\\fortran\\out\\build\\x64-Debug\\coaxial.exe "+str(zSteps)+" "+str(tSteps), cwd=getcwd()+"\\fortran\\out\\build\\x64-Debug\\")
    for_voltage, for_current = readVI(getcwd()+"\\fortran\\logs\\f_output.txt", 1, zSteps)

    #plot before asserting
    plt.figure(figsize=(8, 3.5))

    plt.subplot(211)
    plt.plot(py_voltage, 'k', linewidth=1, label = 'python')
    plt.plot(for_voltage[0], 'r--', linewidth=1, label = 'fortran')
    plt.ylabel('$V$', fontsize='14')
    plt.ylim(0, 35)
    plt.legend()

    plt.subplot(212)
    plt.plot(py_current, 'k', linewidth=1, label = 'python')
    plt.plot(for_current[0], 'r--', linewidth=1, label = 'fortran')
    plt.ylabel('$I$', fontsize='14')
    plt.xlabel('FDTD cells')
    plt.ylim(0, 0.2)

    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    plt.show()
        
    assert np.allclose(
        py_voltage,
        for_voltage,
        rtol=0.1
    )    
    
    assert np.allclose(
        py_current,
        for_current,
        rtol=0.1
    )    


    

def test_coaxial_DC_2conductors():
    #     ________ 
    #    /        \
    #   /          \
    #  |  o --- o   |
    #   \          /
    #    \________/

    eps = epsilon_0
    mu = mu_0
    #pul L and C for a coaxial given shield and wire radius
    shieldR = 5e-3
    wireR = 1e-3
    dToCenter = 2e-3
    
    l = np.zeros([2,2])
    c = np.zeros([2,2])
    
    l[0][0] = (mu/(2*np.pi))*np.log((shieldR**2-dToCenter**2)/(shieldR*wireR))
    l[1][1] = l[0][0]
    l[0][1] = (mu/(2*np.pi))*np.log((dToCenter/shieldR)*np.sqrt((dToCenter**4+shieldR**4+2*dToCenter**2*shieldR**2)/(dToCenter**4+dToCenter**4+2*dToCenter**4)))
    l[1][0] = l[0][1]
    
    c = mu*eps*np.linalg.inv(l)
    
    zSteps = 200
    tSteps = 1000
    
    py_voltage, py_current = TL_Nconductors(zSteps, tSteps, l, c)
    
    filename = "3Conductors_output.txt"
    outfile = open(getcwd()+"\\python\\logs\\"+filename, 'w')

    for k in range(zSteps):
        for n in range(np.shape(l)[0]):
            outfile.write(str(py_voltage[n][k])+" "+str(py_current[n][k])+" ")
        outfile.write("\n")
    outfile.close()

    subprocess.run(getcwd()+"\\fortran\\out\\build\\x64-Debug\\Nconductors_3.exe "+str(zSteps)+" "+str(tSteps), cwd=getcwd()+"\\fortran\\out\\build\\x64-Debug\\")
    for_voltage, for_current = readVI(getcwd()+"\\fortran\\logs\\3conductors_output.txt", 2, zSteps)

    assert np.isnan(py_voltage).any() == False
    assert np.isnan(py_current).any() == False
    
    #plot before asserting
    plt.figure(figsize=(8, 3.5))

    plt.subplot(211)
    plt.plot(py_voltage[0], 'k', linewidth=1, label = 'python #1')
    plt.plot(py_voltage[1], 'k--', linewidth=1, label = 'python #2')
    plt.plot(for_voltage[0], 'r--', linewidth=1, label = 'fortran #1')
    plt.plot(for_voltage[1], 'r-.', linewidth=1, label = 'fortran #2')
    plt.ylabel('$V$', fontsize='14')
    plt.ylim(0, 35)
    plt.legend()

    plt.subplot(212)
    plt.plot(py_current[0], 'k', linewidth=1, label = 'python #1')
    plt.plot(py_current[1], 'k--', linewidth=1, label = 'python #2')
    plt.plot(for_current[0], 'r', linewidth=1, label = 'fortran #1')
    plt.plot(for_current[1], 'r--', linewidth=1, label = 'fortran #2')
    plt.ylabel('$I$', fontsize='14')
    plt.xlabel('FDTD cells')
    plt.ylim(0, 0.2)

    plt.subplots_adjust(bottom=0.2, hspace=0.45)
    plt.show()
        
    for i in range(2):
        assert np.allclose(
            py_voltage[i],
            for_voltage[i],
            rtol=0.1
        )    
        
        assert np.allclose(
            py_current[i],
            for_current[i],
            rtol=0.1
        )    
