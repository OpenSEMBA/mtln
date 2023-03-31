import numpy as np
import matplotlib.pyplot as plt

import scipy as sci
from scipy.constants import epsilon_0, mu_0, speed_of_light

import src.mtln as mtln

def gaussian(x, x0, s0):
    return np.exp( - (x-x0)**2 / (2*s0**2) )

def square_pulse(x, A, x0):
    return A*(x <= x0)*(x >= 0)



def test_get_phase_velocities():
    v = mtln.MTL(l=0.25e-6, c=100e-12).get_phase_velocities()
    assert v.shape == (1,)
    assert np.isclose(2e8, v[0])

def test_coaxial_line_initial_voltage():
    line = mtln.MTL(l=0.25e-6, c= 100e-12, length=400)
    line.set_voltage(lambda x: gaussian(x, 200, 50))
    voltage_probe = line.add_voltage_probe(200)

    finalTime = 5e-6
    for t in np.arange(0, np.floor(finalTime / line.get_timestep())):
        line.step()

    plt.plot(voltage_probe.t, voltage_probe.v)    
    plt.show()
    # assert

def test_coaxial_line_paul_8_6():
    """ 
    Described in Ch. 8 of Paul Clayton, 
    Analysis of Multiconductor Transmission Lines. 2004. 
    """
    
    line = mtln.MTL(l=0.25e-6, c= 100e-12, length=400.0, Zs = 150)
    finalTime = 18e-6
    tRange = np.arange(0, np.floor(finalTime / line.get_timestep()))

    magnitude = lambda t: square_pulse(t, 100, 6e-6)
    line.add_voltage_source(magnitude, position=0.0)
    voltage_probe = line.add_voltage_probe(position=0.0)
    
    for t in tRange:
        line.step()

    plt.plot(voltage_probe.t, voltage_probe.v)
    plt.sho()



