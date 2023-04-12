import numpy as np
import matplotlib.pyplot as plt



import scipy as sci
from scipy.constants import epsilon_0, mu_0, speed_of_light

import src.mtln as mtln

def gaussian(x, x0, s0):
    return np.exp( - (x-x0)**2 / (2*s0**2) )

def square_pulse(x, A, x0):
    return A*(x <= x0)*(x >= 0)

def triangle_pulse(x, A, x0):
    return A*(x/x0)*(x <= x0)*(x >= 0)



def test_get_phase_velocities():
    v = mtln.MTL(l=0.25e-6, c=100e-12).get_phase_velocities()
    assert v.shape == (1,)
    assert np.isclose(2e8, v[0])

def test_coaxial_line_initial_voltage():
    line = mtln.MTL(l=0.25e-6, c= 100e-12, length=400)
    line.set_voltage(lambda x: gaussian(x, 200, 50))
    voltage_probe = line.add_voltage_probe(200)

    finalTime = 5e-6
    for t in np.arange(0, np.floor(finalTime / line.get_max_timestep())):
        line.step()

    plt.plot(voltage_probe.t, voltage_probe.v)    
    plt.show()
    # assert

def test_coaxial_line_paul_8_6_square():
    """ 
    Described in Ch. 8 of Paul Clayton, 
    Analysis of Multiconductor Transmission Lines. 2004. 
    """
    
    line = mtln.MTL(l=0.25e-6, c= 100e-12, length=400.0, Zs = 150)
    finalTime = 18e-6
    tRange = np.arange(0, np.floor(finalTime / line.get_max_timestep()))

    magnitude = lambda t: square_pulse(t, 100, 6e-6)
    line.add_voltage_source(position=0.0, conductor=0, magnitude=magnitude)
    voltage_probe = line.add_voltage_probe(position=0.0)
    current_probe = line.add_current_probe(position=400.0)
    
    for t in tRange:
        line.step()

    # xticks = range(int(np.floor(min(1e6*current_probe.t))), int(np.ceil(max(1e6*current_probe.t))+1))

    # plt.plot(1e6*voltage_probe.t, voltage_probe.v)
    # plt.ylabel(r'$V (0, t)\,[V]$')
    # plt.xlabel(r'$t\,[\mu s]$')
    # plt.grid('both')
    # plt.show()
    
    # plt.plot(1e6*current_probe.t, current_probe.i)
    # plt.ylabel(r'$I (L, t)\,[A]$')
    # plt.xlabel(r'$t\,[\mu s]$')
    # plt.xticks(xticks)
    # plt.grid('both')
    # plt.show()
    
    start_times = [0.1, 4.1, 6.1, 8.1, 10.1, 12.1, 14.1, 16.1]
    end_times =   [3.9, 5.9, 7.9, 9.9, 11.9, 13.9, 15.9, 18.9]
    check_voltages = [25, -12.5, -37.5, -18.75, 18.75, 9.375, -9.375, -4.6875]
    for (t_start, t_end, v) in zip(start_times, end_times, check_voltages):
        start = np.argmin(np.abs(voltage_probe.t - t_start*1e-6))
        end   = np.argmin(np.abs(voltage_probe.t - t_end*1e-6))
        assert np.all(np.isclose(voltage_probe.v[start:end], v))

def test_coaxial_line_paul_8_6_triangle():
    """ 
    Described in Ch. 8 of Paul Clayton, 
    Analysis of Multiconductor Transmission Lines. 2004. 
    """
    
    line = mtln.MTL(l=0.25e-6, c= 100e-12, length=400.0, Zs = 150)
    finalTime = 18e-6
    tRange = np.arange(0, np.floor(finalTime / line.get_max_timestep()))

    magnitude = lambda t: triangle_pulse(t, 100, 6e-6)
    line.add_voltage_source(position=0.0, conductor=0, magnitude=magnitude)
    voltage_probe = line.add_voltage_probe(position=0.0)
    
    for t in tRange:
        line.step()

    # xticks = range(int(np.floor(min(1e6*voltage_probe.t))), int(np.ceil(max(1e6*voltage_probe.t))+1))

    # plt.plot(1e6*voltage_probe.t, voltage_probe.v)
    # plt.ylabel(r'$V (0, t)\,[V]$')
    # plt.xlabel(r'$t\,[\mu s]$')
    # plt.grid('both')
    # plt.show()
    
    times = [4.0, 5.9, 6.1, 8.0, 10.1, 12]
    voltages =   [16.67, 12.5, -12.5, -25, 6.25, 12.5]
    for (t, v) in zip(times,voltages):
        index = np.argmin(np.abs(voltage_probe.t - t*1e-6))
        assert np.all(np.isclose(voltage_probe.v[index], v, atol=0.5))



def test_ribbon_cable_paul_9_3():
    """
    Described in Ch. 9 of Paul Clayton
    Analysis of Multiconductor Transmission Lines. 2007. 
    """
    l = np.array(shape=(2,2))
    l[0] = [0.7485*1e-6 ,0.5077*1e-6]
    l[1] = [0.5077*1e-6, 1.0154*1e-6]
    c = np.array(shape=(2,2))
    c[0] = [37.432*1e-12, -18.716*1e-12]
    c[1] = [-18.716*1e-12, 24.982*1e-12]

    line = mtln.MTL(l=l, c= c, length=2.0, Zs = 150)
    finalTime = 18e-6
    tRange = np.arange(0, np.floor(finalTime / line.get_max_timestep()))

    magnitude = lambda t: triangle_pulse(t, 100, 6e-6)
    line.add_voltage_source(position=0.0, conductor=0, magnitude=magnitude)
    voltage_probe = line.add_voltage_probe(position=0.0)
    
    for t in tRange:
        line.step()
