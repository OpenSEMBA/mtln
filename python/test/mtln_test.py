import numpy as np
import matplotlib.pyplot as plt

import scipy as sci
from scipy.constants import epsilon_0, mu_0, speed_of_light

import src.mtln as mtln

import src.waveforms as wf

from skrf.media import DistributedCircuit    
    

def test_trapezoidal_pulse():
    A = 1
    rise_time = 10e-9
    fall_time = 10e-9
    f0 = 20e6
    D = 0.5
    magnitude = lambda t: wf.trapezoidal_wave(t, A, rise_time, fall_time, f0, D)
    plateu_duration = D/f0 - 0.5 * (rise_time + fall_time)

    # time = np.linspace(0,200e-9,1000)
    # plt.plot(time, magnitude(time))
    # plt.show()
    assert (magnitude(0.5*rise_time) == 0.5*A)
    assert (magnitude(1.1*rise_time) == A)
    assert (magnitude(rise_time + plateu_duration + 0.5*fall_time) == 0.5*A)
    
def test_get_phase_velocities():
    v = mtln.MTL(l=0.25e-6, c=100e-12).get_phase_velocities()
    assert v.shape == (1,)
    assert np.isclose(2e8, v[0])

def test_coaxial_line_initial_voltage():
    line = mtln.MTL(l=0.25e-6, c= 100e-12, length=400)
    line.set_voltage(0, lambda x: wf.gaussian(x, 200, 50))
    v_probe = line.add_probe(position=200, conductor=0, type= 'voltage')

    finalTime = 10e-6
    for t in line.get_time_range(finalTime):
        line.step()

    plt.plot(v_probe.t, v_probe.val)    
    plt.show()
    # assert

def test_coaxial_line_paul_8_6_square():
    """ 
    Described in Ch. 8 of Paul Clayton, 
    Analysis of Multiconductor Transmission Lines. 2004. 
    """
    
    line = mtln.MTL(l=0.25e-6, c= 100e-12, length=400.0, Zs = 150)
    finalTime = 18e-6

    magnitude = lambda t: wf.square_pulse(t, 100, 6e-6)
    line.add_voltage_source(position=0.0, conductor=0, magnitude=magnitude)
    v_probe = line.add_probe(position=0.0, conductor=0, type='voltage')
    i_probe = line.add_probe(position=400.0,conductor=0, type='current')
    
    for t in line.get_time_range(finalTime):
        line.step()

    xticks = range(int(np.floor(min(1e6*i_probe.t))), int(np.ceil(max(1e6*i_probe.t))+1))

    start_times = [0.1, 4.1, 6.1, 8.1, 10.1, 12.1, 14.1, 16.1]
    end_times =   [3.9, 5.9, 7.9, 9.9, 11.9, 13.9, 15.9, 18.9]
    check_voltages = [25, -12.5, -37.5, -18.75, 18.75, 9.375, -9.375, -4.6875]
    for (t_start, t_end, v) in zip(start_times, end_times, check_voltages):
        start = np.argmin(np.abs(v_probe.t - t_start*1e-6))
        end   = np.argmin(np.abs(v_probe.t - t_end*1e-6))
        assert np.all(np.isclose(v_probe.val[start:end], v))

    # plt.figure()
    # plt.plot(1e6*v_probe.t, v_probe.val)
    # plt.ylabel(r'$V (0, t)\,[V]$')
    # plt.xlabel(r'$t\,[\mu s]$')
    # plt.grid('both')
    
    # plt.figure()
    # plt.plot(1e6*i_probe.t, i_probe.val )
    # plt.ylabel(r'$I (L, t)\,[A]$')
    # plt.xlabel(r'$t\,[\mu s]$')
    # plt.xticks(xticks)
    # plt.grid('both')
    
    # plt.show()
    
def test_coaxial_line_paul_8_6_triangle():
    """ 
    Described in Ch. 8 of Paul Clayton, 
    Analysis of Multiconductor Transmission Lines. 2004. 
    """
    
    line = mtln.MTL(l=0.25e-6, c= 100e-12, length=400.0, Zs = 150.0)
    finalTime = 18e-6

    magnitude = lambda t: wf.triangle_pulse(t, 100, 6e-6)
    line.add_voltage_source(position=0.0, conductor=0, magnitude=magnitude)
    v_probe = line.add_probe(position=0.0, conductor=0, type='voltage')
    
    for t in line.get_time_range(finalTime):
        line.step()
   
    times = [4.0, 5.9, 6.1, 8.0, 10.1, 12]
    voltages =   [16.67, 12.5, -12.5, -25, 6.25, 12.5]
    for (t, v) in zip(times,voltages):
        index = np.argmin(np.abs(v_probe.t - t*1e-6))
        assert np.all(np.isclose(v_probe.val[index], v, atol=0.5))

    # xticks = range(int(np.floor(min(1e6*v_probe.t))), int(np.ceil(max(1e6*v_probe.t))+1))

    # plt.plot(1e6*v_probe.t, v_probe.val)
    # plt.ylabel(r'$V (0, t)\,[V]$')
    # plt.xlabel(r'$t\,[\mu s]$')
    # plt.grid('both')
    # plt.show()

def test_ribbon_cable_20ns_paul_9_3():
    """
    Described in Ch. 9.3.1 "Ribbon Cables" of Paul Clayton
    Analysis of Multiconductor Transmission Lines. 2007. 
    """
    l = np.zeros([2,2])
    l[0] = [0.7485*1e-6 ,0.5077*1e-6]
    l[1] = [0.5077*1e-6, 1.0154*1e-6]
    c = np.zeros([2,2])
    c[0] = [37.432*1e-12, -18.716*1e-12]
    c[1] = [-18.716*1e-12, 24.982*1e-12]

    Zs, Zl = np.zeros([1,2]), np.zeros([1,2])
    Zs[:] = [50,50]
    Zl[:] = [50,50]

    line = mtln.MTL(l=l, c= c, length=2.0, nx = 2, Zs = Zs, Zl = Zl)
    finalTime = 200e-9

    magnitude = lambda t: wf.trapezoidal_wave(t, A = 1, rise_time=20e-9, fall_time=20e-9, f0=1e6, D=0.5)
    line.add_voltage_source(position=0.0, conductor=1, magnitude=magnitude)
    v_probe = line.add_probe(position=0.0, conductor= 0, type='voltage')
    
    for t in line.get_time_range(finalTime):
        line.step()

    # From Paul's book: 
    # "The crosstalk waveform rises to a peak of around 110 mV [...]"
    assert(np.isclose(np.max(v_probe.val), 113e-3, atol=1e-3))

    # plt.plot(1e9*v_probe.t, 1e3*v_probe.val)
    # plt.ylabel(r'$V_1 (0, t)\,[mV]$')
    # plt.xlabel(r'$t\,[\mu s]$')
    # plt.xticks(range(0, 200 ,50))
    # plt.grid('both')
    # plt.show()


def test_ribbon_cable_1ns_paul_9_3():
    """
    Described in Ch. 9.3.1 "Ribbon Cables" of Paul Clayton
    Analysis of Multiconductor Transmission Lines. 2007. 
    """
    l = np.zeros([2,2])
    l[0] = [0.7485*1e-6 ,0.5077*1e-6]
    l[1] = [0.5077*1e-6, 1.0154*1e-6]
    c = np.zeros([2,2])
    c[0] = [37.432*1e-12, -18.716*1e-12]
    c[1] = [-18.716*1e-12, 24.982*1e-12]

    Zs, Zl = np.zeros([1,2]), np.zeros([1,2])
    Zs[:] = [50,50]
    Zl[:] = [50,50]

    line = mtln.MTL(l=l, c= c, length=2.0, nx = 100, Zs = Zs, Zl = Zl)
    finalTime = 200e-9

    magnitude = lambda t: wf.trapezoidal_wave(t, A = 1, rise_time=1e-9, fall_time=1e-9, f0=1e6, D=0.5)
    line.add_voltage_source(position=0.0, conductor=1, magnitude=magnitude)
    v_probe = line.add_probe(position=0.0, conductor= 0, type='voltage')
    
    for t in line.get_time_range(finalTime):
        line.step()

    plt.plot(1e9*v_probe.t, 1e3*v_probe.val)
    plt.ylabel(r'$V_1 (0, t)\,[mV]$')
    plt.xlabel(r'$t\,[\mu s]$')
    plt.xticks(range(0, 200 ,50))
    plt.grid('both')
    plt.show()

def test_pcb_paul_9_3_2():
    """
    Described in Ch. 9.3.2 "Printed Circuit Boards" of Paul Clayton
    Analysis of Multiconductor Transmission Lines. 2007. 
    """
    l = np.zeros([2,2])
    l[0] = [1.10515*1e-6, 0.690613*1e-6]
    l[1] = [0.690613*1e-6, 1.38123*1e-6]
    c = np.zeros([2,2])
    c[0] = [40.5985*1e-12, -20.2992*1e-12]
    c[1] = [-20.2992*1e-12, 29.7378*1e-12]

    Zs, Zl = np.zeros([1,2]), np.zeros([1,2])
    Zs[:] = [50,50]
    Zl[:] = [50,50]

    line = mtln.MTL(l=l, c= c, length=0.254, nx = 2, Zs = Zs, Zl = Zl)
    finalTime = 40e-9

    magnitude = lambda t: wf.trapezoidal_wave(t, A = 1, rise_time=6.25e-9, fall_time=6.25e-9, f0=1e6, D=0.5)
    line.add_voltage_source(position=0.0, conductor=1, magnitude=magnitude)
    v_probe = line.add_probe(position=0.0, conductor= 0, type='voltage')
    
    for t in line.get_time_range(finalTime):
        line.step()

    plt.plot(1e9*v_probe.t, 1e3*v_probe.val)
    plt.ylabel(r'$V_1 (0, t)\,[mV]$')
    plt.xlabel(r'$t\,[\mu s]$')
    plt.xticks(range(0, 200 ,50))
    plt.grid('both')
    plt.show()

def test_extract_network_paul_8_6_no_load():
   
    L0 = 0.25e-6
    C0 = 100e-12
    length = 400.0
    Zs = 150.0
    Zl = 0.0

    line = mtln.MTL(l=L0, c=C0, length=length, Zs=Zs, Zl=Zl)
    line_ntw = line.extract_network(fMin = 0.01e6, fMax = 1e6, finalTime = 250e-6)

    media = DistributedCircuit(line_ntw.frequency, C=C0, L=L0)
    skrf_tl = media.line(length-line.dx/2, 'm', name='line') ** media.short()
    
    assert np.allclose(np.abs(skrf_tl.s), np.abs(line_ntw.s))
    assert np.allclose(np.angle(skrf_tl.s), np.angle(line_ntw.s))

    # skrf_tl.plot_s_mag(label='skrf')
    # line_ntw.plot_s_mag(label='mtl')    
    # plt.grid()
    # plt.legend()
    
    # plt.figure()
    # skrf_tl.plot_s_deg(label='skrf')
    # line_ntw.plot_s_deg(label='mtl')    
    # plt.grid()
    # plt.legend()
    
    # plt.show()
    
def test_extract_network_paul_8_6_150ohm_load():
   
    L0 = 0.25e-6
    C0 = 100e-12
    length = 400.0
    Zs = 150.0
    Zl = 150.0

    line = mtln.MTL(l=L0, c=C0, length=length, Zs=Zs, Zl=Zl)
    line_ntw = line.extract_network(fMin = 0.01e6, fMax = 1e6, finalTime = 250e-6)

    media = DistributedCircuit(line_ntw.frequency, C=C0, L=L0)
    skrf_tl = \
        media.line(length - line.dx/2.0, 'm', name='line') \
        ** media.resistor(Zl) ** media.short()
    
    assert np.allclose(np.abs(skrf_tl.s), np.abs(line_ntw.s))
    assert np.allclose(np.angle(skrf_tl.s), np.angle(line_ntw.s))

    # skrf_tl.plot_s_mag(label='skrf')
    # line_ntw.plot_s_mag(label='mtl')    
    # plt.grid()
    # plt.legend()
    
    # plt.figure()
    # skrf_tl.plot_s_deg(label='skrf')
    # line_ntw.plot_s_deg(label='mtl')    
    # plt.grid()
    # plt.legend()
    
    # plt.show()
        

def test_cables_panel_experimental_comparison():
    assert False
