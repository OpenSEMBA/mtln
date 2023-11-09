import numpy as np
import matplotlib.pyplot as plt

import sympy as sp
from scipy.constants import epsilon_0, mu_0, speed_of_light

import src.mtl as mtl
from src.networkExtraction import *

import src.waveforms as wf

import unittest as ut

import skrf as rf
from skrf.media import DistributedCircuit
from skrf.media import Coaxial

EXPERIMENTAL_DATA = 'python/testData/cable_panel/experimental_measurements/'

def fail():
    assert(True==False)



def test_coaxial_line_paul_8_6_square():
    """ 
    Described in Ch. 8 of Paul Clayton, 
    Analysis of Multiconductor Transmission Lines. 2004. 
    """

    line = mtl.MTL(l=0.25e-6, c=100e-12, length=400.0, Zs=150)
    finalTime = 18e-6

    def magnitude(t): return wf.square_pulse(t, 100, 6e-6)
    line.add_voltage_source(position=np.array([0.0,0.0,0.0]), conductor=0, magnitude=magnitude)
    v_probe = line.add_probe(position=np.array([0.0,0.0,0.0]),   probe_type='voltage')
    i_probe = line.add_probe(position=np.array([0.0,0.0,400.0]), probe_type='current')

    line.run_until(finalTime)

    start_times = [0.1, 4.1, 6.1, 8.1, 10.1, 12.1, 14.1, 16.1]
    end_times = [3.9, 5.9, 7.9, 9.9, 11.9, 13.9, 15.9, 18.9]
    check_voltages = [25, -12.5, -37.5, -18.75, 18.75, 9.375, -9.375, -4.6875]
    for (t_start, t_end, v) in zip(start_times, end_times, check_voltages):
        start = np.argmin(np.abs(v_probe.t - t_start*1e-6))
        end = np.argmin(np.abs(v_probe.t - t_end*1e-6))
        assert np.all(np.isclose(v_probe.val[start:end], v))

    # plt.figure()
    # plt.plot(1e6*v_probe.t, v_probe.val)
    # plt.ylabel(r'$V (0, t)\,[V]$')
    # plt.xlabel(r'$t\,[\mu s]$')
    # plt.grid('both')

    # plt.show()

    # xticks = range(int(np.floor(min(1e6*i_probe.t))),
    #                int(np.ceil(max(1e6*i_probe.t))+1))

    # plt.figure()
    # plt.plot(1e6*i_probe.t, i_probe.val )
    # plt.ylabel(r'$I (L, t)\,[A]$')
    # plt.xlabel(r'$t\,[\mu s]$')
    # plt.xticks(xticks)
    # plt.grid('both')



def test_coaxial_line_paul_8_6_triangle():
    """ 
    Described in Ch. 8 of Paul Clayton, 
    Analysis of Multiconductor Transmission Lines. 2004. 
    """

    line = mtl.MTL(l=0.25e-6, c=100e-12, length=400.0, Zs=150.0)
    finalTime = 18e-6

    def magnitude(t): return wf.triangle_pulse(t, 100, 6e-6)
    line.add_voltage_source(position=0.0, conductor=0, magnitude=magnitude)
    v_probe = line.add_probe(position=0.0, probe_type='voltage')

    line.run_until(finalTime)

    times = [4.0, 5.9, 6.1, 8.0, 10.1, 12]
    voltages = [16.67, 12.5, -12.5, -25, 6.25, 12.5]
    for (t, v) in zip(times, voltages):
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
    l = np.zeros([2, 2])
    l[0] = [0.7485*1e-6, 0.5077*1e-6]
    l[1] = [0.5077*1e-6, 1.0154*1e-6]
    c = np.zeros([2, 2])
    c[0] = [37.432*1e-12, -18.716*1e-12]
    c[1] = [-18.716*1e-12, 24.982*1e-12]

    Zs, Zl = np.zeros([1, 2]), np.zeros([1, 2])
    Zs[:] = [50, 50]
    Zl[:] = [50, 50]

    line = mtl.MTL(l=l, c=c, length=2.0, ndiv=2, Zs=Zs, Zl=Zl)
    finalTime = 200e-9

    def magnitude(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=20e-9, fall_time=20e-9, f0=1e6, D=0.5)
    line.add_voltage_source(position=0.0, conductor=1, magnitude=magnitude)
    v_probe = line.add_probe(position=np.array([0.0,0.0,0.0]),  probe_type='voltage')
    v_probe1 = line.add_probe(position=np.array([0.0,0.0,1.0]), probe_type='voltage')
    i_probe1 = line.add_probe(position=np.array([0.0,0.0,1.0]), probe_type='current')

    line.run_until(finalTime)

    # From Paul's book:
    # "The crosstalk waveform rises to a peak of around 110 mV [...]"
    assert (np.isclose(np.max(v_probe.val[:, 0]), 113e-3, atol=1e-3))

    # plt.plot(1e9*v_probe.t, 1e3*v_probe.val)
    # plt.ylabel(r'$V_1 (0, t)\,[mV]$')
    # plt.xlabel(r'$t\,[ns]$')
    # plt.xticks(range(0, 200, 50))
    # plt.grid('both')
    # plt.show()


def test_ribbon_cable_1ns_paul_9_3():
    """
    Described in Ch. 9.3.1 "Ribbon Cables" of Paul Clayton
    Analysis of Multiconductor Transmission Lines. 2007. 
    """
    l = np.zeros([2, 2])
    l[0] = [0.7485*1e-6, 0.5077*1e-6]
    l[1] = [0.5077*1e-6, 1.0154*1e-6]
    c = np.zeros([2, 2])
    c[0] = [37.432*1e-12, -18.716*1e-12]
    c[1] = [-18.716*1e-12, 24.982*1e-12]

    Zs, Zl = np.zeros([1, 2]), np.zeros([1, 2])
    Zs[:] = [50, 50]
    Zl[:] = [50, 50]

    line = mtl.MTL(l=l, c=c, length=2.0, ndiv=100, Zs=Zs, Zl=Zl)
    finalTime = 200e-9

    def magnitude(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=1e-9, fall_time=1e-9, f0=1e6, D=0.5)
    line.add_voltage_source(position=0.0, conductor=1, magnitude=magnitude)
    v_probe = line.add_probe(position=0.0, probe_type='voltage')

    line.run_until(finalTime)

    times = [12.5, 25, 40, 55]
    voltages = [120, 95, 55, 32]
    for (t, v) in zip(times, voltages):
        index = np.argmin(np.abs(v_probe.t - t*1e-9))
        assert np.all(np.isclose(v_probe.val[index, 0], v*1e-3, atol=10e-3))

    # plt.plot(1e9*v_probe.t, 1e3*v_probe.val)
    # plt.ylabel(r'$V_1 (0, t)\,[mV]$')
    # plt.xlabel(r'$t\,[ns]$')
    # plt.xticks(range(0, 200, 50))
    # plt.grid('both')
    # plt.show()



def test_ribbon_cable_1ns_paul_9_3_lossless_lossy():
    """
    Described in Ch. 9.3.1 "Ribbon Cables" of Paul Clayton
    Analysis of Multiconductor Transmission Lines. 2007. 
    """
    """
    Uses lossy mtl class with no losses
    """
    l = np.zeros([2, 2])
    l[0] = [0.7485*1e-6, 0.5077*1e-6]
    l[1] = [0.5077*1e-6, 1.0154*1e-6]
    c = np.zeros([2, 2])
    c[0] = [37.432*1e-12, -18.716*1e-12]
    c[1] = [-18.716*1e-12, 24.982*1e-12]

    Zs, Zl = np.zeros([1, 2]), np.zeros([1, 2])
    Zs[:] = [50, 50]
    Zl[:] = [50, 50]

    g = np.zeros([2, 2])
    r = np.zeros([2, 2])
    line = mtl.MTL(l=l, c=c, g=g, r=r, length=2.0, ndiv=100, Zs=Zs, Zl=Zl)
    finalTime = 200e-9

    def magnitude(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=1e-9, fall_time=1e-9, f0=1e6, D=0.5)
    line.add_voltage_source(position=0.0, conductor=1, magnitude=magnitude)
    v_probe = line.add_probe(position=0.0, probe_type='voltage')

    line.run_until(finalTime)

    # plt.plot(1e9*v_probe.t, 1e3*v_probe.val[:,0])
    # plt.ylabel(r'$V_1 (0, t)\,[mV]$')
    # plt.xlabel(r'$t\,[ns]$')
    # plt.xticks(range(0, 200, 50))
    # plt.grid('both')
    # plt.show()

    times = [12.5, 25, 40, 55]
    voltages = [120, 95, 55, 32]
    for (t, v) in zip(times, voltages):
        index = np.argmin(np.abs(v_probe.t - t*1e-9))
        assert np.all(np.isclose(v_probe.val[index,0], v*1e-3, atol=10e-3))


def test_pcb_paul_9_3_2():
    """
    Described in Ch. 9.3.2 "Printed Circuit Boards" of Paul Clayton
    Analysis of Multiconductor Transmission Lines. 2007. 
    """
    l = np.zeros([2, 2])
    l[0] = [1.10515*1e-6, 0.690613*1e-6]
    l[1] = [0.690613*1e-6, 1.38123*1e-6]
    c = np.zeros([2, 2])
    c[0] = [40.5985*1e-12, -20.2992*1e-12]
    c[1] = [-20.2992*1e-12, 29.7378*1e-12]

    Zs, Zl = np.zeros([1, 2]), np.zeros([1, 2])
    Zs[:] = [50, 50]
    Zl[:] = [50, 50]

    line = mtl.MTL(l=l, c=c, length=0.254, ndiv=2, Zs=Zs, Zl=Zl)
    finalTime = 40e-9

    def magnitude(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=6.25e-9, fall_time=6.25e-9, f0=1e6, D=0.5)
    line.add_voltage_source(position=0.0, conductor=1, magnitude=magnitude)
    v_probe = line.add_probe(position=0.0, probe_type='voltage')

    line.run_until(finalTime)

    times = [5, 10, 15, 20]
    voltages = [80, 62.5, 23, 8]
    for (t, v) in zip(times, voltages):
        index = np.argmin(np.abs(v_probe.t - t*1e-9))
        assert np.all(np.isclose(v_probe.val[index, 0], v*1e-3, atol=10e-3))

    # plt.plot(1e9*v_probe.t, 1e3*v_probe.val)
    # plt.ylabel(r'$V_1 (0, t)\,[mV]$')
    # plt.xlabel(r'$t\,[\mu s]$')
    # plt.xticks(range(0, 40, 5))
    # plt.grid('both')
    # plt.show()


