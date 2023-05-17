import numpy as np
import matplotlib.pyplot as plt

import sympy as sp
from scipy.constants import epsilon_0, mu_0, speed_of_light

import src.mtl as mtl
import src.mtln as mtln


import src.waveforms as wf

import skrf as rf
from skrf.media import DistributedCircuit

def test_ribbon_cable_20ns_paul_9_3():
    """
    Described in Ch. 9.3.1 "Ribbon Cables" of Paul Clayton
    Analysis of Multiconductor Transmission Lines. 2007. 
    """
    """
    Solved with mtl approach and mltn approach: tube + 2 termination networks
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

    line = mtl.MTL(l=l, c=c, length=2.0, nx=2, Zs=Zs, Zl=Zl)
    finalTime = 200e-9

    def magnitude(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=20e-9, fall_time=20e-9, f0=1e6, D=0.5)
    line.add_voltage_source(position=0.0, conductor=1, magnitude=magnitude)
    v_probe = line.add_probe(position=0.0, type='voltage')

    line.run_until(finalTime)

    """
     _             _
    | |     0     | |
    | 0-----------2 |
    | |           | |
    | 1-----------3 |
    |_|     1     |_|
    T1             T2
    
    """

    line_nw = mtln.MTLN(l=l, c=c, length=2.0, nx=2)
    term_1 = mtln.Network([0,1])
    term_2 = mtln.Network([2,3])
    term_1.add_connection(0, [line_nw,0])
    term_1.add_connection(1, [line_nw,1])
    term_2.add_connection(2, [line_nw,0])
    term_2.add_connection(3, [line_nw,1])

    # From Paul's book:
    # "The crosstalk waveform rises to a peak of around 110 mV [...]"
    assert (np.isclose(np.max(v_probe.val[:, 0]), 113e-3, atol=1e-3))

