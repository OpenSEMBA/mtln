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

    # line = mtl.MTL(l=l, c=c, length=2.0, nx=2, Zs=Zs, Zl=Zl)
    finalTime = 200e-9

    def magnitude(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=20e-9, fall_time=20e-9, f0=1e6, D=0.5)

    # line.add_voltage_source(position=0.0, conductor=1, magnitude=magnitude)
    # v_probe = line.add_probe(position=0.0, type='voltage')

    # line.run_until(finalTime)

    """
     _             _
    | |     1     | |
    | 1-----------3 |
    | |     0     | |
    | 0-----------2 |
    |_|           |_|
    term_1(0)     term_2(1)
    
    """
    mtl_nw = mtln.MTLN()
    bundle_0 = mtl.MTL(l=l, c=c, length=2.0, nx=2)
    v_probe = bundle_0.add_probe(position=0.0, type='voltage')

    mtl_nw.add_bundle(0, bundle_0)

    #network definition
    terminal_1 = mtln.Network(nw_number = 0, nodes = [0,1])
    bundle_connections= [{"node" : 0, "conductor" : 0},{"node" : 1, "conductor" : 1}]
    
    terminal_1.add_nodes_in_bundle(bundle_number = 0, 
                                   bundle = bundle_0,
                                   connections = bundle_connections, 
                                   side= "S")
    #network connections
    terminal_1.connect_to_ground(node = 0, R  = 50, side = "S")
    terminal_1.connect_to_ground(node = 1, R= 50, Vt = magnitude, side = "S")
    mtl_nw.add_network(terminal_1)
    
    #network definition
    terminal_2 = mtln.Network(nw_number = 1 ,nodes = [2,3])
    bundle_connections= [{"node" : 2, "conductor" : 0},{"node" : 3, "conductor" : 1}]
    terminal_2.add_nodes_in_bundle(bundle_number = 0, 
                                   bundle = bundle_0, 
                                   connections= bundle_connections, 
                                   side= "L")

    #network connections
    terminal_2.connect_to_ground(2, 50, side = "L")
    terminal_2.connect_to_ground(3, 50, side = "L")
    mtl_nw.add_network(terminal_2)


    mtl_nw.run_until(finalTime)

    # From Paul's book:
    # "The crosstalk waveform rises to a peak of around 110 mV [...]"
    plt.plot(1e9*v_probe.t, 1e3*v_probe.val)
    plt.ylabel(r'$V_1 (0, t)\,[mV]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 200, 50))
    plt.grid('both')
    plt.show()
    assert (np.isclose(np.max(v_probe.val[:, 0]), 113e-3, atol=1e-3))

