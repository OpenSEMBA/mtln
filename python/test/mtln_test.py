import numpy as np
import matplotlib.pyplot as plt

import sympy as sp
from scipy.constants import epsilon_0, mu_0, speed_of_light

import src.mtl as mtl
import src.mtln as mtln


import src.waveforms as wf

import skrf as rf
from skrf.media import DistributedCircuit

def test_ribbon_cable_20ns_termination_network():
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

    finalTime = 200e-9

    def magnitude(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=20e-9, fall_time=20e-9, f0=1e6, D=0.5)


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
    terminal_1 = mtln.Network(nw_number = 0, nodes = [0,1], bundles = [0])
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
    terminal_2 = mtln.Network(nw_number = 1 ,nodes = [2,3], bundles = [0])
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

def test_ribbon_cable_1ns_paul_interconnection_network():
    """
    Similar to MTL described in Ch. 9.3.1 "Ribbon Cables" of Paul Clayton
    Analysis of Multiconductor Transmission Lines. 2007. 
    But: the MTL is divided in two equal length MTLs, which are shorted
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


    """
     _             _             _
    | |     1     | |     1     | |
    | 1-----------3-5-----------7 |
    | |     b0    | |     b1    | |
    | 0-----------2-4-----------6 |
    |_|     0     |_|     0     |_|
    term_1(0)     iconn(1)     term_2(2)
    
    """


    bundle_0 = mtl.MTL(l=l, c=c, length=1.0, nx=50)
    bundle_1 = mtl.MTL(l=l, c=c, length=1.0, nx=50)
    finalTime = 200e-9

    def magnitude(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=1e-9, fall_time=1e-9, f0=1e6, D=0.5)
    

    v_probe = bundle_0.add_probe(position=0.0, type='voltage')

    mtl_nw = mtln.MTLN()
    mtl_nw.add_bundle(0, bundle_0)
    mtl_nw.add_bundle(1, bundle_1)

    #network definition
    terminal_1 = mtln.Network(nw_number = 0, nodes = [0,1], bundles = [0])
    bundle_connections= [{"node" : 0, "conductor" : 0},{"node" : 1, "conductor" : 1}]
    
    terminal_1.add_nodes_in_bundle(bundle_number = 0, 
                                   bundle = bundle_0,
                                   connections = bundle_connections, 
                                   side= "S")
    #network connections
    terminal_1.connect_to_ground(node = 0, R  = 50, side = "S")
    terminal_1.connect_to_ground(node = 1, R= 50, Vt = magnitude, side = "S")

    #interconnection network
    iconn = mtln.Network(nw_number=1, nodes = [2,3,4,5], bundles = [0,1])
    bundle_0_connections = [{"node" : 2, "conductor" : 0},{"node" : 3, "conductor" : 1}]
    bundle_1_connections = [{"node" : 4, "conductor" : 0},{"node" : 5, "conductor" : 1}]
    iconn.add_nodes_in_bundle(bundle_number = 0, 
                                   bundle = bundle_0, 
                                   connections= bundle_0_connections, 
                                   side= "L")
    iconn.add_nodes_in_bundle(bundle_number = 1, 
                                   bundle = bundle_1, 
                                   connections= bundle_1_connections, 
                                   side= "S")
    iconn.short_nodes(4,2)
    iconn.short_nodes(5,3)

    #network definition
    terminal_3 = mtln.Network(nw_number = 2 ,nodes = [6,7], bundles = [1])
    bundle_connections= [{"node" : 6, "conductor" : 0},{"node" : 7, "conductor" : 1}]
    terminal_3.add_nodes_in_bundle(bundle_number = 1, 
                                   bundle = bundle_1, 
                                   connections= bundle_connections, 
                                   side= "L")

    #network connections
    terminal_3.connect_to_ground(6, 50, side = "L")
    terminal_3.connect_to_ground(7, 50, side = "L")

    #add networks
    mtl_nw.add_network(terminal_1)
    mtl_nw.add_network(iconn)
    mtl_nw.add_network(terminal_3)


    mtl_nw.run_until(finalTime)


    plt.plot(1e9*v_probe.t, 1e3*v_probe.val)
    plt.ylabel(r'$V_1 (0, t)\,[mV]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 200, 50))
    plt.grid('both')
    plt.show()

    times = [12.5, 25, 40, 55]
    voltages = [120, 95, 55, 32]
    for (t, v) in zip(times, voltages):
        index = np.argmin(np.abs(v_probe.t - t*1e-9))
        assert np.all(np.isclose(v_probe.val[index, 0], v*1e-3, atol=10e-3))

    #skrf
    freq = [1,2,3] #f?
    media = DistributedCircuit(freq, C=c, L=l)
    ribbon_0 = media.line(1.0, 'm',
                         name='line', embed=True, z0=[91.96,254.93])
    ribbon_1 = media.line(1.0, 'm',
                         name='line', embed=True, z0=[91.96,254.93])
    
    port1 = rf.Circuit.Port(frequency=freq, name='port1', z0=50)
    port2 = rf.Circuit.Port(frequency=freq, name='port2', z0=50)
    cnL = [
        [(port1, 0), (ribbon_0, 0)],
        [(port2, 0), (ribbon_0, 1)]
    ]
    LRibbon = rf.Circuit(cnL)

    power = [1, 0]  # 1 Watt at port1 and 0 at port2
    phase = [0, 0]  # 0 radians
    V_at_ports = LRibbon.voltages_external(power, phase)
    I_at_ports = LRibbon.currents_external(power, phase)
    cnR = [
        [(ribbon_1, 0), (port1, 0)],
        [(ribbon_1, 0), (port2, 1)]
    ]
    RibbonR = rf.Circuit(cnR)
    tl = LRibbon ** RibbonR

    
def test_ribbon_cable_1ns_R_interconnection_network():
    """
    Similar to MTL described in Ch. 9.3.1 "Ribbon Cables" of Paul Clayton
    Analysis of Multiconductor Transmission Lines. 2007. 
    But: the MTL is divided in two equal length MTLs, which are connected by resistors
    Results are compared against NgSpice simulations
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


    """
     _             __________             _
    | |     1     |          |     1     | |
    | 1-----------3--R-------5-----------7 |
    | |     b0    |          |     b1    | |
    | 0-----------2--R-------4-----------6 |
    |_|     0     |__________|     0     |_|
    term_1(0)     iconn(1)     term_2(2)
    
    """


    bundle_0 = mtl.MTL(l=l, c=c, length=1.0, nx=50)
    bundle_1 = mtl.MTL(l=l, c=c, length=1.0, nx=50)
    finalTime = 200e-9

    def magnitude(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=1e-9, fall_time=1e-9, f0=1e6, D=0.5)
    
    def V35(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=15e-9, fall_time=5e-9, f0=1e02, D=9.5e-6)

    v_probe = bundle_0.add_probe(position=0.0, type='voltage')

    mtl_nw = mtln.MTLN()
    mtl_nw.add_bundle(0, bundle_0)
    mtl_nw.add_bundle(1, bundle_1)

    #network definition
    terminal_1 = mtln.Network(nw_number = 0, nodes = [0,1], bundles = [0])
    bundle_connections= [{"node" : 0, "conductor" : 0},{"node" : 1, "conductor" : 1}]
    
    terminal_1.add_nodes_in_bundle(bundle_number = 0, 
                                   bundle = bundle_0,
                                   connections = bundle_connections, 
                                   side= "S")
    #network connections
    terminal_1.connect_to_ground(node = 0, R  = 50, side = "S")
    terminal_1.connect_to_ground(node = 1, R= 50, Vt = magnitude, side = "S")

    #interconnection network
    iconn = mtln.Network(nw_number=1, nodes = [2,3,4,5], bundles = [0,1])
    bundle_0_connections = [{"node" : 2, "conductor" : 0},{"node" : 3, "conductor" : 1}]
    bundle_1_connections = [{"node" : 4, "conductor" : 0},{"node" : 5, "conductor" : 1}]
    iconn.add_nodes_in_bundle(bundle_number = 0, 
                                   bundle = bundle_0, 
                                   connections= bundle_0_connections, 
                                   side= "L")
    iconn.add_nodes_in_bundle(bundle_number = 1, 
                                   bundle = bundle_1, 
                                   connections= bundle_1_connections, 
                                   side= "S")
    
    iconn.connect_nodes(5,3, R = 10)
    iconn.connect_nodes(4,2, R = 25)

    #network definition
    terminal_3 = mtln.Network(nw_number = 2 ,nodes = [6,7],bundles = [1])
    bundle_connections= [{"node" : 6, "conductor" : 0},{"node" : 7, "conductor" : 1}]
    terminal_3.add_nodes_in_bundle(bundle_number = 1, 
                                   bundle = bundle_1, 
                                   connections= bundle_connections, 
                                   side= "L")

    #network connections
    terminal_3.connect_to_ground(6, 50, side = "L")
    terminal_3.connect_to_ground(7, 50, side = "L")

    mtl_nw.add_network(terminal_1)
    mtl_nw.add_network(iconn)
    mtl_nw.add_network(terminal_3)

    mtl_nw.run_until(finalTime)

    t0, V0 = np.genfromtxt('python/testData/ngspice/test_ribbon_cable_1ns_R_interconnection_network/V1.txt', delimiter=',', usecols=(0,1), unpack = True)
    t1, V1 = np.genfromtxt('python/testData/ngspice/test_ribbon_cable_1ns_R_interconnection_network/V2.txt', delimiter=',', usecols=(0,1), unpack = True)

    plt.plot(1e9*v_probe.t, v_probe.val[:,0] ,'r', label = 'Conductor 0')
    plt.plot(1e9*v_probe.t, v_probe.val[:,1] ,'b', label = 'Conductor 1')
    plt.plot(1e9*t0, V0 ,'g--', label = 'Conductor 0 - NgSpice')
    plt.plot(1e9*t1, V1 ,'k--', label = 'Conductor 1 - NgSpice')
    plt.ylabel(r'$V (0, t)\,[V]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 200, 50))
    plt.grid('both')
    plt.legend()
    plt.savefig("test_ribbon_cable_1ns_R_interconnection_network.png")

def test_ribbon_cable_1ns_RV_interconnection_network():
    """
    Similar to MTL described in Ch. 9.3.1 "Ribbon Cables" of Paul Clayton
    Analysis of Multiconductor Transmission Lines. 2007. 
    But: the MTL is divided in two equal length MTLs, which are connected by resistors and sources
    Results are compared against NgSpice simulations
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


    """
     _             __________             _
    | |     1     |          |     1     | |
    | 1-----------3--R--V35--5-----------7 |
    | |     b0    |          |     b1    | |
    | 0-----------2--R-------4-----------6 |
    |_|     0     |__________|     0     |_|
    term_1(0)     iconn(1)     term_2(2)
    
    """


    bundle_0 = mtl.MTL(l=l, c=c, length=1.0, nx=50)
    bundle_1 = mtl.MTL(l=l, c=c, length=1.0, nx=50)
    finalTime = 200e-9

    def magnitude(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=1e-9, fall_time=1e-9, f0=1e6, D=0.5)
    
    def V35(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=15e-9, fall_time=5e-9, f0=1e02, D=9.5e-6)

    v_probe = bundle_0.add_probe(position=0.0, type='voltage')

    mtl_nw = mtln.MTLN()
    mtl_nw.add_bundle(0, bundle_0)
    mtl_nw.add_bundle(1, bundle_1)

    #network definition
    terminal_1 = mtln.Network(nw_number = 0, nodes = [0,1], bundles = [0])
    bundle_connections= [{"node" : 0, "conductor" : 0},{"node" : 1, "conductor" : 1}]
    
    terminal_1.add_nodes_in_bundle(bundle_number = 0, 
                                   bundle = bundle_0,
                                   connections = bundle_connections, 
                                   side= "S")
    #network connections
    terminal_1.connect_to_ground(node = 0, R  = 50, side = "S")
    terminal_1.connect_to_ground(node = 1, R= 50, Vt = magnitude, side = "S")

    #interconnection network
    iconn = mtln.Network(nw_number=1, nodes = [2,3,4,5], bundles = [0,1])
    bundle_0_connections = [{"node" : 2, "conductor" : 0},{"node" : 3, "conductor" : 1}]
    bundle_1_connections = [{"node" : 4, "conductor" : 0},{"node" : 5, "conductor" : 1}]
    iconn.add_nodes_in_bundle(bundle_number = 0, 
                                   bundle = bundle_0, 
                                   connections= bundle_0_connections, 
                                   side= "L")
    iconn.add_nodes_in_bundle(bundle_number = 1, 
                                   bundle = bundle_1, 
                                   connections= bundle_1_connections, 
                                   side= "S")
    
    iconn.connect_nodes(5,3, R = 10, Vt = V35)
    iconn.connect_nodes(4,2, R = 25)

    #network definition
    terminal_3 = mtln.Network(nw_number = 2 ,nodes = [6,7],bundles = [1])
    bundle_connections= [{"node" : 6, "conductor" : 0},{"node" : 7, "conductor" : 1}]
    terminal_3.add_nodes_in_bundle(bundle_number = 1, 
                                   bundle = bundle_1, 
                                   connections= bundle_connections, 
                                   side= "L")

    #network connections
    terminal_3.connect_to_ground(6, 50, side = "L")
    terminal_3.connect_to_ground(7, 50, side = "L")

    mtl_nw.add_network(terminal_1)
    mtl_nw.add_network(iconn)
    mtl_nw.add_network(terminal_3)

    mtl_nw.run_until(finalTime)

    t0, V0 = np.genfromtxt('python/testData/ngspice/test_ribbon_cable_1ns_RV_interconnection_network/V1.txt', delimiter=',', usecols=(0,1), unpack = True)
    t1, V1 = np.genfromtxt('python/testData/ngspice/test_ribbon_cable_1ns_RV_interconnection_network/V2.txt', delimiter=',', usecols=(0,1), unpack = True)

    plt.plot(1e9*v_probe.t, v_probe.val[:,0] ,'r', label = 'Conductor 0')
    plt.plot(1e9*v_probe.t, v_probe.val[:,1] ,'b', label = 'Conductor 1')
    plt.plot(1e9*t0, V0 ,'g--', label = 'Conductor 0 - NgSpice')
    plt.plot(1e9*t1, V1 ,'k--', label = 'Conductor 1 - NgSpice')
    plt.ylabel(r'$V (0, t)\,[V]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 200, 50))
    plt.grid('both')
    plt.legend()
    plt.savefig("test_ribbon_cable_1ns_RV_interconnection_network.png")

def test_ribbon_cable_1ns_RV_T_network():
    """
    Similar to MTL described in Ch. 9.3.1 "Ribbon Cables" of Paul Clayton
    Analysis of Multiconductor Transmission Lines. 2007. 
    But: the MTL is divided in two equal length MTLs, which are connected by resistors and sources
    Results are compared against NgSpice simulations
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


    """
     ______________             __________             _______
    |              |     1     |          |     1     |       |
    | g--Vmag--R0- 1-----------3--R1-V35--5-----------7--R0-g |
    |              |   b0:0.75 |          |  b1:1.5   |       |
    |        g--R0-0-----------2-R2    R3-4-----------6-R0-g  |
    |______________|     0     |___\__/___|     0     |_______|
    term_1(0)                      8  9 iconn(1)        term_2(2)
                                   |  | 
                                   |  | b2:0.5
                                   |  |
                               __10|__|11__
                             |     |  |    |
                             |     R0 R0   |
                             |     |  |    |
                             |     |  V35  |
                             |     |  |    |
                             |     g  g    |
                             |_____________|
    
    """

    R0 = 50
    R1 = 10
    R2 = 25
    R3 = 20

    bundle_0 = mtl.MTL(l=l, c=c, length=1.0, nx=50)
    bundle_1 = mtl.MTL(l=l, c=c, length=1.0, nx=50)
    bundle_2 = mtl.MTL(l=l, c=c, length=1.0, nx=50)
    finalTime = 200e-9

    def magnitude(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=1e-9, fall_time=1e-9, f0=1e6, D=0.5)
    
    def V35(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=15e-9, fall_time=5e-9, f0=1e02, D=9.5e-6)

    v_probe = bundle_0.add_probe(position=0.0, type='voltage')

    mtl_nw = mtln.MTLN()
    mtl_nw.add_bundle(0, bundle_0)
    mtl_nw.add_bundle(1, bundle_1)
    mtl_nw.add_bundle(2, bundle_2)

    #TERMINAL 1
    terminal_1 = mtln.Network(nw_number = 0, nodes = [0,1], bundles = [0])
    bundle_connections= [{"node" : 0, "conductor" : 0},{"node" : 1, "conductor" : 1}]
    
    terminal_1.add_nodes_in_bundle(bundle_number = 0, 
                                   bundle = bundle_0,
                                   connections = bundle_connections, 
                                   side= "S")
    terminal_1.connect_to_ground(node = 0, R = R0, side = "S")
    terminal_1.connect_to_ground(node = 1, R = R0, Vt = magnitude, side = "S")

    #ICONN
    iconn = mtln.Network(nw_number=1, nodes = [2,3,4,5,8,9], bundles = [0,1,2])
    bundle_0_connections = [{"node" : 2, "conductor" : 0},{"node" : 3, "conductor" : 1}]
    bundle_1_connections = [{"node" : 4, "conductor" : 0},{"node" : 5, "conductor" : 1}]
    bundle_2_connections = [{"node" : 8, "conductor" : 0},{"node" : 9, "conductor" : 1}]
    iconn.add_nodes_in_bundle(bundle_number = 0, 
                                   bundle = bundle_0, 
                                   connections= bundle_0_connections, 
                                   side= "L")
    iconn.add_nodes_in_bundle(bundle_number = 1, 
                                   bundle = bundle_1, 
                                   connections= bundle_1_connections, 
                                   side= "S")
    iconn.add_nodes_in_bundle(bundle_number = 2, 
                                   bundle = bundle_2, 
                                   connections= bundle_2_connections, 
                                   side= "S")
    
    iconn.connect_nodes(5,3, R = R1, Vt = V35)
    iconn.connect_nodes(4,9, R = R3)
    iconn.connect_nodes(8,2, R = R2)

    #TERMINAL 3
    terminal_3 = mtln.Network(nw_number = 2 ,nodes = [6,7],bundles = [1])
    bundle_connections= [{"node" : 6, "conductor" : 0},{"node" : 7, "conductor" : 1}]
    terminal_3.add_nodes_in_bundle(bundle_number = 1, 
                                   bundle = bundle_1, 
                                   connections= bundle_connections, 
                                   side= "L")

    terminal_3.connect_to_ground(6, R = R0, side = "L")
    terminal_3.connect_to_ground(7, R = R0, side = "L")

    #TERMINAL 4
    terminal_4 = mtln.Network(nw_number = 3 ,nodes = [10,11],bundles = [2])
    bundle_connections= [{"node" : 10, "conductor" : 0},{"node" : 11, "conductor" : 1}]
    terminal_4.add_nodes_in_bundle(bundle_number = 2, 
                                   bundle = bundle_2, 
                                   connections= bundle_connections, 
                                   side= "L")

    terminal_4.connect_to_ground(10, R0, side = "L")
    terminal_4.connect_to_ground(11, R0, Vt = V35, side = "L")

    mtl_nw.add_network(terminal_1)
    mtl_nw.add_network(iconn)
    mtl_nw.add_network(terminal_3)
    mtl_nw.add_network(terminal_4)

    mtl_nw.run_until(finalTime)

    t0, V0 = np.genfromtxt('python/testData/ngspice/test_ribbon_cable_1ns_RV_T_network/V1.txt', delimiter=',', usecols=(0,1), unpack = True)
    t1, V1 = np.genfromtxt('python/testData/ngspice/test_ribbon_cable_1ns_RV_T_network/V2.txt', delimiter=',', usecols=(0,1), unpack = True)

    plt.plot(1e9*v_probe.t, v_probe.val[:,0] ,'r', label = 'Conductor 0')
    plt.plot(1e9*v_probe.t, v_probe.val[:,1] ,'b', label = 'Conductor 1')
    plt.plot(1e9*t0, V0 ,'g--', label = 'Conductor 0 - NgSpice')
    plt.plot(1e9*t1, V1 ,'k--', label = 'Conductor 1 - NgSpice')
    plt.ylabel(r'$V (0, t)\,[V]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 200, 50))
    plt.grid('both')
    plt.legend()
    # plt.savefig("test_ribbon_cable_1ns_RV_T_network.png")
    plt.show()

    
def test_1_conductor_network_Z50():


    def magnitude(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=20e-9, fall_time=20e-9, f0=1e6, D=0.5)

    """
     _             _
    | |           | |
    | |  0: Z=50  | |
    | 0-----------2 |
    |_|           |_|
    term_1(0)     term_2(1)
    
    """
    mtl_nw = mtln.MTLN()
    bundle_0 = mtl.MTL(l=0.25e-6, c=100e-12, length=1.0, nx=50)
    v_probe = bundle_0.add_probe(position=0.0, type='voltage')

    mtl_nw.add_bundle(0, bundle_0)

    #network definition
    terminal_1 = mtln.Network(nw_number = 0, nodes = [0], bundles = [0])
    bundle_connections= [{"node" : 0, "conductor" : 0}]
    
    terminal_1.add_nodes_in_bundle(bundle_number = 0, 
                                   bundle = bundle_0,
                                   connections = bundle_connections, 
                                   side= "S")
    #network connections
    terminal_1.connect_to_ground(node = 0, R= 50, Vt = magnitude, side = "S")
    mtl_nw.add_network(terminal_1)
    
    #network definition
    terminal_2 = mtln.Network(nw_number = 1 ,nodes = [1,], bundles = [0])
    bundle_connections= [{"node" : 1, "conductor" : 0}]
    terminal_2.add_nodes_in_bundle(bundle_number = 0, 
                                   bundle = bundle_0, 
                                   connections= bundle_connections, 
                                   side= "L")

    #network connections
    terminal_2.connect_to_ground(1, 50, side = "L")
    mtl_nw.add_network(terminal_2)

    mtl_nw.run_until(200e-9)

    t, V0 = np.genfromtxt('python/testData/qucs/MTLN_1_conductor_network_Z50.csv', delimiter=';', names = True, usecols=(0,1), unpack = True)

    plt.plot(1e9*v_probe.t, 1e3*v_probe.val, label = "MTLN")
    plt.plot(1e9*t, 1e3*V0,'--', label = "QUCS")
    plt.ylabel(r'$V_1 (0, t)\,[mV]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 200, 50))
    plt.grid('both')
    plt.legend()
    plt.show()

def test_1_conductor_network_Z5():


    def magnitude(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=20e-9, fall_time=20e-9, f0=1e6, D=0.5)

    """
     _             _
    | |           | |
    | |  0: Z=5   | |
    | 0-----------2 |
    |_|           |_|
    term_1(0)     term_2(1)
    
    """
    mtl_nw = mtln.MTLN()
    bundle_0 = mtl.MTL(l=0.25e-8, c=100e-12, length=1.0, nx=50)
    v_probe = bundle_0.add_probe(position=0.0, type='voltage')

    mtl_nw.add_bundle(0, bundle_0)

    #network definition
    terminal_1 = mtln.Network(nw_number = 0, nodes = [0], bundles = [0])
    bundle_connections= [{"node" : 0, "conductor" : 0}]
    
    terminal_1.add_nodes_in_bundle(bundle_number = 0, 
                                   bundle = bundle_0,
                                   connections = bundle_connections, 
                                   side= "S")
    #network connections
    terminal_1.connect_to_ground(node = 0, R= 50, Vt = magnitude, side = "S")
    mtl_nw.add_network(terminal_1)
    
    #network definition
    terminal_2 = mtln.Network(nw_number = 1 ,nodes = [1,], bundles = [0])
    bundle_connections= [{"node" : 1, "conductor" : 0}]
    terminal_2.add_nodes_in_bundle(bundle_number = 0, 
                                   bundle = bundle_0, 
                                   connections= bundle_connections, 
                                   side= "L")

    #network connections
    terminal_2.connect_to_ground(1, 50, side = "L")
    mtl_nw.add_network(terminal_2)

    mtl_nw.run_until(200e-9)

    t, V0 = np.genfromtxt('python/testData/qucs/MTLN_1_conductor_network_Z5.csv', delimiter=';', names = True, usecols=(0,1), unpack = True)

    plt.plot(1e9*v_probe.t, 1e3*v_probe.val, label = "MTLN")
    plt.plot(1e9*t, 1e3*V0,'--', label = "QUCS")
    plt.ylabel(r'$V_1 (0, t)\,[mV]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 200, 50))
    plt.grid('both')
    plt.legend()
    plt.show()

def test_1_conductor_network_Z100():


    def magnitude(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=20e-9, fall_time=20e-9, f0=1e6, D=0.5)

    """
     _             _
    | |           | |
    | |  0: Z=5   | |
    | 0-----------2 |
    |_|           |_|
    term_1(0)     term_2(1)
    
    """
    mtl_nw = mtln.MTLN()
    bundle_0 = mtl.MTL(l=1e-6, c=100e-12, length=1.0, nx=50)
    v_probe = bundle_0.add_probe(position=0.0, type='voltage')

    mtl_nw.add_bundle(0, bundle_0)

    #network definition
    terminal_1 = mtln.Network(nw_number = 0, nodes = [0], bundles = [0])
    bundle_connections= [{"node" : 0, "conductor" : 0}]
    
    terminal_1.add_nodes_in_bundle(bundle_number = 0, 
                                   bundle = bundle_0,
                                   connections = bundle_connections, 
                                   side= "S")
    #network connections
    terminal_1.connect_to_ground(node = 0, R= 50, Vt = magnitude, side = "S")
    mtl_nw.add_network(terminal_1)
    
    #network definition
    terminal_2 = mtln.Network(nw_number = 1 ,nodes = [1,], bundles = [0])
    bundle_connections= [{"node" : 1, "conductor" : 0}]
    terminal_2.add_nodes_in_bundle(bundle_number = 0, 
                                   bundle = bundle_0, 
                                   connections= bundle_connections, 
                                   side= "L")

    #network connections
    terminal_2.connect_to_ground(1, 50, side = "L")
    mtl_nw.add_network(terminal_2)

    mtl_nw.run_until(200e-9)

    t, V0 = np.genfromtxt('python/testData/qucs/MTLN_1_conductor_network_Z100.csv', delimiter=';', names = True, usecols=(0,1), unpack = True)

    plt.plot(1e9*v_probe.t, 1e3*v_probe.val, label = "MTLN")
    plt.plot(1e9*t, 1e3*V0,'--', label = "QUCS")
    plt.ylabel(r'$V_1 (0, t)\,[mV]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 200, 50))
    plt.grid('both')
    plt.legend()
    plt.show()

def test_1_conductor_adapted_network_R():
    """
     _             _             _
    | |     b0    | |     b1    | |
    | 0-----------1-2-----------3 |
    |_|     0     |_|     0     |_|
    term_1(0)     iconn(1)     term_2(2)
    
    """
    for R in [25,50,100,150]:

        bundle_0 = mtl.MTL(l=0.25e-6, c=100e-12, length=1.0, nx=50)
        bundle_1 = mtl.MTL(l=0.25e-6, c=100e-12, length=1.0, nx=50)
        finalTime = 200e-9

        def magnitude(t): return wf.trapezoidal_wave(
            t, A=1, rise_time=1e-9, fall_time=1e-9, f0=1e6, D=0.5)
        

        v_probe = bundle_0.add_probe(position=0.0, type='voltage')

        mtl_nw = mtln.MTLN()
        mtl_nw.add_bundle(0, bundle_0)
        mtl_nw.add_bundle(1, bundle_1)

        #network definition
        terminal_1 = mtln.Network(nw_number = 0, nodes = [0], bundles = [0])
        bundle_connections= [{"node" : 0, "conductor" : 0}]
        
        terminal_1.add_nodes_in_bundle(bundle_number = 0, 
                                    bundle = bundle_0,
                                    connections = bundle_connections, 
                                    side= "S")
        #network connections
        terminal_1.connect_to_ground(node = 0, R= 50, Vt = magnitude, side = "S")

        #interconnection network
        iconn = mtln.Network(nw_number=1, nodes = [1,2], bundles = [0,1])
        bundle_0_connections = [{"node" : 1, "conductor" : 0}]
        bundle_1_connections = [{"node" : 2, "conductor" : 0}]
        iconn.add_nodes_in_bundle(bundle_number = 0, 
                                    bundle = bundle_0, 
                                    connections= bundle_0_connections, 
                                    side= "L")
        iconn.add_nodes_in_bundle(bundle_number = 1, 
                                    bundle = bundle_1, 
                                    connections= bundle_1_connections, 
                                    side= "S")
        # iconn.short_nodes(2,1)
        iconn.connect_nodes(2,1, R = R)

        #network definition
        terminal_3 = mtln.Network(nw_number = 2 ,nodes = [3], bundles = [1])
        bundle_connections= [{"node" : 3, "conductor" : 0}]
        terminal_3.add_nodes_in_bundle(bundle_number = 1, 
                                    bundle = bundle_1, 
                                    connections= bundle_connections, 
                                    side= "L")

        #network connections
        terminal_3.connect_to_ground(3, 50, side = "L")

        #add networks
        mtl_nw.add_network(terminal_1)
        mtl_nw.add_network(iconn)
        mtl_nw.add_network(terminal_3)


        mtl_nw.run_until(finalTime)

        t, V0 = np.genfromtxt('python/testData/qucs/MTLN_1_conductor_network_R'+str(R)+'.csv', delimiter=';', names = True, usecols=(0,1), unpack = True)

        plt.plot(1e9*v_probe.t, 1e3*v_probe.val, label = "MTLN")
        plt.plot(1e9*t, 1e3*V0,'--', label = "QUCS")
        plt.title("Two 1-cond.lines, R = "+str(R))
        plt.ylabel(r'$V_1 (0, t)\,[mV]$')
        plt.xlabel(r'$t\,[ns]$')
        plt.xticks(range(0, 200, 50))
        plt.grid('both')
        plt.legend()
        plt.savefig("MTLN_1_conductor_network_R"+str(R)+".png")
        plt.clf()

def test_1_conductor_not_adapted_network_R():
    """
     _             ____             _
    | |     b0    |    |     b1    | |
    | 0-----------1-R--2-----------3 |
    |_|     0     |____|     0     |_|
    term_1(0)     iconn(1)     term_2(2)
    
    """
    for R in [25,50,100,150]:

        bundle_0 = mtl.MTL(l=0.25e-6*1.125, c=100e-12/2, length=1.0, nx=50)
        bundle_1 = mtl.MTL(l=0.25e-6*2,     c=100e-12/2, length=1.0, nx=50)
        finalTime = 200e-9

        def magnitude(t): return wf.trapezoidal_wave(
            t, A=1, rise_time=1e-9, fall_time=1e-9, f0=1e6, D=0.5)
        

        v_probe = bundle_0.add_probe(position=0.0, type='voltage')

        mtl_nw = mtln.MTLN()
        mtl_nw.add_bundle(0, bundle_0)
        mtl_nw.add_bundle(1, bundle_1)

        #network definition
        terminal_1 = mtln.Network(nw_number = 0, nodes = [0], bundles = [0])
        bundle_connections= [{"node" : 0, "conductor" : 0}]
        
        terminal_1.add_nodes_in_bundle(bundle_number = 0, 
                                    bundle = bundle_0,
                                    connections = bundle_connections, 
                                    side= "S")
        #network connections
        terminal_1.connect_to_ground(node = 0, R= 50, Vt = magnitude, side = "S")

        #interconnection network
        iconn = mtln.Network(nw_number=1, nodes = [1,2], bundles = [0,1])
        bundle_0_connections = [{"node" : 1, "conductor" : 0}]
        bundle_1_connections = [{"node" : 2, "conductor" : 0}]
        iconn.add_nodes_in_bundle(bundle_number = 0, 
                                    bundle = bundle_0, 
                                    connections= bundle_0_connections, 
                                    side= "L")
        iconn.add_nodes_in_bundle(bundle_number = 1, 
                                    bundle = bundle_1, 
                                    connections= bundle_1_connections, 
                                    side= "S")
        # iconn.short_nodes(2,1)
        iconn.connect_nodes(2,1, R = R)

        #network definition
        terminal_3 = mtln.Network(nw_number = 2 ,nodes = [3], bundles = [1])
        bundle_connections= [{"node" : 3, "conductor" : 0}]
        terminal_3.add_nodes_in_bundle(bundle_number = 1, 
                                    bundle = bundle_1, 
                                    connections= bundle_connections, 
                                    side= "L")

        #network connections
        terminal_3.connect_to_ground(3, 50, side = "L")

        #add networks
        mtl_nw.add_network(terminal_1)
        mtl_nw.add_network(iconn)
        mtl_nw.add_network(terminal_3)


        mtl_nw.run_until(finalTime)

        t, V0 = np.genfromtxt('python/testData/qucs/MTLN_1_conductor_not_adapted_network_R'+str(R)+'.csv', delimiter=';', names = True, usecols=(0,1), unpack = True)

        plt.plot(1e9*v_probe.t, 1e3*v_probe.val, label = "MTLN")
        plt.plot(1e9*t, 1e3*V0,'--', label = "QUCS")
        plt.title("Two 1-cond.lines, R = "+str(R))
        plt.ylabel(r'$V_1 (0, t)\,[mV]$')
        plt.xlabel(r'$t\,[ns]$')
        plt.xticks(range(0, 200, 50))
        plt.grid('both')
        plt.legend()
        plt.savefig("MTLN_1_conductor_not_adapted_network_R"+str(R)+".png")
        plt.clf()
        
def test_1_conductor_network_RV():
    """
     _             _ ____             _
    | |     b0    |      |     b1    | |
    | 0-----------1-R--V-2-----------3 |
    |_|     0     |______|     0     |_|
    term_1(0)     iconn(1)     term_2(2)
    
    """
    for R in [25,50,100,150]:

        bundle_0 = mtl.MTL(l=0.25e-6, c=100e-12, length=1.0, nx=50)
        bundle_1 = mtl.MTL(l=0.25e-6, c=100e-12, length=1.0, nx=50)
        finalTime = 200e-9

        def magnitude(t): return wf.trapezoidal_wave(
            t, A=1, rise_time=1e-9, fall_time=1e-9, f0=1e6, D=0.5)
        

        v_probe = bundle_0.add_probe(position=0.0, type='voltage')

        mtl_nw = mtln.MTLN()
        mtl_nw.add_bundle(0, bundle_0)
        mtl_nw.add_bundle(1, bundle_1)

        #network definition
        terminal_1 = mtln.Network(nw_number = 0, nodes = [0], bundles = [0])
        bundle_connections= [{"node" : 0, "conductor" : 0}]
        
        terminal_1.add_nodes_in_bundle(bundle_number = 0, 
                                    bundle = bundle_0,
                                    connections = bundle_connections, 
                                    side= "S")
        #network connections
        terminal_1.connect_to_ground(node = 0, R= 50, Vt = magnitude, side = "S")

        #interconnection network
        iconn = mtln.Network(nw_number=1, nodes = [1,2], bundles = [0,1])
        bundle_0_connections = [{"node" : 1, "conductor" : 0}]
        bundle_1_connections = [{"node" : 2, "conductor" : 0}]
        iconn.add_nodes_in_bundle(bundle_number = 0, 
                                    bundle = bundle_0, 
                                    connections= bundle_0_connections, 
                                    side= "L")
        iconn.add_nodes_in_bundle(bundle_number = 1, 
                                    bundle = bundle_1, 
                                    connections= bundle_1_connections, 
                                    side= "S")
        iconn.connect_nodes(2,1, R = R, Vt = magnitude)

        #network definition
        terminal_3 = mtln.Network(nw_number = 2 ,nodes = [3], bundles = [1])
        bundle_connections= [{"node" : 3, "conductor" : 0}]
        terminal_3.add_nodes_in_bundle(bundle_number = 1, 
                                    bundle = bundle_1, 
                                    connections= bundle_connections, 
                                    side= "L")

        #network connections
        terminal_3.connect_to_ground(3, 50, side = "L")

        #add networks
        mtl_nw.add_network(terminal_1)
        mtl_nw.add_network(iconn)
        mtl_nw.add_network(terminal_3)


        mtl_nw.run_until(finalTime)

        t, V0 = np.genfromtxt('python/testData/qucs/MTLN_1_conductor_network_R'+str(R)+'_V.csv', delimiter=';', names = True, usecols=(0,1), unpack = True)

        plt.plot(1e9*v_probe.t, 1e3*v_probe.val, label = "MTLN")
        plt.plot(1e9*t, 1e3*V0,'--', label = "QUCS")
        plt.title("Two 1-cond.lines, R = "+str(R))
        plt.ylabel(r'$V_1 (0, t)\,[mV]$')
        plt.xlabel(r'$t\,[ns]$')
        plt.xticks(range(0, 200, 50))
        plt.grid('both')
        plt.legend()
        plt.show()

def test_1_conductor_not_adapted_network_RV():
    """
     _             _ ____             _
    | |     b0    |      |     b1    | |
    | 0-----------1-R--V-2-----------3 |
    |_|     0     |______|     0     |_|
    term_1(0)     iconn(1)     term_2(2)
    
    """
    for R in [25,50,100,150]:

        bundle_0 = mtl.MTL(l=0.25e-6*1.125, c=100e-12/2, length=1.0, nx=50)
        bundle_1 = mtl.MTL(l=0.25e-6*2, c=100e-12/2, length=1.0, nx=50)
        finalTime = 200e-9

        def magnitude(t): return wf.trapezoidal_wave(
            t, A=1, rise_time=1e-9, fall_time=1e-9, f0=1e6, D=0.5)
        def V35(t): return wf.trapezoidal_wave(
            t, A=1, rise_time=15e-9, fall_time=5e-9, f0=1e02, D=9.5e-6)


        v_probe = bundle_0.add_probe(position=0.0, type='voltage')

        mtl_nw = mtln.MTLN()
        mtl_nw.add_bundle(0, bundle_0)
        mtl_nw.add_bundle(1, bundle_1)

        #network definition
        terminal_1 = mtln.Network(nw_number = 0, nodes = [0], bundles = [0])
        bundle_connections= [{"node" : 0, "conductor" : 0}]
        
        terminal_1.add_nodes_in_bundle(bundle_number = 0, 
                                    bundle = bundle_0,
                                    connections = bundle_connections, 
                                    side= "S")
        #network connections
        terminal_1.connect_to_ground(node = 0, R= 50, Vt = magnitude, side = "S")

        #interconnection network
        iconn = mtln.Network(nw_number=1, nodes = [1,2], bundles = [0,1])
        bundle_0_connections = [{"node" : 1, "conductor" : 0}]
        bundle_1_connections = [{"node" : 2, "conductor" : 0}]
        iconn.add_nodes_in_bundle(bundle_number = 0, 
                                    bundle = bundle_0, 
                                    connections= bundle_0_connections, 
                                    side= "L")
        iconn.add_nodes_in_bundle(bundle_number = 1, 
                                    bundle = bundle_1, 
                                    connections= bundle_1_connections, 
                                    side= "S")
        iconn.connect_nodes(2,1, R = R, Vt = V35)

        #network definition
        terminal_3 = mtln.Network(nw_number = 2 ,nodes = [3], bundles = [1])
        bundle_connections= [{"node" : 3, "conductor" : 0}]
        terminal_3.add_nodes_in_bundle(bundle_number = 1, 
                                    bundle = bundle_1, 
                                    connections= bundle_connections, 
                                    side= "L")

        #network connections
        terminal_3.connect_to_ground(3, 50, side = "L")

        #add networks
        mtl_nw.add_network(terminal_1)
        mtl_nw.add_network(iconn)
        mtl_nw.add_network(terminal_3)


        mtl_nw.run_until(finalTime)

        t, V0 = np.genfromtxt('python/testData/qucs/MTLN_1_conductor_not_adapted_network_R'+str(R)+'_V35.csv', delimiter=';', names = True, usecols=(0,1), unpack = True)
        plt.plot(1e9*v_probe.t, 1e3*v_probe.val, label = "MTLN")
        plt.plot(1e9*t, 1e3*V0,'--', label = "QUCS")
        plt.title("Two 1-cond.lines, R = "+str(R))
        plt.ylabel(r'$V_1 (0, t)\,[mV]$')
        plt.xlabel(r'$t\,[ns]$')
        plt.xticks(range(0, 200, 50))
        plt.grid('both')
        plt.legend()
        # plt.savefig("MTLN_1_conductor_not_adapted_network_R"+str(R)+"_V35.png")
        plt.show()
        plt.clf()
        
def test_1_conductor_network_RV():
    """
     _             _ ____             _
    | |     b0    |      |     b1    | |
    | 0-----------1-R--V-2-----------3 |
    |_|     0     |______|     0     |_|
    term_1(0)     iconn(1)     term_2(2)
    
    """
    for R in [25,50,100,150]:

        bundle_0 = mtl.MTL(l=0.25e-6, c=100e-12, length=1.0, nx=50)
        bundle_1 = mtl.MTL(l=0.25e-6, c=100e-12, length=1.0, nx=50)
        finalTime = 200e-9

        def magnitude(t): return wf.trapezoidal_wave(
            t, A=1, rise_time=1e-9, fall_time=1e-9, f0=1e6, D=0.5)
        def V35(t): return wf.trapezoidal_wave(
            t, A=1, rise_time=15e-9, fall_time=5e-9, f0=1e02, D=9.5e-6)


        v_probe = bundle_0.add_probe(position=0.0, type='voltage')

        mtl_nw = mtln.MTLN()
        mtl_nw.add_bundle(0, bundle_0)
        mtl_nw.add_bundle(1, bundle_1)

        #network definition
        terminal_1 = mtln.Network(nw_number = 0, nodes = [0], bundles = [0])
        bundle_connections= [{"node" : 0, "conductor" : 0}]
        
        terminal_1.add_nodes_in_bundle(bundle_number = 0, 
                                    bundle = bundle_0,
                                    connections = bundle_connections, 
                                    side= "S")
        #network connections
        terminal_1.connect_to_ground(node = 0, R= 50, Vt = magnitude, side = "S")

        #interconnection network
        iconn = mtln.Network(nw_number=1, nodes = [1,2], bundles = [0,1])
        bundle_0_connections = [{"node" : 1, "conductor" : 0}]
        bundle_1_connections = [{"node" : 2, "conductor" : 0}]
        iconn.add_nodes_in_bundle(bundle_number = 0, 
                                    bundle = bundle_0, 
                                    connections= bundle_0_connections, 
                                    side= "L")
        iconn.add_nodes_in_bundle(bundle_number = 1, 
                                    bundle = bundle_1, 
                                    connections= bundle_1_connections, 
                                    side= "S")
        iconn.connect_nodes(2,1, R = R, Vt = V35)

        #network definition
        terminal_3 = mtln.Network(nw_number = 2 ,nodes = [3], bundles = [1])
        bundle_connections= [{"node" : 3, "conductor" : 0}]
        terminal_3.add_nodes_in_bundle(bundle_number = 1, 
                                    bundle = bundle_1, 
                                    connections= bundle_connections, 
                                    side= "L")

        #network connections
        terminal_3.connect_to_ground(3, 50, side = "L")

        #add networks
        mtl_nw.add_network(terminal_1)
        mtl_nw.add_network(iconn)
        mtl_nw.add_network(terminal_3)


        mtl_nw.run_until(finalTime)

        t, V0 = np.genfromtxt('python/testData/qucs/MTLN_1_conductor_network_R'+str(R)+'_V35.csv', delimiter=';', names = True, usecols=(0,1), unpack = True)
        plt.plot(1e9*v_probe.t, 1e3*v_probe.val, label = "MTLN")
        plt.plot(1e9*t, 1e3*V0,'--', label = "QUCS")
        plt.title("Two 1-cond.lines, R = "+str(R))
        plt.ylabel(r'$V_1 (0, t)\,[mV]$')
        plt.xlabel(r'$t\,[ns]$')
        plt.xticks(range(0, 200, 50))
        plt.grid('both')
        plt.legend()
        plt.show()
        # plt.savefig("MTLN_1_conductor_network_R"+str(R)+"_V35.png")
        plt.clf()