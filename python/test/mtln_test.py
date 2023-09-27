from src.jsonParser import Parser

import numpy as np
import matplotlib.pyplot as plt

import sympy as sp
from scipy.constants import epsilon_0, mu_0, speed_of_light

import src.mtl as mtl
import src.mtln as mtln
import src.networks as nw

import src.waveforms as wf

import unittest as ut

import pickle
import dill

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


    finalTime = 200e-9

    def magnitude(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=20e-9, fall_time=20e-9, f0=1e6, D=0.5)


    """
     _             _
    | |     1     | |
    | 2-----------4 |
    | |     0     | |
    | 1-----------3 |
    |_|           |_|
    term_1(0)     term_2(1)
    
    """
    line = mtl.MTL(l=l, c=c, length=2.0, ndiv=2, name = "line_0")
    bundle = mtl.MTLD({0 : [line]}, name = "bundle")
    v_probe = bundle.add_probe(position=0.0, probe_type='voltage')

    t1  = {line: {"connections" : [[0,0], [1,1]], "side": "S", "bundle" : bundle }}
    t2  = {line: {"connections" : [[2,0], [3,1]], "side": "L", "bundle" : bundle }}

    terminal_1 = nw.Network(t1)
    terminal_1.connect_to_ground(node = 0, R  = 50)
    terminal_1.connect_to_ground(node = 1, R= 50, Vt = magnitude)
    terminal_L = nw.NetworkD({0 : terminal_1})

    terminal_2 = nw.Network(t2)
    terminal_2.connect_to_ground(2, 50)
    terminal_2.connect_to_ground(3, 50)
    terminal_R = nw.NetworkD({0 : terminal_2})

    mtl_nw = mtln.MTLN()
    mtl_nw.add_bundle(bundle)
    mtl_nw.add_network(terminal_L)
    mtl_nw.add_network(terminal_R)

    # ####    
    # file = 'python/testData/parser/ribbon_cable_20ns_termination_network.smb.json'
    # p = Parser(file)
    # ####

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


    """
     _             _             _
    | |     1     | |     1     | |
    | 1-----------3-5-----------7 |
    | |     b0    | |     b1    | |
    | 0-----------2-4-----------6 |
    |_|     0     |_|     0     |_|
    term_1(0)     iconn(1)     term_2(2)
    
    """

    finalTime = 200e-9

    nodes_0 = np.array([[0.0,0.0,0.0],[0.0,0.0,1.0]])
    nodes_1 = np.array([[0.0,0.0,1.0],[0.0,0.0,2.0]])

    line_0 = mtl.MTL(l=l, c=c, node_positions=nodes_0, ndiv=50, name = "line_0")
    line_1 = mtl.MTL(l=l, c=c, node_positions=nodes_1, ndiv=50, name = "line_1")
    # line_0 = mtl.MTL(l=l, c=c, length=1.0, ndiv=50, name = "line_0")
    # line_1 = mtl.MTL(l=l, c=c, length=1.0, ndiv=50, name = "line_1")

    bundle_0 = mtl.MTLD(levels = {0:[line_0]}, name = "bundle_0")
    bundle_1 = mtl.MTLD(levels = {0:[line_1]}, name = "bundle_1")


    def magnitude(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=1e-9, fall_time=1e-9, f0=1e6, D=0.5)
    
    term_1  = {line_0: {"bundle" : bundle_0, "connections" : [[0,0], [1,1]], "side": "S" }}
    
    iconn   = {line_1: {"bundle" : bundle_1, "connections" : [[5,1], [4,0]], "side": "S" },
               line_0: {"bundle" : bundle_0, "connections" : [[3,1], [2,0]], "side": "L" }}
               
    term_2  = {line_1: {"bundle" : bundle_1, "connections" : [[6,0], [7,1]], "side": "L" }}
    
    v_probe = bundle_0.add_probe(position=0.0, probe_type='voltage')

    terminal_1 = nw.Network(term_1)
    terminal_1.connect_to_ground(node = 0, R  = 50)
    terminal_1.connect_to_ground(node = 1, R= 50, Vt = magnitude)
    terminal_1D = nw.NetworkD({0:terminal_1})

    junction = nw.Network(iconn)
    junction.short_nodes(4,2)
    junction.short_nodes(5,3)
    junctionD = nw.NetworkD({0:junction})

    terminal_2 = nw.Network(term_2)
    terminal_2.connect_to_ground(6, 50)
    terminal_2.connect_to_ground(7, 50)
    terminal_2D = nw.NetworkD({0:terminal_2})
    
    mtl_nw = mtln.MTLN()
    mtl_nw.add_bundle(bundle_0)
    mtl_nw.add_bundle(bundle_1)

    mtl_nw.add_network(terminal_1D)
    mtl_nw.add_network(junctionD)
    mtl_nw.add_network(terminal_2D)

    # # ####
    # file = 'python/testData/parser/ribbon_cable_1ns_paul_interconnection_network.smb.json'
    # p = Parser(file)
    # p.run(finalTime = 200e-9)
    # # ####

    f = open("mtl_dump.pkl",'wb')
    f.write(dill.dumps(mtl_nw))
    f.close()

    mtl_nw.run_until(finalTime)


    plt.plot(1e9*v_probe.t, 1e3*v_probe.val, 'r', label ="mtl_nw")
    # plt.plot(1e9*p.probes["v0"].t, 1e3*p.probes["v0"].val,'b', label='parse')
    plt.ylabel(r'$V_1 (0, t)\,[mV]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 200, 50))
    plt.grid('both')
    plt.legend()
    plt.show()

    times = [12.5, 25, 40, 55]
    voltages = [120, 95, 55, 32]
    for (t, v) in zip(times, voltages):
        index = np.argmin(np.abs(v_probe.t - t*1e-9))
        assert np.all(np.isclose(v_probe.val[index, 0], v*1e-3, atol=10e-3))


    
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



    """
     _             __________             _
    | |     1     |          |     1     | |
    | 1-----------3--R-------5-----------7 |
    | |     b0    |          |     b1    | |
    | 0-----------2--R-------4-----------6 |
    |_|     0     |__________|     0     |_|
    term_1(0)     iconn(1)     term_2(2)
    
    """

    finalTime = 200e-9

    line_0 = mtl.MTL(l=l, c=c, length=1.0, ndiv=50, name = "line_0")
    line_1 = mtl.MTL(l=l, c=c, length=1.0, ndiv=50, name = "line_1")
    bundle_0 = mtl.MTLD({0:[line_0]}, name = "bundle_0")
    bundle_1 = mtl.MTLD({0:[line_1]}, name = "bundle_1")

    v_probe = bundle_0.add_probe(position=0.0, probe_type='voltage')


    def magnitude(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=1e-9, fall_time=1e-9, f0=1e6, D=0.5)
    

    term_1  = {line_0: {"connections" : [[0,0], [1,1]], "side": "S", "bundle" : bundle_0 }}
    
    iconn   = {line_0: {"connections" : [[2,0], [3,1]], "side": "L", "bundle" : bundle_0 },
               line_1: {"connections" : [[4,0], [5,1]], "side": "S", "bundle" : bundle_1 }}

    term_2  = {line_1: {"connections" : [[6,0], [7,1]], "side": "L", "bundle" : bundle_1 }}


    mtl_nw = mtln.MTLN()
    mtl_nw.add_bundle(bundle_0)
    mtl_nw.add_bundle(bundle_1)

    #network definition
    terminal_1 = nw.Network(term_1)
    terminal_1.connect_to_ground(node = 0, R  = 50)
    terminal_1.connect_to_ground(node = 1, R= 50, Vt = magnitude)

    #interconnection network
    junction = nw.Network(iconn)
    junction.connect_nodes(5,3, R = 10)
    junction.connect_nodes(4,2, R = 25)

    #network definition
    terminal_2 = nw.Network(term_2)
    terminal_2.connect_to_ground(6, 50)
    terminal_2.connect_to_ground(7, 50)
    
    mtl_nw.add_network(nw.NetworkD({0:terminal_1}))
    mtl_nw.add_network(nw.NetworkD({0:junction}))
    mtl_nw.add_network(nw.NetworkD({0:terminal_2}))

    mtl_nw.run_until(finalTime)

    t0, V0 = np.genfromtxt('python/testData/ngspice/test_ribbon_cable_1ns_R_interconnection_network/V1.txt', delimiter=',', usecols=(0,1), unpack = True)
    t1, V1 = np.genfromtxt('python/testData/ngspice/test_ribbon_cable_1ns_R_interconnection_network/V2.txt', delimiter=',', usecols=(0,1), unpack = True)

    V0_resampled = np.interp(v_probe.t, t0, V0)
    V1_resampled = np.interp(v_probe.t, t1, V1)

    assert(np.allclose(V0_resampled[:-1], v_probe.val[1:,0], atol = 0.01, rtol=0.05))
    assert(np.allclose(V1_resampled[:-1], v_probe.val[1:,1], atol = 0.01, rtol=0.05))
    
    plt.plot(1e9*v_probe.t, v_probe.val[:,0] ,'r', label = 'Conductor 0')
    plt.plot(1e9*v_probe.t, v_probe.val[:,1] ,'b', label = 'Conductor 1')
    plt.plot(1e9*v_probe.t, V0_resampled ,'g--', label = 'Conductor 0 - NgSpice')
    plt.plot(1e9*v_probe.t, V1_resampled ,'k--', label = 'Conductor 1 - NgSpice')
    plt.ylabel(r'$V (0, t)\,[V]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 200, 50))
    plt.grid('both')
    plt.legend()
    plt.show()
    # plt.savefig("python/testData/output/test_ribbon_cable_1ns_R_interconnection_network/test_ribbon_cable_1ns_R_interconnection_network.png")

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


    line_0 = mtl.MTL(l=l, c=c, length=1.0, ndiv=50, name= "line_0")
    line_1 = mtl.MTL(l=l, c=c, length=1.0, ndiv=50, name= "line_1")
    bundle_0 = mtl.MTLD({0 :[line_0]}, name = "bundle_0")
    bundle_1 = mtl.MTLD({0 :[line_1]}, name = "bundle_1")
    finalTime = 200e-9

    def magnitude(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=1e-9, fall_time=1e-9, f0=1e6, D=0.5)
    
    def V35(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=15e-9, fall_time=5e-9, f0=1e02, D=9.5e-6)

    v_probe = bundle_0.add_probe(position=0.0, probe_type='voltage')

    mtl_nw = mtln.MTLN()
    mtl_nw.add_bundle(bundle_0)
    mtl_nw.add_bundle(bundle_1)

    term_1  = {line_0: {"connections" : [[0,0], [1,1]], "side": "S", "bundle" : bundle_0 }}
    
    iconn   = {line_0: {"connections" : [[2,0], [3,1]], "side": "L", "bundle" : bundle_0 },
               line_1: {"connections" : [[4,0], [5,1]], "side": "S", "bundle" : bundle_1 }}

    term_2  = {line_1: {"connections" : [[6,0], [7,1]], "side": "L", "bundle" : bundle_1 }}



    #network definition
    terminal_1 = nw.Network(term_1)
    terminal_1.connect_to_ground(node = 0, R  = 50)
    terminal_1.connect_to_ground(node = 1, R= 50, Vt = magnitude)

    #interconnection network
    junction = nw.Network(iconn)
    junction.connect_nodes(5,3, R = 10, Vt = V35)
    junction.connect_nodes(4,2, R = 25)

    #network definition
    terminal_2 = nw.Network(term_2)
    terminal_2.connect_to_ground(6, 50)
    terminal_2.connect_to_ground(7, 50)

    mtl_nw.add_network(nw.NetworkD({0:terminal_1}))
    mtl_nw.add_network(nw.NetworkD({0:junction}))
    mtl_nw.add_network(nw.NetworkD({0:terminal_2}))

    mtl_nw.run_until(finalTime)

    t0, V0 = np.genfromtxt('python/testData/ngspice/test_ribbon_cable_1ns_RV_interconnection_network/V1.txt', delimiter=',', usecols=(0,1), unpack = True)
    t1, V1 = np.genfromtxt('python/testData/ngspice/test_ribbon_cable_1ns_RV_interconnection_network/V2.txt', delimiter=',', usecols=(0,1), unpack = True)

    V0_resampled = np.interp(v_probe.t, t0, V0)
    V1_resampled = np.interp(v_probe.t, t1, V1)


    plt.plot(1e9*v_probe.t, v_probe.val[:,0] ,'r', label = 'Conductor 0')
    plt.plot(1e9*v_probe.t, v_probe.val[:,1] ,'b', label = 'Conductor 1')
    plt.plot(1e9*t0, V0 ,'g--', label = 'Conductor 0 - NgSpice')
    plt.plot(1e9*t1, V1 ,'k--', label = 'Conductor 1 - NgSpice')
    plt.ylabel(r'$V (0, t)\,[V]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 200, 50))
    plt.grid('both')
    plt.legend()
    plt.show()

    assert(np.allclose(V0_resampled[:-1], v_probe.val[1:,0], atol = 0.01, rtol=0.05))
    assert(np.allclose(V1_resampled[:-1], v_probe.val[1:,1], atol = 0.01, rtol=0.05))

    # plt.savefig("python/testData/output/test_ribbon_cable_1ns_RV_interconnection_network/test_ribbon_cable_1ns_RV_interconnection_network.png")
def test_ribbon_cable_1ns_RV_interconnection_network_2():
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

    finalTime = 200e-9

    """
     _             __________             _
    | |     1     |          |     1     | |
    | 1-----------3--R--V35--5-----------7 |
    | |     b0    |          |     b1    | |
    | 0-----------2--R-------4-----------6 |
    |_|     0     |__________|     0     |_|
    term_1(0)     iconn(1)     term_2(2)
    
    """


    line_0 = mtl.MTL(l=l, c=c, length=1.0, ndiv=50, name= "line_0")
    line_1 = mtl.MTL(l=l, c=c, length=1.0, ndiv=50, name= "line_1")

    bundle_0 = mtl.MTLD(levels = {0:[line_0]}, name = "bundle_0")
    bundle_1 = mtl.MTLD(levels = {0:[line_1]}, name = "bundle_1")

    def magnitude(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=1e-9, fall_time=1e-9, f0=1e6, D=0.5)
    
    def V35(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=15e-9, fall_time=5e-9, f0=1e02, D=9.5e-6)

    v_probe = bundle_0.add_probe(position=0.0, probe_type='voltage')

    mtl_nw = mtln.MTLN()
    mtl_nw.add_bundle(bundle_0)
    mtl_nw.add_bundle(bundle_1)

    term_1  = {line_0 :  {"bundle" : bundle_0 , "connections" : [[0,0], [1,1]], "side": "S" }}

    iconn   = { line_0 : {"bundle" : bundle_0, "connections" : [[2,0], [3,1]], "side": "L"},
                line_1 : {"bundle" : bundle_1, "connections" : [[4,0], [5,1]], "side": "S"} }

    term_2  = {line_1:  {"bundle" : bundle_1, "connections" : [[6,0], [7,1]], "side": "L" }}
                 



    #network definition
    terminal_1 = nw.Network(term_1)
    terminal_1.connect_to_ground(node = 0, R  = 50)
    terminal_1.connect_to_ground(node = 1, R= 50, Vt = magnitude)
    terminal_1D = nw.NetworkD({0:terminal_1})
    #interconnection network
    junction = nw.Network(iconn)
    junction.connect_nodes(5,3, R = 10, Vt = V35)
    junction.connect_nodes(4,2, R = 25)
    junctionD = nw.NetworkD({0:junction})

    #network definition
    terminal_2 = nw.Network(term_2)
    terminal_2.connect_to_ground(6, 50)
    terminal_2.connect_to_ground(7, 50)
    terminal_2D = nw.NetworkD({0:terminal_2})

    mtl_nw.add_network(terminal_1D)
    mtl_nw.add_network(junctionD)
    mtl_nw.add_network(terminal_2D)

    mtl_nw.run_until(finalTime)

    t0, V0 = np.genfromtxt('python/testData/ngspice/test_ribbon_cable_1ns_RV_interconnection_network/V1.txt', delimiter=',', usecols=(0,1), unpack = True)
    t1, V1 = np.genfromtxt('python/testData/ngspice/test_ribbon_cable_1ns_RV_interconnection_network/V2.txt', delimiter=',', usecols=(0,1), unpack = True)

    V0_resampled = np.interp(v_probe.t, t0, V0)
    V1_resampled = np.interp(v_probe.t, t1, V1)

    assert(np.allclose(V0_resampled[:-1], v_probe.val[1:,0], atol = 0.01, rtol=0.05))
    assert(np.allclose(V1_resampled[:-1], v_probe.val[1:,1], atol = 0.01, rtol=0.05))

    # plt.plot(1e9*v_probe.t, v_probe.val[:,0] ,'r', label = 'Conductor 0')
    # plt.plot(1e9*v_probe.t, v_probe.val[:,1] ,'b', label = 'Conductor 1')
    # plt.plot(1e9*t0, V0 ,'g--', label = 'Conductor 0 - NgSpice')
    # plt.plot(1e9*t1, V1 ,'k--', label = 'Conductor 1 - NgSpice')
    # plt.ylabel(r'$V (0, t)\,[V]$')
    # plt.xlabel(r'$t\,[ns]$')
    # plt.xticks(range(0, 200, 50))
    # plt.grid('both')
    # plt.legend()
    # plt.show()
    # plt.savefig("python/testData/output/test_ribbon_cable_1ns_RV_interconnection_network/test_ribbon_cable_1ns_RV_interconnection_network.png")

def test_ribbon_cable_1ns_RV_T_network_alternative_creator():
    l = np.zeros([2, 2])
    l[0] = [0.7485*1e-6, 0.5077*1e-6]
    l[1] = [0.5077*1e-6, 1.0154*1e-6]
    c = np.zeros([2, 2])
    c[0] = [37.432*1e-12, -18.716*1e-12]
    c[1] = [-18.716*1e-12, 24.982*1e-12]

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

    bundle_0 = mtl.MTL(l=l, c=c, length=1.0, ndiv=50, name = "bundle_0")
    bundle_1 = mtl.MTL(l=l, c=c, length=1.0, ndiv=50, name = "bundle_1")
    bundle_2 = mtl.MTL(l=l, c=c, length=1.0, ndiv=50, name = "bundle_2")
    finalTime = 200e-9

    def magnitude(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=1e-9, fall_time=1e-9, f0=1e6, D=0.5)
    
    def V35(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=15e-9, fall_time=5e-9, f0=1e02, D=9.5e-6)

    v_probe = bundle_0.add_probe(position=0.0, probe_type='voltage')

    manager = mtln.MTLN()
    manager.add_bundle(bundle_0)
    manager.add_bundle(bundle_1)
    manager.add_bundle(bundle_2)



    t1  = {bundle_0: {"connections" : [[0,0], [1,1]], "side": "S" }}
    t3  = {bundle_1: {"connections" : [[6,0], [7,1]], "side": "L" }}
    t4  = {bundle_2: {"connections" : [[10,0], [11,1]], "side": "L"}}
    
    i1 = {bundle_0: {"connections" : [[2,0], [3,1]], "side": "L" },
          bundle_1: {"connections" : [[4,0], [5,1]], "side": "S" },
          bundle_2: {"connections" : [[8,0], [9,1]], "side": "S" }}


    terminal_1 = nw.Network(t1)
    terminal_1.connect_to_ground(node = 0, R = R0)
    terminal_1.connect_to_ground(node = 1, R = R0, Vt = magnitude)

    iconn = nw.Network(i1)
    iconn.connect_nodes(5,3, R = R1, Vt = V35)
    iconn.connect_nodes(4,9, R = R3)
    iconn.connect_nodes(8,2, R = R2)

    terminal_3 = nw.Network(t3)
    terminal_3.connect_to_ground(6, R = R0)
    terminal_3.connect_to_ground(7, R = R0)

    terminal_4 = nw.Network(t4)
    terminal_4.connect_to_ground(10, R0)
    terminal_4.connect_to_ground(11, R0, Vt = V35)

    manager.add_network(terminal_1)
    manager.add_network(iconn)
    manager.add_network(terminal_3)
    manager.add_network(terminal_4)

    manager.run_until(finalTime)

    t0, V0 = np.genfromtxt('python/testData/ngspice/test_ribbon_cable_1ns_RV_T_network/V1.txt', usecols=(0,1), unpack = True)
    t1, V1 = np.genfromtxt('python/testData/ngspice/test_ribbon_cable_1ns_RV_T_network/V2.txt', usecols=(0,1), unpack = True)

    V0_resampled = np.interp(v_probe.t, t0, V0)
    V1_resampled = np.interp(v_probe.t, t1, V1)

    assert(np.allclose(V0_resampled[:-1], v_probe.val[1:,0], atol = 0.01, rtol=0.05))
    assert(np.allclose(V1_resampled[:-1], v_probe.val[1:,1], atol = 0.01, rtol=0.05))

    # plt.plot(1e9*v_probe.t, v_probe.val[:,0] ,'r', label = 'Conductor 0')
    # plt.plot(1e9*v_probe.t, v_probe.val[:,1] ,'b', label = 'Conductor 1')
    # plt.plot(1e9*t0, V0 ,'g--', label = 'Conductor 0 - NgSpice')
    # plt.plot(1e9*t1, V1 ,'k--', label = 'Conductor 1 - NgSpice')
    # plt.ylabel(r'$V (0, t)\,[V]$')
    # plt.xlabel(r'$t\,[ns]$')
    # plt.xticks(range(0, 200, 50))
    # plt.grid('both')
    # plt.legend()
    # plt.show()
#    # plt.savefig("python/testData/output/test_ribbon_cable_1ns_RV_T_network/test_ribbon_cable_1ns_RV_T_network.png")

    
def test_1_conductor_network_Z50():


    def magnitude(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=1e-9, fall_time=1e-9, f0=1e6, D=0.5)

    """
     _             _
    | |           | |
    | |  0: Z=50  | |
    | 0-----------1 |
    |_|           |_|
    term_1(0)     term_2(1)
    
    """
    for l,c,n,Z in [[0.25e-6,100e-12,50,50],[0.25e-8,100e-12,50,5],[0.5e-6,50e-12,100,100]]:
        mtl_nw = mtln.MTLN()
        bundle_0 = mtl.MTL(l=l, c=c, length=1.0, ndiv=n, name = "b0")
        v_probe = bundle_0.add_probe(position=0.0, probe_type='voltage')

        mtl_nw.add_bundle(bundle_0)

        term_1  = {bundle_0: {"connections" : [[0,0]], "side": "S" }}
        term_2  = {bundle_0: {"connections" : [[1,0]], "side": "L" }}


        #network definition
        terminal_1 = nw.Network(term_1)
        terminal_1.connect_to_ground(node = 0, R= 50, Vt = magnitude)
        mtl_nw.add_network(terminal_1)
        
        #network definition
        terminal_2 = nw.Network(term_2)
        terminal_2.connect_to_ground(1, 50)
        mtl_nw.add_network(terminal_2)

        mtl_nw.run_until(200e-9)

        t_sp, V0_sp = np.genfromtxt('python/testData/ngspice/test_1_conductor_network_Z'+str(Z)+'/V2.txt', usecols=(0,1), unpack = True)
        V0_sp_resampled = np.interp(v_probe.t, t_sp, V0_sp)
        assert(np.allclose(V0_sp_resampled[:-1], v_probe.val[1:,0], rtol=0.01))

        # plt.plot(1e9*v_probe.t, v_probe.val, label = "MTLN")
        # plt.plot(1e9*t_sp, V0_sp,'-.', label = "NgSpice")
        # plt.ylabel(r'$V_1 (0, t)\,[mV]$')
        # plt.xlabel(r'$t\,[ns]$')
        # plt.xticks(range(0, 200, 50))
        # plt.grid('both')
        # plt.legend()
        # plt.show()


def test_1_conductor_adapted_network_R():
    """
     _             _             _
    | |     b0    | |     b1    | |
    | 0-----------1-2-----------3 |
    |_|     0     |_|     0     |_|
    term_1(0)     iconn(1)     term_2(2)
    
    """
    for R in [25,50,100,150]:

        bundle_0 = mtl.MTL(l=0.25e-6, c=100e-12, length=1.0, ndiv=50, name = "bundle_0")
        bundle_1 = mtl.MTL(l=0.25e-6, c=100e-12, length=1.0, ndiv=50, name = "bundle_1")
        finalTime = 200e-9

        def magnitude(t): return wf.trapezoidal_wave(
            t, A=1, rise_time=1e-9, fall_time=1e-9, f0=1e6, D=0.5)
        

        v_probe = bundle_0.add_probe(position=0.0, probe_type='voltage')

        mtl_nw = mtln.MTLN()
        mtl_nw.add_bundle(bundle_0)
        mtl_nw.add_bundle(bundle_1)

        term_1  = {bundle_0: {"connections" : [[0,0]], "side": "S" }}
        
        iconn   = {bundle_0: {"connections" : [[1,0]], "side": "L" },
                   bundle_1: {"connections" : [[2,0]], "side": "S" }}
        
        term_2  = {bundle_1: {"connections" : [[3,0]], "side": "L" }}


        terminal_1 = nw.Network(term_1)
        terminal_1.connect_to_ground(node = 0, R= 50, Vt = magnitude)

        junction = nw.Network(iconn)
        junction.connect_nodes(2,1, R = R)

        terminal_2 = nw.Network(term_2)
        terminal_2.connect_to_ground(3, 50)

        #add networks
        mtl_nw.add_network(terminal_1)
        mtl_nw.add_network(junction)
        mtl_nw.add_network(terminal_2)


        mtl_nw.run_until(finalTime)

        t_sp, V0_sp = np.genfromtxt('python/testData/ngspice/test_1_conductor_adapted_network_R/V2_R'+str(R)+'.txt', usecols=(0,1), unpack = True)

        V0_sp_resampled = np.interp(v_probe.t, t_sp, V0_sp)
        assert(np.allclose(V0_sp_resampled[:-1], v_probe.val[1:,0], rtol=0.01))

        # plt.plot(1e9*v_probe.t, 1e3*v_probe.val, label = "MTLN")
        # plt.plot(1e9*t_sp, 1e3*V0_sp,'--', label = "NgSpice")
        # plt.title("Two 1-cond.lines, R = "+str(R))
        # plt.ylabel(r'$V_1 (0, t)\,[mV]$')
        # plt.xlabel(r'$t\,[ns]$')
        # plt.xticks(range(0, 200, 50))
        # plt.grid('both')
        # plt.legend()
        # # plt.show()
        # plt.savefig("python/testData/output/test_1_conductor_adapted_network_R/MTLN_1_conductor_network_R"+str(R)+".png")
        # plt.clf()

def test_1_conductor_not_adapted_network_R():
    """
     _             ____             _
    | |     b0    |    |     b1    | |
    | 0-----------1-R--2-----------3 |
    |_|     0     |____|     0     |_|
    term_1(0)     iconn(1)     term_2(2)
    
    """
    for R in [25,50,100,150]:

        bundle_0 = mtl.MTL(l=0.25e-6*1.5, c=100e-12/1.5, length=1.0, ndiv=50, name = "bundle_0")
        bundle_1 = mtl.MTL(l=0.25e-6*2,     c=100e-12/2, length=1.0, ndiv=50, name = "bundle_1")
        finalTime = 200e-9

        def magnitude(t): return wf.trapezoidal_wave(
            t, A=1, rise_time=1e-9, fall_time=1e-9, f0=1e6, D=0.5)
        

        v_probe = bundle_0.add_probe(position=0.0, probe_type='voltage')

        mtl_nw = mtln.MTLN()
        mtl_nw.add_bundle(bundle_0)
        mtl_nw.add_bundle(bundle_1)

        term_1  = {bundle_0: {"connections" : [[0,0]], "side": "S" }}
        
        iconn   = {bundle_0: {"connections" : [[1,0]], "side": "L" },
                   bundle_1: {"connections" : [[2,0]], "side": "S" }}
        
        term_2  = {bundle_1: {"connections" : [[3,0]], "side": "L" }}

        #network definition
        terminal_1 = nw.Network(term_1)
        terminal_1.connect_to_ground(node = 0, R= 50, Vt = magnitude)

        #interconnection network
        junction = nw.Network(iconn)
        junction.connect_nodes(2,1, R = R)

        #network definition
        terminal_2 = nw.Network(term_2)
        terminal_2.connect_to_ground(3, 50)

        #add networks
        mtl_nw.add_network(terminal_1)
        mtl_nw.add_network(junction)
        mtl_nw.add_network(terminal_2)


        mtl_nw.run_until(finalTime)

        t_sp, V0_sp = np.genfromtxt('python/testData/ngspice/test_1_conductor_not_adapted_network_R/V2_R'+str(R)+'.txt', usecols=(0,1), unpack = True)        

        V0_sp_resampled = np.interp(v_probe.t, t_sp, V0_sp)
        assert(np.allclose(V0_sp_resampled[:-1], v_probe.val[1:,0], rtol=0.01))

        # plt.plot(1e9*v_probe.t, 1e3*v_probe.val, label = "MTLN")
        # plt.plot(1e9*t_sp, 1e3*V0_sp,'--', label = "NgSpice TL")
        # # plt.plot(1e9*t_sp_m, 1e3*V0_sp_m,'-.', label = "NgSpice MTLN")
        # plt.title("Two 1-cond.lines, R = "+str(R))
        # plt.ylabel(r'$V_1 (0, t)\,[mV]$')
        # plt.xlabel(r'$t\,[ns]$')
        # plt.xticks(range(0, 200, 50))
        # plt.grid('both')
        # plt.legend()
        # # plt.show()
        # plt.savefig("python/testData/output/test_1_conductor_not_adapted_network_R/MTLN_1_conductor_not_adapted_network_R"+str(R)+".png")
        # plt.clf()
        
def test_1_conductor_network_RV():
    """
     _             _ ____             _
    | |     b0    |      |     b1    | |
    | 0-----------1-R--V-2-----------3 |
    |_|     0     |______|     0     |_|
    term_1(0)     iconn(1)     term_2(2)
    
    """
    for R in [25,50,100,150]:

        bundle_0 = mtl.MTL(l=0.25e-6, c=100e-12, length=1.0, ndiv=50, name = "bundle_0")
        bundle_1 = mtl.MTL(l=0.25e-6, c=100e-12, length=1.0, ndiv=50, name = "bundle_1")
        finalTime = 200e-9

        def magnitude(t): return wf.trapezoidal_wave(
            t, A=1, rise_time=1e-9, fall_time=1e-9, f0=1e6, D=0.5)
        

        v_probe = bundle_0.add_probe(position=0.0, probe_type='voltage')

        mtl_nw = mtln.MTLN()
        mtl_nw.add_bundle(bundle_0)
        mtl_nw.add_bundle(bundle_1)


        term_1  = {bundle_0: {"connections" : [[0,0]], "side": "S" }}
        
        iconn   = {bundle_0: {"connections" : [[1,0]], "side": "L" },
                   bundle_1: {"connections" : [[2,0]], "side": "S" }}
        
        term_2  = {bundle_1: {"connections" : [[3,0]], "side": "L" }}

        terminal_1 = nw.Network(term_1)
        terminal_1.connect_to_ground(node = 0, R= 50, Vt = magnitude)

        junction = nw.Network(iconn)
        junction.connect_nodes(2,1, R = R, Vt = magnitude)

        #network definition
        terminal_2 = nw.Network(term_2)
        terminal_2.connect_to_ground(3, 50)

        #add networks
        mtl_nw.add_network(terminal_1)
        mtl_nw.add_network(junction)
        mtl_nw.add_network(terminal_2)


        mtl_nw.run_until(finalTime)

        t_sp, V0_sp = np.genfromtxt('python/testData/ngspice/test_1_conductor_network_RV/V2_R'+str(R)+'.txt', usecols=(0,1), unpack = True)        

        V0_sp_resampled = np.interp(v_probe.t, t_sp, V0_sp)
        a = np.isclose(V0_sp_resampled[:-1], v_probe.val[1:,0], rtol=0.03)
        assert(np.all(np.isclose(V0_sp_resampled[:-1], v_probe.val[1:,0], rtol=0.05)))

        # plt.plot(1e9*v_probe.t, 1e3*v_probe.val, label = "MTLN")
        # plt.plot(1e9*t_sp, 1e3*V0_sp,'--', label = "NgSpice")
        # plt.title("Two 1-cond.lines, R = "+str(R))
        # plt.ylabel(r'$V_1 (0, t)\,[mV]$')
        # plt.xlabel(r'$t\,[ns]$')
        # plt.xticks(range(0, 200, 50))
        # plt.grid('both')
        # plt.legend()
        # plt.savefig("python/testData/output/test_1_conductor_network_RV/MTLN_1_conductor_network_RV"+str(R)+".png")
        # plt.clf()
        
def test_1_conductor_not_adapted_network_RV():
    """
     _             _ ____             _
    | |     b0    |      |     b1    | |
    | 0-----------1-R--V-2-----------3 |
    |_|     0     |______|     0     |_|
    term_1(0)     iconn(1)     term_2(2)
    
    """
    for R in [25,50,100,150]:

        bundle_0 = mtl.MTL(l=0.25e-6*1.5, c=100e-12/1.5, length=1.0, ndiv=50, name = "bundle_0")
        bundle_1 = mtl.MTL(l=0.25e-6*2,     c=100e-12/2, length=1.0, ndiv=50, name = "bundle_1")
        finalTime = 200e-9

        def magnitude(t): return wf.trapezoidal_wave(
            t, A=1, rise_time=1e-9, fall_time=1e-9, f0=1e6, D=0.5)
        
        def V35(t): return wf.trapezoidal_wave(
            t, A=1, rise_time=15e-9, fall_time=5e-9, f0=1e02, D=9.5e-6)


        v_probe = bundle_0.add_probe(position=0.0, probe_type='voltage')

        mtl_nw = mtln.MTLN()
        mtl_nw.add_bundle(bundle_0)
        mtl_nw.add_bundle(bundle_1)

        term_1  = {bundle_0: {"connections" : [[0,0]], "side": "S" }}
        
        iconn   = {bundle_0: {"connections" : [[1,0]], "side": "L" },
                   bundle_1: {"connections" : [[2,0]], "side": "S" }}
        
        term_2  = {bundle_1: {"connections" : [[3,0]], "side": "L" }}

        terminal_1 = nw.Network(term_1)
        terminal_1.connect_to_ground(node = 0, R= 50, Vt = magnitude)

        junction = nw.Network(iconn)
        junction.connect_nodes(2,1, R = R, Vt = V35)

        #network definition
        terminal_2 = nw.Network(term_2)
        terminal_2.connect_to_ground(3, 50)

        #add networks
        mtl_nw.add_network(terminal_1)
        mtl_nw.add_network(junction)
        mtl_nw.add_network(terminal_2)


        mtl_nw.run_until(finalTime)

        t_sp, V0_sp = np.genfromtxt('python/testData/ngspice/test_1_conductor_not_adapted_network_RV/V2_R'+str(R)+'.txt', usecols=(0,1), unpack = True)        

        V0_sp_resampled = np.interp(v_probe.t, t_sp, V0_sp)
        assert(np.allclose(V0_sp_resampled[:-1], v_probe.val[1:,0], rtol=0.02))


        # plt.plot(1e9*v_probe.t, 1e3*v_probe.val, label = "MTLN")
        # plt.plot(1e9*t_sp, 1e3*V0_sp,'--', label = "NgSpice")
        # plt.title("Two 1-cond.lines, R = "+str(R))
        # plt.ylabel(r'$V_1 (0, t)\,[mV]$')
        # plt.xlabel(r'$t\,[ns]$')
        # plt.xticks(range(0, 200, 50))
        # plt.grid('both')
        # plt.legend()
        # plt.savefig("python/testData/output/test_1_conductor_not_adapted_network_RV/MTLN_1_conductor_adapted_network_RV"+str(R)+"_V35.png")
        # plt.clf()
        
def test_coaxial_line_paul_8_6_square():
    """ 
    Described in Ch. 8 of Paul Clayton, 
    Analysis of Multiconductor Transmission Lines. 2004. 
    """
    finalTime = 18e-6
    
    manager = mtln.MTLN()
    line = mtl.MTL(l=0.25e-6, c=100e-12, length=400.0, ndiv=100, name="line")
    v_probe = line.add_probe(position=0.0, probe_type='voltage')
    i_probe = line.add_probe(position=400.0, probe_type='current')
    manager.add_bundle(line)
    
    def magnitude(t): return wf.square_pulse(t, 100, 6e-6)
    
    term_S  = {line: {"connections" : [[0,0]], "side": "S" }}
    term_L  = {line: {"connections" : [[1,0]], "side": "L" }}
    
    terminal_S = nw.Network(term_S)
    terminal_S.connect_to_ground(node = 0, R= 150, Vt = magnitude)
    terminal_L = nw.Network(term_L)
    terminal_L.short_to_ground(node = 1)

    manager.add_networks([terminal_S, terminal_L])
    manager.run_until(finalTime)


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

    # xticks = range(int(np.floor(min(1e6*i_probe.t))),
    #                int(np.ceil(max(1e6*i_probe.t))+1))

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

    finalTime = 18e-6
    
    manager = mtln.MTLN()
    line = mtl.MTL(l=0.25e-6, c=100e-12, length=400.0, ndiv=100, name = "line")
    v_probe = line.add_probe(position=0.0, probe_type='voltage')
    i_probe = line.add_probe(position=400.0, probe_type='current')
    manager.add_bundle(line)
    
    def magnitude(t): return wf.triangle_pulse(t, 100, 6e-6)
    term_S  = {line: {"connections" : [[0,0]], "side": "S" }}
    term_L  = {line: {"connections" : [[1,0]], "side": "L" }}
    
    terminal_S = nw.Network(term_S)
    terminal_S.connect_to_ground(node = 0, R= 150, Vt = magnitude)
    terminal_L = nw.Network(term_L)
    terminal_L.short_to_ground(node = 1)

    manager.add_networks([terminal_S, terminal_L])
    
    manager.run_until(finalTime)

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

    finalTime = 200e-9
    
    manager = mtln.MTLN()
    line = mtl.MTL(l=l, c=c, length=2.0, ndiv=2, name = "line")

    def magnitude(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=20e-9, fall_time=20e-9, f0=1e6, D=0.5)
    # line.add_voltage_source(position=0.0, conductor=1, magnitude=magnitude)
    v_probe = line.add_probe(position=0.0, probe_type='voltage')
    v_probe1 = line.add_probe(position=1.0, probe_type='voltage')
    i_probe1 = line.add_probe(position=1.0, probe_type='current')

    manager.add_bundle(line)

    term_S  = {line: {"connections" : [[0,0], [1,1]], "side": "S" }}
    term_L  = {line: {"connections" : [[2,0], [3,1]], "side": "L" }}
    
    terminal_S = nw.Network(term_S)
    terminal_S.connect_to_ground(node = 0, R= 50)
    terminal_S.connect_to_ground(node = 1, R= 50, Vt = magnitude)
    terminal_L = nw.Network(term_L)
    terminal_L.connect_to_ground(node = 2, R= 50)
    terminal_L.connect_to_ground(node = 3, R= 50)

    manager.add_networks([terminal_S, terminal_L])
    manager.run_until(finalTime)

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

    finalTime = 200e-9
    
    manager = mtln.MTLN()
    line = mtl.MTL(l=l, c=c, length=2.0, ndiv=100, name = "line")

    def magnitude(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=1e-9, fall_time=1e-9, f0=1e6, D=0.5)
    # line.add_voltage_source(position=0.0, conductor=1, magnitude=magnitude)
    v_probe = line.add_probe(position=0.0, probe_type='voltage')
    v_probe1 = line.add_probe(position=1.0, probe_type='voltage')
    i_probe1 = line.add_probe(position=1.0, probe_type='current')

    manager.add_bundle(line)

    term_S  = {line: {"connections" : [[0,0], [1,1]], "side": "S" }}
    term_L  = {line: {"connections" : [[2,0], [3,1]], "side": "L" }}
    
    terminal_S = nw.Network(term_S)
    terminal_S.connect_to_ground(node = 0, R= 50)
    terminal_S.connect_to_ground(node = 1, R= 50, Vt = magnitude)
    terminal_L = nw.Network(term_L)
    terminal_L.connect_to_ground(node = 2, R= 50)
    terminal_L.connect_to_ground(node = 3, R= 50)

    manager.add_networks([terminal_S, terminal_L])
    manager.run_until(finalTime)

    # plt.plot(1e9*v_probe.t, 1e3*v_probe.val)
    # plt.ylabel(r'$V_1 (0, t)\,[mV]$')
    # plt.xlabel(r'$t\,[ns]$')
    # plt.xticks(range(0, 200, 50))
    # plt.grid('both')
    # plt.show()

    times = [12.5, 25, 40, 55]
    voltages = [120, 95, 55, 32]
    for (t, v) in zip(times, voltages):
        index = np.argmin(np.abs(v_probe.t - t*1e-9))
        assert np.all(np.isclose(v_probe.val[index, 0], v*1e-3, atol=10e-3))

def test_wire_over_ground_incident_E_paul_11_3_6_50ns():
    """
    Described in Ch. 11.3.2 "Computed results" of Paul Clayton
    Analysis of Multiconductor Transmission Lines. 2007. 
    Computes the induced voltage at the left end of the line
    when excited by an incident external field with rise time 50 ns
    """

    wire_radius = 0.254e-3
    wire_h = 0.02
    wire_separation = 2.*wire_h
    l = (mu_0/(2*np.pi))*np.arccosh(wire_separation/wire_radius)
    c = 2*np.pi*epsilon_0/np.arccosh(wire_separation/wire_radius)

    nx, finalTime, rise_time, fall_time = 10, 100e-9, 50e-9, 50e-9

    manager = mtln.MTLN()
    line = mtl.MTL(l=l, c=c, length=1.0, ndiv=nx, name = "line")

    A, v, x0 = 1.0, np.max(line.get_phase_velocities()), rise_time
    ex = wf.null()
    ey = wf.null()
    ez = wf.trapezoidal_wave_sp(A=1, rise_time=rise_time, fall_time=fall_time, f0=1e6, D=0.5, v = v)
    
    distances = np.zeros([1, line.u.shape[0], 3])
    distances[:,:,0] = wire_separation
    
    line.add_external_field(mtl.Field(ex,ey,ez), distances)

    v_probe = line.add_probe(position=0.0, probe_type='voltage')
    manager.add_bundle(line)
    
    term_left  = {line: {"connections" : [[0,0]], "side": "S" }}
    term_right  = {line: {"connections" : [[1,0]], "side": "L" }}
    
    left = nw.Network(term_left)
    left.connect_to_ground(node = 0, R= 500)

    right = nw.Network(term_right)
    right.connect_to_ground(node = 1, R= 1000)
    
    manager.add_networks([left, right])

    manager.run_until(finalTime)

    plt.plot(1e9*v_probe.t, 1e3*v_probe.val, label='v probe')
    plt.ylabel(r'$V_1 (0, t)\,[mV]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, int(finalTime*1e9), 5))
    plt.grid('both')
    plt.legend()
    plt.show()

    times = [3.5, 7, 25, 53, 56.6, 59.8, 80]
    voltages = [-1.4, -0.7, -0.8 ,0.45, -0.1315, 0.1, -0.015]
    for (t, v) in zip(times, voltages):
        index = np.argmin(np.abs(v_probe.t - t*1e-9))
        assert np.all(np.isclose(v_probe.val[index, 0]*1e3, v, atol=0.25))



def test_wire_over_ground_incident_E_paul_11_3_6_10ns():
    """
    Described in Ch. 11.3.2 "Computed results" of Paul Clayton
    Analysis of Multiconductor Transmission Lines. 2007. 
    Computes the induced voltage at the left end of the line
    when excited by an incident external field with rise time 10 ns
    """

    wire_radius = 0.254e-3
    wire_h = 0.02
    wire_separation = 2.*wire_h
    l = (mu_0/(2*np.pi))*np.arccosh(wire_separation/wire_radius)
    c = 2*np.pi*epsilon_0/np.arccosh(wire_separation/wire_radius)

    nx, finalTime, rise_time, fall_time = 10, 40e-9, 10e-9, 10e-9

    manager = mtln.MTLN()
    line = mtl.MTL(l=l, c=c, length=1.0, ndiv=nx, name = "line")

    A, v, x0 = 1.0, np.max(line.get_phase_velocities()), rise_time
    ex = wf.null()
    ey = wf.null()
    ez = wf.trapezoidal_wave_sp(A=1, rise_time=rise_time, fall_time=fall_time, f0=1e6, D=0.5, v = v)
    
    distances = np.zeros([1, line.u.shape[0], 3])
    distances[:,:,0] = wire_separation
    
    line.add_external_field(mtl.Field(ex,ey,ez), distances)

    v_probe = line.add_probe(position=0.0, probe_type='voltage')
    manager.add_bundle(line)
    
    term_left  = {line: {"connections" : [[0,0]], "side": "S" }}
    term_right  = {line: {"connections" : [[1,0]], "side": "L" }}
    left = nw.Network(term_left)
    left.connect_to_ground(node = 0, R= 500)
    right = nw.Network(term_right)
    right.connect_to_ground(node = 1, R= 1000)
    

    
    # left = nw.Network([0],[0])
    # conn = [{"node":0, "conductor":0}]
    # left.add_nodes_in_line(0, line, conn, "S")
    # left.connect_to_ground(node = 0, R= 500)

    # right = nw.Network([1],[0])
    # conn = [{"node":1, "conductor":0}]
    # right.add_nodes_in_line(0, line, conn, "L")
    # right.connect_to_ground(node = 1, R= 1000)
    
    manager.add_networks([left, right])

    manager.run_until(finalTime)

    plt.plot(1e9*v_probe.t, 1e3*v_probe.val, label='v probe')
    plt.ylabel(r'$V_1 (0, t)\,[mV]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 45, 5))
    # plt.xticks(range(0, int(finalTime*1e9), 5))
    plt.grid('both')
    plt.legend()
    plt.show()

    times = [3.4, 6.8, 9.9, 16.7, 20, 23.3, 35]
    voltages = [-8.2, -3.8, -4.8, -0.55, 0.52, -0.019, 6e-3]
    for (t, v) in zip(times, voltages):
        index = np.argmin(np.abs(v_probe.t - t*1e-9))
        assert np.all(np.isclose(v_probe.val[index, 0], v*1e-3, atol=1.5e-3))



def test_wire_over_ground_incident_E_paul_11_3_6_1ns():
    """
    Described in Ch. 11.3.2 "Computed results" of Paul Clayton
    Analysis of Multiconductor Transmission Lines. 2007. 
    Computes the induced voltage at the left end of the line
    when excited by an incident external field with rise time 1 ns
    """

    wire_radius = 0.254e-3
    wire_h = 0.02
    wire_separation = 2.*wire_h
    l = (mu_0/(2*np.pi))*np.arccosh(wire_separation/wire_radius)
    c = 2*np.pi*epsilon_0/np.arccosh(wire_separation/wire_radius)

    nx, finalTime, rise_time, fall_time = 50, 30e-9, 1e-9, 1e-9

    manager = mtln.MTLN()
    line = mtl.MTL(l=l, c=c, length=1.0, ndiv=nx, name = "line")

    A, v, x0 = 1.0, np.max(line.get_phase_velocities()), rise_time
    ex = wf.null()
    ey = wf.null()
    ez = wf.trapezoidal_wave_sp(A=1, rise_time=rise_time, fall_time=fall_time, f0=1e6, D=0.5, v = v)
    
    distances = np.zeros([1, line.u.shape[0], 3])
    distances[:,:,0] = wire_separation
    
    line.add_external_field(mtl.Field(ex,ey,ez), distances)

    v_probe = line.add_probe(position=0.0, probe_type='voltage')
    manager.add_bundle(line)
    
    # left = nw.Network([0],[0])
    # conn = [{"node":0, "conductor":0}]
    # left.add_nodes_in_line(0, line, conn, "S")
    # left.connect_to_ground(node = 0, R= 500)

    # right = nw.Network([1],[0])
    # conn = [{"node":1, "conductor":0}]
    # right.add_nodes_in_line(0, line, conn, "L")
    # right.connect_to_ground(node = 1, R= 1000)

    term_left  = {line: {"connections" : [[0,0]], "side": "S" }}
    term_right  = {line: {"connections" : [[1,0]], "side": "L" }}
    left = nw.Network(term_left)
    left.connect_to_ground(node = 0, R= 500)
    right = nw.Network(term_right)
    right.connect_to_ground(node = 1, R= 1000)

    manager.add_networks([left, right])

    manager.run_until(finalTime)

    # plt.plot(1e9*v_probe.t, 1e3*v_probe.val, label='v probe')
    # plt.ylabel(r'$V_1 (0, t)\,[mV]$')
    # plt.xlabel(r'$t\,[ns]$')
    # plt.xticks(range(0, int(finalTime*1e9), 5))
    # plt.grid('both')
    # plt.legend()
    # plt.show()

    times = [3, 5, 8.5, 12, 15, 19, 25]
    voltages = [-24, 12.9, -3.2, 1.5, -0.6, 0.08, -0.38]
    for (t, v) in zip(times, voltages):
        index = np.argmin(np.abs(v_probe.t - t*1e-9))
        assert np.all(np.isclose(v_probe.val[index, 0], v*1e-3, atol=2.5e-3))

    
def test_wire_over_ground_incident_E_transversal_paul_12_4_100ns():

    """
    Described in Ch. 12.4 "Computed results" of Paul Clayton
    Analysis of Multiconductor Transmission Lines. 2007. 
    """
    finalTime = 200e-9
    l = np.zeros([2,2])
    l[0] = [0.7485e-6, 0.2408e-6]
    l[1] = [0.2408e-6, 0.7485e-6]

    c = np.zeros([2,2])
    c[0] = [24.982e-12, -6.266e-12]
    c[1] = [-6.266e-12, 24.982e-12]

    manager = mtln.MTLN()    
    line = mtl.MTL(l=l, c=c, length=2.0, ndiv=5)

    rise_time=100e-9
    wire_separation = 0.00127
    e_y = wf.null()
    e_z = wf.null()
    e_x = wf.ramp_pulse_x_sp(A=1, x0=rise_time)
    
    distances = np.zeros([2, line.u.shape[0], 3])
    distances[0,:,0] = -wire_separation
    distances[1,:,0] = wire_separation
    line.add_external_field(mtl.Field(e_x,e_y,e_z), distances)

    v_probe = line.add_probe(position=0.0, probe_type='voltage')
    
    term_left  = {line: {"connections" : [[0,0], [1,1]], "side": "S" }}
    term_right  = {line: {"connections" : [[2,0], [3,1]], "side": "L" }}
    
    left = nw.Network(term_left)
    left.connect_to_ground(node = 0, R= 500)
    left.connect_to_ground(node = 1, R= 500)

    right = nw.Network(term_right)
    right.connect_to_ground(node = 2, R= 500)
    right.connect_to_ground(node = 3, R= 500)
    

    # left = nw.Network([0,1],[0])
    # conn = [{"node":0, "conductor":0}, {"node":1, "conductor":1}]
    # left.add_nodes_in_line(0, line, conn, "S")
    # left.connect_to_ground(node = 0, R= 500)
    # left.connect_to_ground(node = 1, R= 500)

    # right = nw.Network([2,3],[0])
    # conn = [{"node":2, "conductor":0}, {"node":3, "conductor":1}]
    # right.add_nodes_in_line(0, line, conn, "L")
    # right.connect_to_ground(node = 2, R= 500)
    # right.connect_to_ground(node = 3, R= 500)


    manager.add_bundle(line)
    manager.add_networks([left, right])
    # line.dt = finalTime/200
    manager.run_until(finalTime)

    # times = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    # voltages = []
    # for (t, v) in zip(times, voltages):
    #     index = np.argmin(np.abs(v_probe.t - t*1e-9))
    #     assert np.all(np.isclose(v_probe.val[index, 0], v*1e-6, atol=2.5e-3))

    # plt.plot(1e9*probe.v0.t, 1e3*probe.v0.val, label='port')
    plt.plot(1e9*v_probe.t, 1e6*v_probe.val, label='v probe')
    plt.ylabel(r'$V_1 (0, t)\,[\mu V]$')
    plt.xlabel(r'$t\,[ns]$')
    # plt.xticks(range(0, int(finalTime*1e9), 5))
    plt.grid('both')
    plt.legend()
    plt.show()
    
def test_wire_over_ground_incident_E_transversal_paul_12_4_10ns():

    """
    Described in Ch. 12.4 "Computed results" of Paul Clayton
    Analysis of Multiconductor Transmission Lines. 2007. 
    """
    l = np.zeros([2,2])
    l[0] = [0.7485e-6, 0.2408e-6]
    l[1] = [0.2408e-6, 0.7485e-6]

    c = np.zeros([2,2])
    c[0] = [24.982e-12, -6.266e-12]
    c[1] = [-6.266e-12, 24.982e-12]

    finalTime = 100e-9
    manager = mtln.MTLN()    
    line = mtl.MTL(l=l, c=c, length=2.0, ndiv=10, name = "line")

    rise_time=10e-9
    wire_separation = 0.00127
    e_y = wf.null()
    e_z = wf.null()
    e_x = wf.ramp_pulse_x_sp(A=1, x0=rise_time)
    
    distances = np.zeros([2, line.u.shape[0], 3])
    distances[0,:,0] = -wire_separation
    distances[1,:,0] = wire_separation
    line.add_external_field(mtl.Field(e_x,e_y,e_z), distances)

    v_probe = line.add_probe(position=0.0, probe_type='voltage')
    
    # left = nw.Network([0,1],[0])
    # conn = [{"node":0, "conductor":0}, {"node":1, "conductor":1}]
    # left.add_nodes_in_line(0, line, conn, "S")
    # left.connect_to_ground(node = 0, R= 500)
    # left.connect_to_ground(node = 1, R= 500)

    # right = nw.Network([2,3],[0])
    # conn = [{"node":2, "conductor":0}, {"node":3, "conductor":1}]
    # right.add_nodes_in_line(0, line, conn, "L")
    # right.connect_to_ground(node = 2, R= 500)
    # right.connect_to_ground(node = 3, R= 500)

    term_left  = {line: {"connections" : [[0,0], [1,1]], "side": "S" }}
    term_right  = {line: {"connections" : [[2,0], [3,1]], "side": "L" }}
    
    left = nw.Network(term_left)
    left.connect_to_ground(node = 0, R= 500)
    left.connect_to_ground(node = 1, R= 500)

    right = nw.Network(term_right)
    right.connect_to_ground(node = 2, R= 500)
    right.connect_to_ground(node = 3, R= 500)


    manager.add_bundle(line)
    manager.add_networks([left, right])
    # manager.dt = finalTime/200
    manager.run_until(finalTime)

    # times = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    # voltages = []
    # for (t, v) in zip(times, voltages):
    #     index = np.argmin(np.abs(v_probe.t - t*1e-9))
    #     assert np.all(np.isclose(v_probe.val[index, 0], v*1e-6, atol=2.5e-3))

    # plt.plot(1e9*probe.v0.t, 1e3*probe.v0.val, label='port')
    plt.plot(1e9*v_probe.t, 1e6*v_probe.val, label='v probe')
    plt.ylabel(r'$V_1 (0, t)\,[\mu V]$')
    plt.xlabel(r'$t\,[ns]$')
    # plt.xticks(range(0, int(finalTime*1e9), 5))
    plt.grid('both')
    plt.legend()
    plt.show()

@ut.skip("trapezoidal wave implementation has changed. Probably, outdated test")
def test_wire_over_ground_incident_E_transversal_paul_12_4_1ns():

    """
    Described in Ch. 12.4 "Computed results" of Paul Clayton
    Analysis of Multiconductor Transmission Lines. 2007. 
    """
    l = np.zeros([2,2])
    l[0] = [0.7485e-6, 0.2408e-6]
    l[1] = [0.2408e-6, 0.7485e-6]

    c = np.zeros([2,2])
    c[0] = [24.982e-12, -6.266e-12]
    c[1] = [-6.266e-12, 24.982e-12]

    finalTime = 100e-9
    manager = mtln.MTLN()    
    line = mtl.MTL(l=l, c=c, length=2.0, ndiv=100, name ="line")

    rise_time=1e-9
    wire_separation = 0.00127
    e_y = wf.null()
    e_z = wf.null()
    e_x = wf.ramp_pulse_x_sp(A=1, x0=rise_time)
    
    distances = np.zeros([2, line.u.shape[0], 3])
    distances[0,:,0] = -wire_separation
    distances[1,:,0] = wire_separation
    line.add_external_field(mtl.Field(e_x,e_y,e_z), distances)

    v_probe = line.add_probe(position=0.0, probe_type='voltage')
    
    term_left  = {line: {"connections" : [[0,0], [1,1]], "side": "S" }}
    term_right  = {line: {"connections" : [[2,0], [3,1]], "side": "L" }}
    
    left = nw.Network(term_left)
    left.connect_to_ground(node = 0, R= 500)
    left.connect_to_ground(node = 1, R= 500)

    right = nw.Network(term_right)
    right.connect_to_ground(node = 2, R= 500)
    right.connect_to_ground(node = 3, R= 500)

    # left = nw.Network([0,1],[0])
    # conn = [{"node":0, "conductor":0}, {"node":1, "conductor":1}]
    # left.add_nodes_in_line(0, line, conn, "S")
    # left.connect_to_ground(node = 0, R= 500)
    # left.connect_to_ground(node = 1, R= 500)

    # right = nw.Network([2,3],[0])
    # conn = [{"node":2, "conductor":0}, {"node":3, "conductor":1}]
    # right.add_nodes_in_line(0, line, conn, "L")
    # right.connect_to_ground(node = 2, R= 500)
    # right.connect_to_ground(node = 3, R= 500)


    manager.add_bundle(line)
    manager.add_networks([left, right])
    manager.dt = finalTime/1500
    manager.run_until(finalTime)

    # times = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    # voltages = []
    # for (t, v) in zip(times, voltages):
    #     index = np.argmin(np.abs(v_probe.t - t*1e-9))
    #     assert np.all(np.isclose(v_probe.val[index, 0], v*1e-6, atol=2.5e-3))

    # plt.plot(1e9*probe.v0.t, 1e3*probe.v0.val, label='port')
    plt.plot(1e9*v_probe.t, 1e6*v_probe.val, label='v probe')
    plt.ylabel(r'$V_1 (0, t)\,[\mu V]$')
    plt.xlabel(r'$t\,[ns]$')
    # plt.xticks(range(0, int(finalTime*1e9), 5))
    plt.grid('both')
    plt.legend()
    plt.show()
