import numpy as np
import matplotlib.pyplot as plt

import sympy as sp
from scipy.constants import epsilon_0, mu_0, speed_of_light

# import src.mtlnd as mtld
import src.mtln as mtln
import src.mtl as mtl
import src.networks as nw

from src.networkExtraction import *

import src.waveforms as wf

#test from m_harness examples manual (v5)

def test_example_1():
    
    """
    External field: double exponential 
     _________              __________
    |         |            |          |          
    | g--R1---0------------1---R2--g  |                
    |_________|     b0     |__________|                    
    """
    
    finalTime =  80e-9
    R1 = 1e6
    R2 = 1e-6

    manager = mtln.MTLN()
    
    c = 6.667e-12
    l  = 1/(c*(3.0e8)**2)
    r = 5.0e-3
    # line = mtl.MTL_losses(l = l, c = c, r = r, g = 0.0, length=3.0, nx=50, Zs=1e6, Zl=1e-6)
    line = mtl.MTL_losses(l=l, c=c, r = r, g = 0.0,length=3.0, nx=50)
    line.dt = 1.9e-10
    # line.dx = 0.06

    
    # rise_time, fall_time = 50e-9, 50e-9
    # x, z, t = sp.symbols('x z t')
    # magnitude = wf.trapezoidal_wave_sp(
    #     A=1, rise_time=rise_time, fall_time=fall_time, f0=1e6, D=0.5)
    x, z, t = sp.symbols('x z t')
    magnitude = wf.double_exp_sp(C =5.25e4, a = 4.0e6, b = 4.76e8)

    def e_z(x, z, t): return (x+z+t)*0
    def e_x(x, z, t): return magnitude
    line.add_external_field(e_x, e_z, ref_distance=0.0,
                            distances=np.array([0.0508]))


    probe_v = line.add_probe(0.0, "voltage")
    probe_i = line.add_probe(3.0, "current")
    
    # line.run_until(finalTime)
    
    manager.add_bundle(0,line)
    
    t1 = nw.Network(nw_number = 0, nodes = [0], bundles = [0])
    t1.add_nodes_in_bundle(bundle_number=0, bundle=line, connections=[{"node" : 0, "conductor" : 0}], side="S")
    t1.connect_to_ground(node = 0, R  = R1)
    
    t2 = nw.Network(nw_number = 1, nodes = [1], bundles = [0])
    t2.add_nodes_in_bundle(bundle_number=0, bundle=line, connections=[{"node" : 1, "conductor" : 0}], side="L")
    t2.connect_to_ground(node = 1, R = R2)
    # t2.short_to_ground(node = 1)
    
    manager.add_networks([t1,t2])
    
    manager.run_until(finalTime)

    fig, ax1 = plt.subplots()
    ax1.set_xlabel(r'$t\,[ns]$')
    ax1.set_ylabel(r'$V (t)\,[V]$', color='tab:red')
    ax1.plot(1e9*probe_v.t, probe_v.val, color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_xticks(range(0, 80, 20))

    ax2 = ax1.twinx() 
    ax2.set_ylabel(r'$I (t)\,[A]$', color='tab:blue') 
    ax2.plot(1e9*probe_i.t, probe_i.val, color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    fig.tight_layout() 
    plt.show()
    
    
def test_example_3():
    """
    Described in Ch. 9.3.1 "Ribbon Cables" of Paul Clayton
    Analysis of Multiconductor Transmission Lines. 2007. 
    """
    """
    Solved with mtl approach and mltn approach: tube + 2 termination networks
    """
    l = np.zeros([2, 2])
    l[0] = [0.4946E-6, 0.0633E-6 ]
    l[1] = [0.0633E-6,  0.4946E-6]
    
    c = np.zeros([2, 2])
    c[0] = [62.8E-12,-4.94E-12]
    c[1] = [-4.94E-12, 62.8E-12]

    def magnitude(t): return wf.ramp_pulse(t, 4.0, 1.5e-9)
    

    """
     _             _
    | |     1     | |
    | 1-----------3 |
    | |     0     | |
    | 0-----------2 |
    |_|           |_|
    term_1(0)     term_2(1)
    
    """
    finalTime = 5e-9
    mtl_nw = mtln.MTLN()
    
    bundle_0 = mtl.MTL(l=l, c=c, length=0.3048, nx=100)
    
    v_probe_L = bundle_0.add_probe(position=0.0, type='voltage')
    v_probe_R = bundle_0.add_probe(position=0.3048, type='voltage')
    i_probe_L = bundle_0.add_probe(position=0.0, type='current')
    i_probe_R = bundle_0.add_probe(position=0.3048, type='current')
    
    # bundle_0.dt = 1.80*bundle_0.dt
    # bundle_0.run_until(finalTime)
    
    mtl_nw.add_bundle(0, bundle_0)
    
    # mtl_nw.dt = 0.90*mtl_nw.dt
    # mtl_nw.dx = 0.3048/10
    
    #network definition
    terminal_1 = nw.Network(nw_number = 0, nodes = [0,1], bundles = [0])
    bundle_connections= [{"node" : 0, "conductor" : 0},{"node" : 1, "conductor" : 1}]
    
    terminal_1.add_nodes_in_bundle(bundle_number = 0, 
                                   bundle = bundle_0,
                                   connections = bundle_connections, 
                                   side= "S")
    #network connections
    terminal_1.connect_to_ground(node = 0, R  = 100)
    terminal_1.connect_to_ground(node = 1, R= 50, Vt = magnitude)
    mtl_nw.add_network(terminal_1)
    
    #network definition
    terminal_2 = nw.Network(nw_number = 1 ,nodes = [2,3], bundles = [0])
    bundle_connections= [{"node" : 2, "conductor" : 0},{"node" : 3, "conductor" : 1}]
    terminal_2.add_nodes_in_bundle(bundle_number = 0, 
                                   bundle = bundle_0, 
                                   connections= bundle_connections, 
                                   side= "L")

    #network connections
    terminal_2.connect_to_ground(2, R= 102)
    terminal_2.connect_to_ground(3, R= 102)
    mtl_nw.add_network(terminal_2)

    # mtl_nw.dt = 0.9*mtl_nw.dt
    mtl_nw.run_until(finalTime)

    plt.plot(1e9*v_probe_L.t, v_probe_L.val[:,0], label = 'V')
    plt.plot(1e9*v_probe_R.t, v_probe_R.val[:,0], label = 'V')
    plt.ylabel(r'$V (t)\,[V]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 5, 1))
    # plt.xlim(0,1)
    # plt.ylim(0,1)
    plt.grid('both')
    plt.legend()
    plt.show()

    
    # plt.ylabel(r'$V_1 (0, t)\,[V]$')
    # plt.xlabel(r'$t\,[ns]$')
    # plt.xticks(range(0, 5, 1))
    # plt.grid('both')
    # plt.show()


def test_example_3_old_style():
    """
    g---V---R---1|-----------|3---R---g 
                 |           |
    g--- ---R---0|-----------|2---R---g 
    
    """
    l = np.zeros([2, 2])
    l[0] = [0.4946E-6, 0.0633E-6 ]
    l[1] = [0.0633E-6,  0.4946E-6]
    
    c = np.zeros([2, 2])
    c[0] = [62.8E-12, -4.94E-12]
    c[1] = [-4.94E-12, 62.8E-12]


    # def magnitude(t): return wf.ramp_pulse(t, 4.0, 1.5e-9)
    def magnitude(t): return wf.ramp_pulse(t, 4.0, 1e-12)
    

    finalTime = 5e-9
    
    eig , _ = linalg.eig(l.dot(c))
    v = np.real(np.max(1/np.sqrt(eig)))
    fMax = 1e9
    dx = v/(fMax*10)
    nx = int(0.3048/dx)
    
    line = mtl.MTL(l=l, c=c, length=0.3048, nx=nx, Zs=[100,50], Zl=[1e-6,1e-6])
    line.add_voltage_source(position=0.0,conductor=1,magnitude=magnitude)
    
    v_probe_L = line.add_probe(position=0.0, type='voltage')
    v_probe_R = line.add_probe(position=0.3048, type='voltage')
    i_probe_L = line.add_probe(position=0.0, type='current')
    i_probe_R = line.add_probe(position=0.3048, type='current')
    

    line.dt = line.dt/100
    line.run_until(finalTime)

    # plt.figure()
    # plt.plot(1e9*v_probe_L.t, v_probe_R.val[:,1], label = 'End2 of C1')
    # plt.ylabel(r'$V (t)\,[V]$')
    # plt.xlabel(r'$t\,[ns]$')
    # plt.xticks(range(0, 6, 1))
    # plt.xlim(0,4)
    # plt.grid('both')
    # plt.legend()

    # plt.plot(1e9*v_probe_R.t, v_probe_L.val[:,0], label = 'End1 of C2')
    plt.plot(1e9*v_probe_R.t, v_probe_L.val[:,1], label = 'End1 of C1')
    plt.plot(1e9*v_probe_R.t, v_probe_R.val[:,1], label = 'End2 of C1')
    plt.ylabel(r'$V (t)\,[V]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 6, 1))
    # plt.ylim(-0.1,0.15)
    plt.grid('both')
    plt.legend()
    
    plt.show()
    # plt.ylabel(r'$V_1 (0, t)\,[V]$')
    # plt.xlabel(r'$t\,[ns]$')
    # plt.xticks(range(0, 5, 1))
    # plt.grid('both')
    # plt.show()

    

def test_example_4():
    
    
    """
     _     l2      _     _     l4      _
    | |     b1    | | 1 | |    b3     | |
    |_1-----------3-5---9-11----------7_|
    t2            | |   | |            t4
                  | |b2 | |
     _      b0    | | 0 | |    b4      _
    | 0-----------2-4---8-10----------6 |
    |_|     l1    |_| l3|_|     l5    |_|
    t1            j1    j2             t3 
    
    """
    finalTime = 6.46e-9

    manager = mtln.MTLN()

    c_wire = 1.915e-10
    v_wire = 1.041e8
    l_wire = 1/(c_wire*v_wire**2)
    line_1 = mtl.MTL(l=l_wire, c=c_wire, length=0.120, nx=80)
    line_2 = mtl.MTL(l=l_wire, c=c_wire, length=0.120, nx=80)
    line_4 = mtl.MTL(l=l_wire, c=c_wire, length=0.120, nx=80)
    line_5 = mtl.MTL(l=l_wire, c=c_wire, length=0.120, nx=80)

    assert (line_1.dx == 1.5e-3)
    assert (line_2.dx == 1.5e-3)
    assert (line_4.dx == 1.5e-3)
    assert (line_5.dx == 1.5e-3)

    v3 = 1.059e8
    c3 = np.zeros([2, 2])
    c3[0] = [2.242e-10, -7.453e-11]
    c3[1] = [-7.453e-11, 2.242e-10]
    l3 = np.zeros([2, 2])
    l3 = linalg.inv(c3)/v3**2
    n = 1
    line_3 = mtl.MTL(l=l3, c=c3, length=n*0.0245, nx=24)
    # assert(np.isclose(line_3.dx, 1.02e-3, 0.01))

    
    #probes
    probe_1 = line_1.add_probe(position = 0.0,   type = "voltage")
    probe_4 = line_4.add_probe(position = 0.120, type = "voltage")
    probe_5 = line_5.add_probe(position = 0.120, type = "voltage")
    
    manager.add_bundle(0, line_1)    
    manager.add_bundle(1, line_2)    
    manager.add_bundle(2, line_3)    
    manager.add_bundle(3, line_4)    
    manager.add_bundle(4, line_5)    
    #terminal 1
    def magnitude(t): return wf.gaussian_2(t, 400e-12, 100e-12)
    t1 = nw.Network(nw_number = 0, nodes = [0], bundles = [0])
    t1.add_nodes_in_bundle(bundle_number=0, bundle=line_1, connections=[{"node" : 0, "conductor" : 0}], side="S")
    t1.connect_to_ground(node = 0, R  = 50, Vt=magnitude)
    
    #terminal 2
    t2 = nw.Network(1, [1], [1])
    t2.add_nodes_in_bundle(1, line_2, [{"node" : 1, "conductor" : 0}], "S")
    t2.connect_to_ground(node = 1, R = 1e-6)
    
    #terminal 3
    t3 = nw.Network(2, [6], [4])
    t3.add_nodes_in_bundle(4, line_5, [{"node" : 6, "conductor" : 0}], "L")
    t3.connect_to_ground(node = 6, R  = 50)
    
    #terminal 4
    t4 = nw.Network(3, [7], [3])
    t4.add_nodes_in_bundle(3, line_4, [{"node" : 7, "conductor" : 0}], "L")
    t4.connect_to_ground(node = 7, R  = 50)

    #junction1
    j1 = nw.Network(4, [2,3,4,5], [0,1,2])
    j1.add_nodes_in_bundle(0, line_1, [{"node":2,"conductor":0}],"L")
    j1.add_nodes_in_bundle(1, line_2, [{"node":3,"conductor":0}],"L")
    j1.add_nodes_in_bundle(2, line_3, [{"node":4,"conductor":0},{"node":5,"conductor":1}],"S")
    j1.short_nodes(3,5)
    j1.short_nodes(2,4)

    j2 = nw.Network(5, [8,9,10,11], [2,3,4])
    j2.add_nodes_in_bundle(2, line_3, [{"node":8,"conductor":0},{"node":9,"conductor":1}],"L")
    j2.add_nodes_in_bundle(4, line_5, [{"node":10,"conductor":0}],"S")
    j2.add_nodes_in_bundle(3, line_4, [{"node":11,"conductor":0}],"S")
    j2.short_nodes(9,11)
    j2.short_nodes(8,10)

    manager.add_networks([t1,t2,t3,t4,j1,j2])

    # manager.dt = 3.23e-12

    manager.run_until(finalTime)
    fig, ax = plt.subplots(3,1)
    ax[0].plot(1e9*probe_1.t, probe_1.val, '-.', label = 'probe 1')
    ax[0].set_ylabel(r'$V (t)\,[V]$')
    ax[0].set_xticks(range(0, 9, 1))
    ax[0].set_ylim(-0.1,0.6)
    ax[0].grid('both')
    ax[0].legend()

    ax[1].plot(1e9*probe_4.t, probe_4.val, label = 'probe 4')
    ax[1].set_ylabel(r'$V (t)\,[V]$')
    ax[1].set_xticks(range(0, 9, 1))
    ax[1].set_ylim(-0.1,0.6)
    ax[1].grid('both')
    ax[1].legend()

    ax[2].plot(1e9*probe_5.t, probe_5.val, label = 'probe 5')
    ax[2].set_ylabel(r'$V (t)\,[V]$')
    ax[2].set_xlabel(r'$t\,[ns]$')
    ax[2].set_xticks(range(0, 9, 1))
    ax[2].set_ylim(-0.1,0.6)
    ax[2].grid('both')
    ax[2].legend()

    fig.tight_layout()
    plt.show()

    
    
    