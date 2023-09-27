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

    
    c = 6.667e-12
    l  = 1/(c*(3.0e8)**2)
    r = 5.0e-3
    line0 = mtl.MTL(l=l, c=c, r = r,length=3.0, ndiv=50, name = "line0")
    line0.dt = 1.9e-10
    bundle0 = mtl.MTLD(levels={0:[line0]}, name="bundle0")
    bundle0.dt = 1.9e-10

    probe_v = bundle0.add_probe(0.0, "voltage")
    probe_i = bundle0.add_probe(3.0, "current")
    

    term_1  = {line0: {"connections" : [[0,0]], "side": "S", "bundle":bundle0 }}
    term_2  = {line0: {"connections" : [[1,0]], "side": "L", "bundle":bundle0 }}

    t1 = nw.Network(term_1)
    t1.connect_to_ground(node = 0, R  = R1)
    
    t2 = nw.Network(term_2)
    t2.connect_to_ground(node = 1, R = R2)
    
    manager = mtln.MTLN()
    manager.add_bundle(bundle0)
    manager.dt = 1.9e-10

    distances = np.zeros([1, line0.u.shape[0], 3])
    distances[:,:,0] = 0.0508
    ey = wf.null()
    ez = wf.null()
    ex = wf.double_exp_sp(C =5.25e4, a = 4.0e6, b = 4.76e8)
    manager.add_external_field(Field(ex,ey,ez), distances)

    manager.add_network(nw.NetworkD({0:t1}))
    manager.add_network(nw.NetworkD({0:t2}))
    
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
    

def test_example_2():
    #external fields, polar angle, how to take into account delay?
    #need to provide orientation/position of conductors?    
    finalTime = 80e-9
    nt = 565
    manager = mtln.MTLN()
    c_wire = 6.667e-12
    length = 4.24
    l_wire = 1.0/(c_wire*(3.0e8)**2)
    r_wire = 1e-3
    bundles = []
    bundles.append(mtl.MTL(l=l_wire, c=c_wire ,r=r_wire, g=0.0,node_positions=np.array([[0.0,0.0,0.0],[ 0.0,  3.0,  3.0  ]]),   ndiv=50))
    bundles.append(mtl.MTL(l=l_wire, c=c_wire ,r=r_wire, g=0.0,node_positions=np.array([[0.0,0.0,0.0],[ 0.0, -3.0,  3.0  ]]),   ndiv=50))
    bundles.append(mtl.MTL(l=l_wire, c=c_wire ,r=r_wire, g=0.0,node_positions=np.array([[0.0,3.0,-3.0],[ 0.0, 0.0,  0.0 ]]),    ndiv=50))
    bundles.append(mtl.MTL(l=l_wire, c=c_wire ,r=r_wire, g=0.0,node_positions=np.array([[0.0,-3.0,-3.0],[ 0.0, 0.0, 0.0 ]]),    ndiv=50))

    probe_end_1 = bundles[0].add_probe(position = np.array([0.0,0.0,0.0]), probe_type = "current")
    probe_end_2 = bundles[0].add_probe(position = np.array([0.0,3.0,3.0]), probe_type = "voltage")

    distances = np.zeros([1, bundles[0].u.shape[0], 3])
    distances[:,:,0] = 0.0508



    [manager.add_bundle(i, bundles[i]) for i in range(4)]

    e = wf.double_exp_xy_sp(C =5.25e4, a = 4.0e6, b = 4.76e8)
    manager.add_external_field(Field(wf.null(), wf.null(), e), 2*distances)
    
    # magnitude = lambda t : wf.double_exp(t, C =5.25e4, a = 4.0e6, b = 4.76e8)
    # pw = PlaneWave(magnitude, np.deg2rad(120), np.deg2rad(0), np.deg2rad(0))
    # manager.add_planewave(pw, 2*distances)


    
    t = []
    for i in range(4):
        t.append(nw.Network(nodes = [i], bundles = [i]))
        t[i].add_nodes_in_line(bundle_number=i, bundle=bundles[i], connections=[{"node" : i, "conductor" : 0}], side="L")
        t[i].connect_to_ground(node = i, R = 1e6)
    # for i in range(2,4):
    #     t.append(nw.Network(nodes = [i], bundles = [i]))
    #     t[i].add_nodes_in_line(bundle_number=i, bundle=bundles[i], connections=[{"node" : i, "conductor" : 0}], side="S")
    #     t[i].connect_to_ground(node = i, R = 1e6)
    
    t.append(nw.Network(nodes=[4,5,6,7], bundles=[0,1,2,3]))
    t[4].add_nodes_in_line(bundle_number=0, bundle=bundles[0], connections=[{"node" : 4, "conductor" : 0}], side="S")
    t[4].add_nodes_in_line(bundle_number=1, bundle=bundles[1], connections=[{"node" : 5, "conductor" : 0}], side="S")
    t[4].add_nodes_in_line(bundle_number=2, bundle=bundles[2], connections=[{"node" : 6, "conductor" : 0}], side="S")
    t[4].add_nodes_in_line(bundle_number=3, bundle=bundles[3], connections=[{"node" : 7, "conductor" : 0}], side="S")
    t[4].short_nodes(4,5)
    t[4].short_nodes(4,6)
    t[4].short_nodes(4,7)
    t[4].short_nodes(5,6)
    t[4].short_nodes(5,7)
    t[4].short_nodes(6,7)

    manager.add_networks(t)
    manager.dt = 1.414e-10
    manager.run_until(finalTime)

    plt.figure()
    plt.plot(1e9*probe_end_1.t, probe_end_1.val, label = 'Current on S1 at the junction')
    plt.ylabel(r'$A (t)\,[A]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 90, 10))
    plt.grid('both')
    plt.legend()

    plt.figure()
    plt.plot(1e9*probe_end_2.t, probe_end_2.val, label = 'Voltage at floating end of S1')
    plt.ylabel(r'$V (t)\,[V]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 90, 10))
    plt.grid('both')
    plt.legend()
    
    plt.show()

def test_example_3():
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
    
    line = mtl.MTL(l=l, c=c, node_positions=np.array([[0.0,0.0,0.0],[0.0,0.0,0.3048]]), ndiv=100, name = "line")
    bundle = mtl.MTLD({0 : [line]}, name = "bundle")
    v_probe_L = bundle.add_probe(position=np.array([0.0,0.0,0.0]), probe_type='voltage')
    v_probe_R = bundle.add_probe(position=np.array([0.0,0.0,0.3048]), probe_type='voltage')
    
    
    term_1  = {line: {"connections" : [[0,0], [1,1]], "side": "S", "bundle" : bundle }}
    term_2  = {line: {"connections" : [[2,0], [3,1]], "side": "L", "bundle" : bundle }}

    terminal_1 = nw.Network(term_1)
    terminal_1.connect_to_ground(node = 0, R  = 100)
    terminal_1.connect_to_ground(node = 1, R= 50, Vt = magnitude)
    
    terminal_2 = nw.Network(term_2)
    terminal_2.connect_to_ground(2, R= 102)
    terminal_2.connect_to_ground(3, R= 102)

    T1 =  nw.NetworkD({0:terminal_1})
    T2 =  nw.NetworkD({0:terminal_2})
    
    mtl_nw = mtln.MTLN()
    mtl_nw.add_bundle(bundle)
    mtl_nw.add_network(T1)
    mtl_nw.add_network(T2)
    mtl_nw.run_until(finalTime)

    plt.figure()
    plt.plot(1e9*v_probe_R.t, v_probe_R.val[:,1], label = 'End2 of C1')
    plt.ylabel(r'$V (t)\,[V]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 6, 1))
    plt.xlim(0,4)
    plt.grid('both')
    plt.legend()

    plt.figure()
    plt.plot(1e9*v_probe_L.t, v_probe_L.val[:,0], label = 'End1 of C2')
    plt.plot(1e9*v_probe_R.t, v_probe_R.val[:,0], label = 'End2 of C2')
    plt.ylabel(r'$V (t)\,[V]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 6, 1))
    plt.grid('both')
    plt.legend()
    
    plt.show()


def test_example_3_old_style():
    """
    g---V---R---1|-----------|3---R---g 
                 |           |
    g--- ---R---0|-----------|2---R---g 
    
    """
    l = np.array([
        [0.4946E-6, 0.0633E-6 ],
        [0.0633E-6,  0.4946E-6]])
    
    c = np.array([
        [62.8E-12, -4.94E-12],
        [-4.94E-12, 62.8E-12]])


    def magnitude(t): return wf.ramp_pulse(t, 4.0, 1.5e-9)
    

    finalTime = 5e-9
    
    eig , _ = linalg.eig(l.dot(c))
    v = np.real(np.max(1/np.sqrt(eig)))
    fMax = 1e9
    dx = v/(fMax*10)
    nx = int(0.3048/dx)
    
    line = mtl.MTL(l=l, c=c, length=0.3048, ndiv=nx, Zs=[100,50], Zl=[102,102])
    line.add_voltage_source(position=0.0,conductor=1,magnitude=magnitude)
    
    v_probe_L = line.add_probe(position=0.0, probe_type='voltage')
    v_probe_R = line.add_probe(position=0.3048, probe_type='voltage')
    i_probe_L = line.add_probe(position=0.0, probe_type='current')
    i_probe_R = line.add_probe(position=0.3048, probe_type='current')
    

    # line.dt = 0.95*line.dt
    line.run_until(finalTime)

    plt.figure()
    plt.plot(1e9*v_probe_L.t, v_probe_R.val[:,1], label = 'End2 of C1')
    plt.ylabel(r'$V (t)\,[V]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 6, 1))
    plt.xlim(0,4)
    plt.grid('both')
    plt.legend()

    plt.figure()
    plt.plot(1e9*v_probe_R.t, v_probe_L.val[:,0], label = 'End1 of C2')
    plt.plot(1e9*v_probe_R.t, v_probe_R.val[:,0], label = 'End2 of C2')
    plt.ylabel(r'$V (t)\,[V]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 6, 1))
    plt.grid('both')
    plt.legend()
    
    plt.show()

    


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
    line_1 = mtl.MTL(l=l_wire, c=c_wire, length=0.120, ndiv=80, name = "line_1")
    line_2 = mtl.MTL(l=l_wire, c=c_wire, length=0.120, ndiv=80, name = "line_2")
    line_4 = mtl.MTL(l=l_wire, c=c_wire, length=0.120, ndiv=80, name = "line_4")
    line_5 = mtl.MTL(l=l_wire, c=c_wire, length=0.120, ndiv=80, name = "line_5")

    assert (line_1.du[0][2] == 1.5e-3)
    assert (line_2.du[0][2] == 1.5e-3)
    assert (line_4.du[0][2] == 1.5e-3)
    assert (line_5.du[0][2] == 1.5e-3)

    v3 = 1.059e8
    c3 = np.zeros([2, 2])
    c3[0] = [2.242e-10, -7.453e-11]
    c3[1] = [-7.453e-11, 2.242e-10]
    l3 = np.zeros([2, 2])
    l3 = linalg.inv(c3)/v3**2
    
    line_3 = mtl.MTL(l=l3, c=c3, length=0.0245, ndiv=24, name = "line_3")
    pv = line_3.get_phase_velocities()
    assert(np.isclose(line_3.du[0][2], 1.02e-3, 0.01))

    
    #probes
    
    bundle_1 = mtl.MTLD(levels = {0:[line_1]}, name = "bundle_1")
    bundle_2 = mtl.MTLD(levels = {0:[line_2]}, name = "bundle_2")
    bundle_3 = mtl.MTLD(levels = {0:[line_3]}, name = "bundle_3")
    bundle_4 = mtl.MTLD(levels = {0:[line_4]}, name = "bundle_4")
    bundle_5 = mtl.MTLD(levels = {0:[line_5]}, name = "bundle_5")
    
    probe_1 = bundle_1.add_probe(position = 0.0,   probe_type = "voltage")
    probe_4 = bundle_4.add_probe(position = 0.120, probe_type = "voltage")
    probe_5 = bundle_5.add_probe(position = 0.120, probe_type = "voltage")

    manager.add_bundle(bundle_1)    
    manager.add_bundle(bundle_2)    
    manager.add_bundle(bundle_3)    
    manager.add_bundle(bundle_4)    
    manager.add_bundle(bundle_5)    
    
    def magnitude(t): return wf.gaussian_2(t, 1.0, 400e-12, 100e-12)
    
    term_1  = {line_1: {"connections" : [[0,0]], "side": "S", "bundle":bundle_1 }}
    term_2  = {line_2: {"connections" : [[1,0]], "side": "S", "bundle":bundle_2 }}

    conn_1  = {line_1: {"connections" : [[2,0]], "side": "L", "bundle":bundle_1 },
               line_2: {"connections" : [[3,0]], "side": "L", "bundle":bundle_2 },
               line_3: {"connections" : [[4,0], [5,1]], "side": "S", "bundle":bundle_3}}

    conn_2  = {line_3: {"connections" : [[8,0], [9,1]], "side": "L", "bundle":bundle_3   },
               line_4: {"connections" : [[11,0]], "side": "S", "bundle":bundle_4   },
               line_5: {"connections" : [[10,0]], "side": "S", "bundle":bundle_5   }}

    term_3  = {line_5: {"connections" : [[6,0]], "side": "L", "bundle":bundle_5   }}
    term_4  = {line_4: {"connections" : [[7,0]], "side": "L", "bundle":bundle_4   }}

    #terminal 1
    t1 = nw.Network(term_1)
    t1.connect_to_ground(node = 0, R  = 50, Vt=magnitude)
    
    #terminal 2
    t2 = nw.Network(term_2)
    t2.connect_to_ground(node = 1, R = 1e-6)
    
    #terminal 3
    t3 = nw.Network(term_3)
    t3.connect_to_ground(node = 6, R  = 50)
    
    #terminal 4
    t4 = nw.Network(term_4)
    t4.connect_to_ground(node = 7, R  = 50)

    #junction1
    j1 = nw.Network(conn_1)
    j1.short_nodes(3,5)
    j1.short_nodes(2,4)

    j2 = nw.Network(conn_2)
    j2.short_nodes(9,11)
    j2.short_nodes(8,10)

    manager.add_network(nw.NetworkD({0:t1}))
    manager.add_network(nw.NetworkD({0:t2}))
    manager.add_network(nw.NetworkD({0:t3}))
    manager.add_network(nw.NetworkD({0:t4}))
    manager.add_network(nw.NetworkD({0:j1}))
    manager.add_network(nw.NetworkD({0:j2}))

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

    
def test_5():
    """
    Coaxial with two wires, with shield
    Two levels:
    0: coaxial shield and reference (external reference)
    1: two wires and  reference (shield)
    """
    
    finalTime = 30e-9
    ### lines and bundles
    c0 =  20.72e-12
    v0 = 3.0e8
    l0 = 1/(c0*v0**2)
    line_0 = mtl.MTL(l=l0, c=c0, r=22.9e-3, length=0.540, ndiv=18, name = "line_0")
    
    
    c1 = np.array([[85.0e-12, -20.5e-12],
                   [-20.5e-12, 85.0e-12 ]])
    v1 = 2.01e8
    l1 = linalg.inv(c1)/v1**2
    line_1 = mtl.MTL(l=l1, c=c1, length=0.540, ndiv=18, name = "line_1")

    bundle = mtl.MTLD({0:[line_0],1:[line_1]}, name = "bundle_0")
    
    # term_0_L  = {line_0: {"connections" : {{"node" : 0, "conductor":0},{"node" : 0, "conductor":0}}, "side": "S" , "bundle_name" : "bundle_0" }}

    term_0_L  = {line_0: {"connections" : [[0,0]], "side": "S" , "bundle" : bundle }}
    # term_0_L  = {line_0: {"connections" : [[0,0]], "side": "S" }}
    terminal_0_left = nw.Network(term_0_L)
    terminal_0_left.connect_to_ground(node=0, R = 50)
    
    term_0_R  = {line_0: {"connections" : [[1,0]], "side": "L", "bundle" : bundle  }}
    terminal_0_right = nw.Network(term_0_R)
    terminal_0_right.connect_to_ground(node=1, R = 50)
    
    ##### level 1 terminals
    term_1_L = {line_1: {"connections" : [[2,0], [3,1]], "side": "S", "bundle" : bundle }}
    terminal_1_left = nw.Network(term_1_L)
    terminal_1_left.connect_to_ground(node=2, R = 50)
    terminal_1_left.connect_to_ground(node=3, R = 50)
    
    term_1_R = {line_1: {"connections" : [[4,0], [5,1]], "side": "L", "bundle" : bundle  }}
    terminal_1_right = nw.Network(term_1_R)
    terminal_1_right.connect_to_ground(node=4, R = 50)
    terminal_1_right.connect_to_ground(node=5, R = 50)
    
    
    ##### terminals
    
    terminal_left  = nw.NetworkD({0:terminal_0_left, 1:terminal_1_left})
    terminal_right = nw.NetworkD({0:terminal_0_right, 1:terminal_1_right})

    ##### probes
    
    v_probe = bundle.add_probe(position = 0.540, probe_type = "voltage")
    i_probe = bundle.add_probe(position = 0.540, probe_type = "current")

    ##### external field

    distances = np.zeros([1, bundle.u.shape[0], 3])
    distances[:,:,0] = 0.0508

    bundle.add_external_field(mtl.Field(wf.null(), wf.null(), wf.double_exp_xy_sp(C =5.25e4, a = 4.0e6, b = 4.76e8)), 
                              distances)


    #transfer inductance pul
    yt01 =  4.0e-9
    #zt01 =  s*yt01 -> yt01 corrs. to the proportional term
    bundle.add_transfer_impedance(out_level=0, out_level_conductors=[0],
                                  in_level=1, in_level_conductors=[0,1],
                                  d=0, e=yt01, poles=np.array([]), residues=np.array([]))
    # bundle.add_transfer_impedance(0,1, 0, yt01, np.array([]), np.array([]))

    ##### manager and run

    manager = mtln.MTLN([bundle], [terminal_left, terminal_right])
    manager.run_until(finalTime)

    plt.figure()
    plt.plot(1e9*i_probe.t, i_probe.val[:,0], label = 'Current on shield')
    plt.ylabel(r'$I (t)\,[V]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 40, 10))
    plt.yticks(range(-2, 10, 2))
    plt.grid('both')
    plt.legend()

    plt.figure()
    plt.plot(1e9*v_probe.t, v_probe.val[:,1], label = 'Voltage on inner conductor 1')
    plt.plot(1e9*v_probe.t, v_probe.val[:,2], label = 'Voltage on inner conductor 2')
    plt.ylabel(r'$V (t)\,[V]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 40, 10))
    plt.yticks(range(-1, 4, 1))
    plt.grid('both')
    plt.legend()
    
    plt.show()


def test_6():
    """
    Coaxial with two wires, with shield
    Two levels:
    0: coaxial shield and reference (external reference)
    1: two wires and  reference (shield)
    """
    
    finalTime = 70e-9
    # lines and bundles
    ## level 0
    c0 = np.array([[85.0e-12, -20.5e-12, 1.00],
                   [-20.5e-12, 85.0e-12, 1.00],
                   [-20.5e-12, 85.0e-12, 1.00] ])
    l0 = np.array([[85.0e-12, -20.5e-12, 1.00],
                   [-20.5e-12, 85.0e-12 , 1.00],
                   [-20.5e-12, 85.0e-12 , 1.00]])
    
    line_0 = mtl.MTL(l=l0, c=c0, length=10, ndiv=300, name = "line_0")
    
    ##level 1
    c1_0 =  20.72e-12
    l1_0 =  20.72e-12
    line_1_0 = mtl.MTL(l=l1_0, c=c1_0, length=10, ndiv=300, name = "line_1_0")
    c1_1 =  20.72e-12
    l1_1 =  20.72e-12
    line_1_1 = mtl.MTL(l=l1_1, c=c1_1, length=10, ndiv=300, name = "line_1_1")
    c1_2 =  20.72e-12
    l1_2 =  20.72e-12
    line_1_2 = mtl.MTL(l=l1_2, c=c1_2, length=10, ndiv=300, name = "line_1_3")
    
    bundle = mtl.MTLD({0:[line_0],1:[line_1_0, line_1_1, line_1_2]}, name = "bundle")
    
    # networks
    ## level 0
    term_0_L  = {line_0: {"connections" : [[0,0], [1,1], [2,2]], "side": "S" , "bundle" : bundle }}
    term_0_R  = {line_0: {"connections" : [[3,0], [4,1], [5,2]], "side": "L" , "bundle" : bundle }}

    terminal_0_left = nw.Network(term_0_L)
    terminal_0_right = nw.Network(term_0_R)

    terminal_0_left.short_to_ground(node = 0)
    terminal_0_left.short_to_ground(node=1)
    terminal_0_left.short_to_ground(node=2)

    terminal_0_right.short_to_ground(node=3)
    terminal_0_right.short_to_ground(node=4)
    terminal_0_right.short_to_ground(node=5)
    # terminal_0_left.connect_to_ground(node=0, R = 50)
    # terminal_0_left.connect_to_ground(node=1, R = 50)
    # terminal_0_left.connect_to_ground(node=2, R = 50)

    # terminal_0_right.connect_to_ground(node=3, R = 50)
    # terminal_0_right.connect_to_ground(node=4, R = 50)
    # terminal_0_right.connect_to_ground(node=5, R = 50)

    ## level 1
    term_1_L  = {  line_1_0: {"connections" : [[6,0]], "side": "S" , "bundle" : bundle },
                   line_1_1: {"connections" : [[7,0]], "side": "S" , "bundle" : bundle },
                   line_1_2: {"connections" : [[8,0]], "side": "S" , "bundle" : bundle } }

    term_1_R  = {  line_1_0: {"connections" : [[9,0]], "side": "L" , "bundle" : bundle },
                   line_1_1: {"connections" : [[10,0]], "side": "L" , "bundle" : bundle},
                   line_1_2: {"connections" : [[11,0]], "side": "L" , "bundle" : bundle } }

    terminal_1_left = nw.Network(term_1_L)
    terminal_1_right = nw.Network(term_1_R)

    def pulse(t): return wf.sin_sq_pulse(t, 1.0, 1.6667e8)

    terminal_1_left.connect_to_ground(node=6, R = 50, Vt = pulse)
    terminal_1_left.connect_to_ground(node=7, R = 50)
    terminal_1_left.connect_to_ground(node=8, R = 50)
    
    terminal_1_right.connect_to_ground(node=9, R = 50)
    terminal_1_right.connect_to_ground(node=10, R = 50)
    terminal_1_right.connect_to_ground(node=11, R = 50)

    terminal_left   = nw.NetworkD({0:terminal_0_left,  1:terminal_1_left})
    terminal_right  = nw.NetworkD({0:terminal_0_right, 1:terminal_1_right})

    #manager

    v_probe = bundle.add_probe(position = 10.0, probe_type = "voltage")

    manager = mtln.MTLN([bundle], [terminal_left, terminal_right])
    manager.dt = 0.5e-10
    manager.run_until(finalTime)

    plt.figure()
    plt.plot(1e9*v_probe.t, 1e3*v_probe.val[:,3], label = 'Voltage on inner conductor 1')
    plt.plot(1e9*v_probe.t, 1e3*v_probe.val[:,4], label = 'Voltage on inner conductor 2')
    plt.plot(1e9*v_probe.t, 1e3*v_probe.val[:,5], label = 'Voltage on inner conductor 3')
    plt.ylabel(r'$V (t)\,[mV]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 60, 10))
    plt.yticks(range(-200, 400, 100))
    plt.grid('both')
    plt.legend()
    
    plt.show()

    
def test_7():
    """
    Three levels
    0: external shield
    1: coaxial shields 
    2: coaxials
    """
    
    finalTime = 50e-9
    dt = 0.5e-10
    v0 = 3.0e8
    v1 = 1.544e8
    v2 = 2.01e9
    dx = 3.0e-2
    
    # C, L, R MATRICES
    
    ## S1
    c_s1_0 = 20.27e-12
    l_s1_0 = 1/(c_s1_0*v0**2)
    r_s1_0 = 22.9e-3
    
    c_s1_1 = np.array([[588.4e-12, 0.0, 0.0, 0.0],
                       [0.0, 588.4e-12, 0.0, 0.0],
                       [0.0, 0.0, 588.4e-12, 0.0],
                       [0.0, 0.0, 0.0, 588.4e-12]]
                      )
    l_s1_1 = linalg.inv(c_s1_1)/v1**2
    
    r_s1_1 = np.array([[3.9e-3, 0.0, 0.0, 0.0],
                       [0.0, 3.9e-3, 0.0, 0.0],
                       [0.0, 0.0, 3.9e-3, 0.0],
                       [0.0, 0.0, 0.0, 3.9e-3]])
    
    c_s1_2 = np.array([[105.5e-12, -20.5e-12],
                      [-20.5e-12, 105.5e-12]])
    l_s1_2 = linalg.inv(c_s1_2)/v2**2

    ## S2
    c_s2_0 = 17.14e-12
    l_s2_0 = c_s2_0/v0**2
    r_s2_0 = 11.8e-3
    
    c_s2_1 = 323.1e-12
    l_s2_1 = 1/(c_s2_1*v1**2)
    r_s2_1 = 12.2e-3
    
    c_s2_2 = np.array([[105.5e-12, -20.5e-12],
                      [-20.5e-12, 105.5e-12]])
    l_s2_2 = linalg.inv(c_s2_2)/v2**2

    ## S3
    c_s3_0 = 19.15e-12
    l_s3_0 = c_s3_0/v0**2
    r_s3_0 = 17.3e-3
    
    c_s3_1 = np.array([[471.9e-12, 0.0, 0.0],
                       [0.0, 471.9e-12, 0.0],
                       [0.0, 0.0, 471.9e-12]]
                      )
    l_s3_1 = linalg.inv(c_s3_1)/v1**2
    
    r_s3_1 = np.array([[6.5e-3, 0.0, 0.0],
                       [0.0, 6.5e-3, 0.0],
                       [0.0, 0.0, 6.5e-3]])
    
    c_s3_2 = np.array([[105.5e-12, -20.5e-12],
                      [-20.5e-12, 105.5e-12]])
    l_s3_2 = linalg.inv(c_s3_2)/v2**2

    ## S4
    c_s4_0 = 18.35e-12
    l_s4_0 = c_s4_0/v0**2
    r_s4_0 = 14.8e-3
    
    c_s4_1 = np.array([[363.7e-12, 0.0],
                       [0.0, 363.7e-12]])
    l_s4_1 = linalg.inv(c_s4_1)/v1**2
    
    r_s4_1 = np.array([[4.2e-3, 0.0],
                       [0.0, 4.2e-3]])
    
    c_s4_2 = np.array([[105.5e-12, -20.5e-12],
                      [-20.5e-12, 105.5e-12]])
    l_s4_2 = linalg.inv(c_s4_2)/v2**2

    ## s5
    c_s5_0 = 17.14e-12
    l_s5_0 = c_s5_0/v0**2
    r_s5_0 = 11.8e-3
    
    c_s5_1 = 323.1e-12
    l_s5_1 = 1/(c_s5_1*v1**2)
    r_s5_1 = 5.7e-3
    
    c_s5_2 = np.array([[105.5e-12, -20.5e-12],
                      [-20.5e-12, 105.5e-12]])
    l_s5_2 = linalg.inv(c_s5_2)/v2**2

    # lines and bundles
    ## BUNDLE S1
    ### LEVEL 0
    line_s1_0 = mtl.MTL(l = l_s1_0, c = c_s1_0, r = r_s1_0, length=0.54, ndiv=18, name = "s1_0")
    ### LEVEL 1
    line_s1_1 = mtl.MTL(l = l_s1_1, c = c_s1_1, r = r_s1_1, length=0.54, ndiv=18, name = "s1_1")
    ### LEVEL 2
    line_s1_2a = mtl.MTL(l = l_s1_2, c = c_s1_2, length=0.54, ndiv=18, name = "s1_2a")
    line_s1_2b = mtl.MTL(l = l_s1_2, c = c_s1_2, length=0.54, ndiv=18, name = "s1_2b")
    line_s1_2c = mtl.MTL(l = l_s1_2, c = c_s1_2, length=0.54, ndiv=18, name = "s1_2c")
    line_s1_2d = mtl.MTL(l = l_s1_2, c = c_s1_2, length=0.54, ndiv=18, name = "s1_2d")
    ## BUNDLE    
    bundle_s1 = mtl.MTLD({0:[line_s1_0],1:[line_s1_1], 2:[line_s1_2a, line_s1_2b,line_s1_2c,line_s1_2d]}, name = "bundle_s1")
    
    # lines and bundles
    ## BUNDLE s2
    ### LEVEL 0
    line_s2_0 = mtl.MTL(l = l_s2_0, c = c_s2_0, r = r_s2_0, length=0.343, ndiv=18, name = "s2_0")
    ### LEVEL 1
    line_s2_1 = mtl.MTL(l = l_s2_1, c = c_s2_1, r = r_s2_1, length=0.343, ndiv=18, name = "s2_1")
    ### LEVEL 2
    line_s2_2a = mtl.MTL(l = l_s2_2, c = c_s2_2, length=0.343, ndiv=18, name = "s2_2a")
    ##BUNDLE
    bundle_s2 = mtl.MTLD({0:[line_s2_0],1:[line_s2_1], 2:[line_s2_2a]}, name = "bundle_s2")
    
    ## BUNDLE S3
    ### LEVEL 0
    line_s3_0 = mtl.MTL(l = l_s3_0, c = c_s3_0, r = r_s3_0, length=0.165, ndiv=18, name = "s3_0")
    ### LEVEL 1
    line_s3_1 = mtl.MTL(l = l_s3_1, c = c_s3_1, r = r_s3_1, length=0.165, ndiv=18, name = "s3_1")
    ### LEVEL 2
    line_s3_2a = mtl.MTL(l = l_s3_2, c = c_s3_2, length=0.165, ndiv=18, name = "s3_2a")
    line_s3_2b = mtl.MTL(l = l_s3_2, c = c_s3_2, length=0.165, ndiv=18, name = "s3_2b")
    line_s3_2c = mtl.MTL(l = l_s3_2, c = c_s3_2, length=0.165, ndiv=18, name = "s3_2c")
    ## BUNDLE    
    bundle_s3 = mtl.MTLD({0:[line_s3_0],1:[line_s3_1], 2:[line_s3_2a, line_s3_2b,line_s3_2c]}, name = "bundle_s3")

    ## BUNDLE s4
    ### LEVEL 0
    line_s4_0 = mtl.MTL(l = l_s4_0, c = c_s4_0, r = r_s4_0, length=0.356, ndiv=18, name = "s4_0")
    ### LEVEL 1
    line_s4_1 = mtl.MTL(l = l_s4_1, c = c_s4_1, r = r_s4_1, length=0.356, ndiv=18, name = "s4_1")
    ### LEVEL 2
    line_s4_2a = mtl.MTL(l = l_s4_2, c = c_s4_2, length=0.356, ndiv=18, name = "s4_2a")
    line_s4_2b = mtl.MTL(l = l_s4_2, c = c_s4_2, length=0.356, ndiv=18, name = "s4_2b")
    line_s4_2c = mtl.MTL(l = l_s4_2, c = c_s4_2, length=0.356, ndiv=18, name = "s4_2c")
    ## BUNDLE    
    bundle_s4 = mtl.MTLD({0:[line_s4_0],1:[line_s4_1], 2:[line_s4_2a, line_s4_2b]}, name = "bundle_4")

    # lines and bundles
    ## BUNDLE s5
    ### LEVEL 0
    line_s5_0 = mtl.MTL(l = l_s5_0, c = c_s5_0, r = r_s5_0, length=0.178, ndiv=18, name = "s5_0")
    ### LEVEL 1
    line_s5_1 = mtl.MTL(l = l_s5_1, c = c_s5_1, r = r_s5_1, length=0.178, ndiv=18, name = "s5_1")
    ### LEVEL 2
    line_s5_2a = mtl.MTL(l = l_s5_2, c = c_s5_2, length=0.178, ndiv=18, name = "s5_2a")
    ##BUNDLE
    bundle_s5 = mtl.MTLD({0:[line_s5_0],1:[line_s5_1], 2:[line_s5_2a]}, name = "bundle_5")

    # TERMINALS
    ## T1
    ###LEVEL 0
    
    ###LEVEL 1
    
    ###LEVEL 2
    
    # T1 = nw.NetworkD({0 : t1_0, 1: t1_1, 2: t1_2})
    
    
    # manager = mtln.MTLN([bundle_s1, bundle_s2, bundle_s3, bundle_s4, bundle_s5], [T1, T2, J1, J2, T4, T5])
    # manager.dt = dt
    # manager.run_until(finalTime)

    
    # # networks
    # ## level 0
    # term_0_L  = {line_0: {"connections" : [[0,0], [1,1], [2,2]], "side": "S" , "bundle" : bundle }}
    # term_0_R  = {line_0: {"connections" : [[3,0], [4,1], [5,2]], "side": "L" , "bundle" : bundle }}

    # terminal_0_left = nw.Network(term_0_L)
    # terminal_0_right = nw.Network(term_0_R)

    # terminal_0_left.short_to_ground(node = 0)
    # terminal_0_left.short_to_ground(node=1)
    # terminal_0_left.short_to_ground(node=2)

    # terminal_0_right.short_to_ground(node=3)
    # terminal_0_right.short_to_ground(node=4)
    # terminal_0_right.short_to_ground(node=5)
    # # terminal_0_left.connect_to_ground(node=0, R = 50)
    # # terminal_0_left.connect_to_ground(node=1, R = 50)
    # # terminal_0_left.connect_to_ground(node=2, R = 50)

    # # terminal_0_right.connect_to_ground(node=3, R = 50)
    # # terminal_0_right.connect_to_ground(node=4, R = 50)
    # # terminal_0_right.connect_to_ground(node=5, R = 50)

    # ## level 1
    # term_1_L  = {  line_1_0: {"connections" : [[6,0]], "side": "S" , "bundle" : bundle },
    #                line_1_1: {"connections" : [[7,0]], "side": "S" , "bundle" : bundle },
    #                line_1_2: {"connections" : [[8,0]], "side": "S" , "bundle" : bundle } }

    # term_1_R  = {  line_1_0: {"connections" : [[9,0]], "side": "L" , "bundle" : bundle },
    #                line_1_1: {"connections" : [[10,0]], "side": "L" , "bundle" : bundle},
    #                line_1_2: {"connections" : [[11,0]], "side": "L" , "bundle" : bundle } }

    # terminal_1_left = nw.Network(term_1_L)
    # terminal_1_right = nw.Network(term_1_R)

    # def pulse(t): return wf.sin_sq_pulse(t, 1.0, 1.6667e8)

    # terminal_1_left.connect_to_ground(node=6, R = 50, Vt = pulse)
    # terminal_1_left.connect_to_ground(node=7, R = 50)
    # terminal_1_left.connect_to_ground(node=8, R = 50)
    
    # terminal_1_right.connect_to_ground(node=9, R = 50)
    # terminal_1_right.connect_to_ground(node=10, R = 50)
    # terminal_1_right.connect_to_ground(node=11, R = 50)

    # terminal_left   = nw.NetworkD({0:terminal_0_left,  1:terminal_1_left})
    # terminal_right  = nw.NetworkD({0:terminal_0_right, 1:terminal_1_right})

    # #manager

    # v_probe = bundle.add_probe(position = 10.0, probe_type = "voltage")

    # manager = mtln.MTLN([bundle], [terminal_left, terminal_right])
    # manager.dt = 0.5e-10
    # manager.run_until(finalTime)

    # plt.figure()
    # plt.plot(1e9*v_probe.t, 1e3*v_probe.val[:,3], label = 'Voltage on inner conductor 1')
    # plt.plot(1e9*v_probe.t, 1e3*v_probe.val[:,4], label = 'Voltage on inner conductor 2')
    # plt.plot(1e9*v_probe.t, 1e3*v_probe.val[:,5], label = 'Voltage on inner conductor 3')
    # plt.ylabel(r'$V (t)\,[mV]$')
    # plt.xlabel(r'$t\,[ns]$')
    # plt.xticks(range(0, 60, 10))
    # plt.yticks(range(-200, 400, 100))
    # plt.grid('both')
    # plt.legend()
    
    # plt.show()

    
def test_example_8():
    l = np.zeros([2, 2])
    l[0] = [0.4946E-6, 0.0633E-6 ]
    l[1] = [0.0633E-6,  0.4946E-6]
    
    c = np.zeros([2, 2])
    c[0] = [62.8E-12,-4.94E-12]
    c[1] = [-4.94E-12, 62.8E-12]

    def magnitude_n0(t): return wf.ramp_pulse(t, -4.0, 1.5e-9)
    def magnitude_n1(t): return wf.ramp_pulse(t, 4.0, 1.5e-9)
    

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
    
    line = mtl.MTL(l=l, c=c, node_positions=np.array([[0.0,0.0,0.0],[0.0,0.0,0.3048]]), ndiv=100, name = "line")
    bundle = mtl.MTLD({0:[line]}, name = "bundle")
    v_probe_L = bundle.add_probe(position=np.array([0.0,0.0,0.0]), probe_type='voltage')
    v_probe_R = bundle.add_probe(position=np.array([0.0,0.0,0.3048]), probe_type='voltage')
    
    
    term_1  = {line: {"connections" : [[0,0], [1,1]], "side": "S", "bundle" : bundle }}
    term_2  = {line: {"connections" : [[2,0], [3,1]], "side": "L", "bundle" : bundle }}

    terminal_1 = nw.Network(term_1)
    terminal_1.connect_to_ground(node = 0, R  = 100, Vt = magnitude_n0)
    terminal_1.connect_to_ground(node = 1, R= 50, Vt = magnitude_n1)
    
    terminal_2 = nw.Network(term_2)
    terminal_2.connect_to_ground(2, R= 102, )
    terminal_2.connect_to_ground(3, R= 102)

    T1 = nw.NetworkD({0 : terminal_1})
    T2 = nw.NetworkD({0 : terminal_2})
    
    mtl_nw = mtln.MTLN()
    mtl_nw.add_bundle(bundle)
    mtl_nw.add_network(T1)
    mtl_nw.add_network(T2)
    mtl_nw.run_until(finalTime)

    plt.figure()
    plt.plot(1e9*v_probe_L.t, v_probe_R.val[:,1], label = 'End2 of C1')
    plt.ylabel(r'$V (t)\,[V]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 6, 1))
    plt.yticks(range(-3, 4, 1))
    plt.xlim(0,4)
    plt.grid('both')
    plt.legend()

    plt.figure()
    plt.plot(1e9*v_probe_R.t, v_probe_L.val[:,0], label = 'End1 of C2')
    plt.plot(1e9*v_probe_R.t, v_probe_R.val[:,0], label = 'End2 of C2')
    plt.ylabel(r'$V (t)\,[V]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 6, 1))
    plt.yticks(range(-3, 4, 1))
    plt.grid('both')
    plt.legend()
    
    plt.show()
    