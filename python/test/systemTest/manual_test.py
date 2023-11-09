import numpy as np
import matplotlib.pyplot as plt

import pytest

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
    
    term_1 = {'name': 'T1', 
            'materialType': 'Connector', 
            'connectorType': 'Conn_R',
            'resistance' : 1e6, 
            'inductance' : 0.0, 
            'capacitance': 1e+22}
    term_2 = {'name': 'T2', 
            'materialType': 'Connector', 
            'connectorType': 'Conn_R',
            'resistance' : 1e-6, 
            'inductance' : 0.0, 
            'capacitance': 1e+22}

    t1 = nw.Network({line0: {"connections" : [[0,0,term_1]], "side": "S", "bundle":bundle0 }})
    t1.connect_to_ground(node = 0, R  = R1)
    
    t2 = nw.Network({line0: {"connections" : [[1,0,term_2]], "side": "L", "bundle":bundle0 }})
    t2.connect_to_ground(node = 1, R = R2)
    
    manager = mtln.MTLN()
    manager.add_bundle(bundle0)
    

    distances = np.zeros([1, line0.u.shape[0], 3])
    distances[:,:,0] = 0.0508
    ey = wf.null()
    ez = wf.null()
    ex = wf.double_exp_sp(C =5.25e4, a = 4.0e6, b = 4.76e8)
    manager.add_external_field(Field(ex,ey,ez), distances)

    manager.add_network(nw.NetworkD({0:t1}))
    manager.add_network(nw.NetworkD({0:t2}))
    
    manager.run_until(finalTime,dt = 1.9e-10)

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
    
@pytest.mark.skip(reason="External fields are not going to be used")
def test_example_2():
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
    
    term_1 = {'name': 'T1', 
              'materialType': 'MutiwireConnector', 
              'connectorType': ['Conn_R','Conn_R'],
              'resistanceVector': [50,50], 
              'inductanceVector': [0.0,0.0], 
              'capacitanceVector': [1e+22,1e22]}
    term_2 = {'name': 'T2', 
              'materialType': 'MutiwireConnector', 
              'connectorType': ['Conn_R','Conn_R'],
              'resistanceVector': [102,102], 
              'inductanceVector': [0.0,0.0], 
              'capacitanceVector': [1e+22,1e22]}
    
    
    term_1_nw  = {line: {"connections" : [[0,0, term_1], [1,1,term_1]], "side": "S", "bundle" : bundle }}
    term_2_nw  = {line: {"connections" : [[2,0, term_2], [3,1,term_2]], "side": "L", "bundle" : bundle }}

    terminal_1 = nw.Network(term_1_nw)
    terminal_1.connect_to_ground(node = 0, R  = 100)
    terminal_1.connect_to_ground(node = 1, R= 50, Vt = magnitude)
    
    terminal_2 = nw.Network(term_2_nw)
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
    l_wire = 4.8186979E-07
    line_1 = mtl.MTL(l=l_wire, c=c_wire, length=0.120, ndiv=80, name = "line_1")
    line_2 = mtl.MTL(l=l_wire, c=c_wire, length=0.120, ndiv=80, name = "line_2")
    line_4 = mtl.MTL(l=l_wire, c=c_wire, length=0.120, ndiv=80, name = "line_4")
    line_5 = mtl.MTL(l=l_wire, c=c_wire, length=0.120, ndiv=80, name = "line_5")

    assert (line_1.du[0][2] == 1.5e-3)
    assert (line_2.du[0][2] == 1.5e-3)
    assert (line_4.du[0][2] == 1.5e-3)
    assert (line_5.du[0][2] == 1.5e-3)

    c3 = np.zeros([2, 2])
    c3[0] = [2.242e-10, -7.453e-11]
    c3[1] = [-7.453e-11, 2.242e-10]
    l3 = np.zeros([2, 2])
    l3[0] = [4.4712610E-07, 1.4863653E-07]
    l3[1] = [1.4863653E-07, 4.4712610E-07]

    
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
    
    T1 = {'name': 'T1', 
            'materialType': 'Connector', 
            'connectorType': 'Conn_R',
            'resistance' : 50, 
            'inductance' : 0.0, 
            'capacitance': 1e+22}
    T2 = {'name': 'T2', 
            'materialType': 'Connector', 
            'connectorType': 'Conn_R',
            'resistance' : 1e-6, 
            'inductance' : 0.0, 
            'capacitance': 1e+22}
    T3 = {'name': 'T3', 
            'materialType': 'Connector', 
            'connectorType': 'Conn_R',
            'resistance' : 50, 
            'inductance' : 0.0, 
            'capacitance': 1e+22}
    T4 = {'name': 'T4', 
            'materialType': 'Connector', 
            'connectorType': 'Conn_R',
            'resistance' : 50, 
            'inductance' : 0.0, 
            'capacitance': 1e+22}
    J1_L = {'name': 'junction_1_L', 
            'materialType': 'MultiwireConnector', 
            'connectorType': ['Conn_short','Conn_short'],
            'resistanceVector' : [0, 0],
            'inductanceVector' : [0.0, 0.0],
            'capacitanceVector': [1e+22,1e+22 ]}
    J1_S = {'name': 'junction_1_S', 
            'materialType': 'MultiwireConnector', 
            'connectorType': ['Conn_short','Conn_short'],
            'resistanceVector' : [0, 0],
            'inductanceVector' : [0.0, 0.0],
            'capacitanceVector': [1e+22,1e+22 ]}
    J2_L = {'name': 'junction_2_L', 
            'materialType': 'MultiwireConnector', 
            'connectorType': ['Conn_short','Conn_short'],
            'resistanceVector' : [0, 0],
            'inductanceVector' : [0.0, 0.0],
            'capacitanceVector': [1e+22,1e+22 ]}
    J2_S = {'name': 'junction_2_S', 
            'materialType': 'MultiwireConnector', 
            'connectorType': ['Conn_short','Conn_short'],
            'resistanceVector' : [0, 0],
            'inductanceVector' : [0.0, 0.0],
            'capacitanceVector': [1e+22,1e+22 ]}

    
    
    term_1  = {line_1: {"connections" : [[0,0,T1]], "side": "S", "bundle":bundle_1 }}
    term_2  = {line_2: {"connections" : [[1,0,T2]], "side": "S", "bundle":bundle_2 }}

    conn_1  = {line_1: {"connections" : [[2,0, J1_L]], "side": "L", "bundle":bundle_1 },
               line_2: {"connections" : [[3,0, J1_L]], "side": "L", "bundle":bundle_2 },
               line_3: {"connections" : [[4,0, J1_S], [5,1, J1_S]], "side": "S", "bundle":bundle_3}}

    conn_2  = {line_3: {"connections" : [[8,0, J2_L], [9,1, J2_L]], "side": "L", "bundle":bundle_3   },
               line_4: {"connections" : [[11,0, J2_S]], "side": "S", "bundle":bundle_4   },
               line_5: {"connections" : [[10,0, J2_S]], "side": "S", "bundle":bundle_5   }}

    term_3  = {line_5: {"connections" : [[6,0,T3]], "side": "L", "bundle":bundle_5   }}
    term_4  = {line_4: {"connections" : [[7,0, T4]], "side": "L", "bundle":bundle_4   }}

    #terminal 1
    t1 = nw.Network(term_1)
    t1.connect_to_ground(node = 0, R  = 50, Vt=magnitude)
    
    #terminal 2
    t2 = nw.Network(term_2)
    t2.short_to_ground(node = 1)
    
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

    dt = 3.23e-12
    manager.run_until(finalTime, dt = dt)

    t, V1, V2, V3 = np.genfromtxt('python/testData/test4/ex4voltout', delimiter=',',usecols=(0,1,2,3), unpack = True)

    assert(np.allclose(V1[:-1], probe_1.val[:,0], atol = 5e-3))
#     assert(np.allclose(V4_resampled, probe_4.val, atol = 1e-4))
    assert(np.allclose(V3[:-1], probe_5.val[:,0], atol = 5e-3))


#     fig, ax = plt.subplots(3,1)
#     ax[0].plot(1e9*(probe_1.t-dt), probe_1.val, '-.', label = 'probe 1')
#     ax[0].plot(1e9*t, V1, '-.', label = 'probe 1 - mHa')
#     ax[0].set_ylabel(r'$V (t)\,[V]$')
#     ax[0].set_xticks(range(0, 9, 1))
#     ax[0].set_ylim(-0.1,0.6)
#     ax[0].grid('both')
#     ax[0].legend()

#     ax[1].plot(1e9*(probe_4.t-dt), probe_4.val, label = 'probe 4')
#     ax[1].plot(1e9*t, V2, '-.', label = 'probe 2 - mHa')
#     ax[1].set_ylabel(r'$V (t)\,[V]$')
#     ax[1].set_xticks(range(0, 9, 1))
#     ax[1].set_ylim(-0.1,0.6)
#     ax[1].grid('both')
#     ax[1].legend()

#     ax[2].plot(1e9*(probe_5.t-dt), probe_5.val, label = 'probe 5')
#     ax[2].plot(1e9*t, V3, '-.', label = 'probe 3 - mHa')
#     ax[2].set_ylabel(r'$V (t)\,[V]$')
#     ax[2].set_xlabel(r'$t\,[ns]$')
#     ax[2].set_xticks(range(0, 9, 1))
#     ax[2].set_ylim(-0.1,0.6)
#     ax[2].grid('both')
#     ax[2].legend()

#     fig.tight_layout()
#     plt.show()

    
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
    

    Tsh  = {'name': 'Tsh', 
            'materialType': 'Connector', 
            'connectorType': 'Conn_R',
            'resistance' : 50, 
            'inductance' : 0.0, 
            'capacitance': 1e+22}
    Tw   = {'name': 'Tw', 
            'materialType': 'MultiwireConnector', 
            'connectorType': ['Conn_R','Conn_R'],
            'resistanceVector' : [50, 50],
            'inductanceVector' : [0.0, 0.0],
            'capacitanceVector': [1e+22,1e+22 ]}



    term_0_L  = {line_0: {"connections" : [[0,0,Tsh]], "side": "S" , "bundle" : bundle }}
    # term_0_L  = {line_0: {"connections" : [[0,0]], "side": "S" }}
    terminal_0_left = nw.Network(term_0_L)
    terminal_0_left.connect_to_ground(node=0, R = 50)
    
    term_0_R  = {line_0: {"connections" : [[1,0,Tsh]], "side": "L", "bundle" : bundle  }}
    terminal_0_right = nw.Network(term_0_R)
    terminal_0_right.connect_to_ground(node=1, R = 50)
    
    ##### level 1 terminals
    term_1_L = {line_1: {"connections" : [[2,0,Tw], [3,1,Tw]], "side": "S", "bundle" : bundle }}
    terminal_1_left = nw.Network(term_1_L)
    terminal_1_left.connect_to_ground(node=2, R = 50)
    terminal_1_left.connect_to_ground(node=3, R = 50)
    
    term_1_R = {line_1: {"connections" : [[4,0,Tw], [5,1,Tw]], "side": "L", "bundle" : bundle  }}
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

    bundle.add_external_field(mtl.Field(wf.null(), wf.null(), wf.double_exp_xy_sp(C =2*5.25e4, a = 4.0e6, b = 4.76e8)), 
                              distances)


    transferImpedance = {
        "cte" : 0.0,
        "prop" : 4.0e-9,
        "poles" :    {"real" : [], "imag": []},
        "residues" : {"real" : [], "imag": []},
        "direction" : "in"
        }

    #zt01 =  s*yt01 -> yt01 corrs. to the proportional term
    bundle.add_transfer_impedance(out_level=0, out_level_conductors=[0],
                                  in_level=1, in_level_conductors=[0,1],
                                  transfer_impedance=transferImpedance)
    # bundle.add_transfer_impedance(0,1, 0, yt01, np.array([]), np.array([]))

    ##### manager and run

    manager = mtln.MTLN([bundle], [terminal_left, terminal_right])
    manager.run_until(finalTime, dt = 0.5e-10)

    t, _, I = np.genfromtxt('python/testData/test5/ex5s1cout', delimiter = ',',usecols=(0,1,2), unpack = True)
    _, _, V = np.genfromtxt('python/testData/test5/ex5vout', delimiter = ',',usecols=(0,1,2), unpack = True)


    plt.figure()
    plt.plot(1e9*i_probe.t, i_probe.val[:,0], label = 'Current on shield')
    plt.plot(1e9*(t-0.5e-10), I, label = 'Result from manual')

    plt.ylabel(r'$I (t)\,[V]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 40, 10))
    plt.grid('both')
    plt.legend()

    plt.figure()
    plt.plot(1e9*v_probe.t, v_probe.val[:,1], label = 'Voltage on inner conductor 1')
#     plt.plot(1e9*v_probe.t, v_probe.val[:,2], label = 'Voltage on inner conductor 2')
    plt.plot(1e9*(t-0.5e-10), V, label = 'Result from manual')
    plt.ylabel(r'$V (t)\,[V]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 40, 10))
    plt.grid('both')
    plt.legend()
    
    plt.show()


def test_6():
    """Agrawal"""
    
    finalTime = 70e-9
    # lines and bundles
    ## level 0
    c0 = np.array([[7.7864194E-11  ,-3.6387251E-11, -3.6349781E-11],
                  [-3.6349750E-11 , 7.8139752E-11, -3.6234689E-11],
                  [-3.6387251E-11 ,-3.6197323E-11,  7.8139696E-11]])
    l0 = np.array([[8.2171925E-07  ,6.5145252E-07  ,6.5145258E-07],
                  [6.5145264E-07  ,8.1376146E-07  ,6.4746973E-07],
                  [6.5145252E-07  ,6.4747002E-07  ,8.1376146E-07]])
    
    line_0 = mtl.MTL(l=l0, c=c0, length=10, ndiv=300, name = "line_0")
    
    bundle = mtl.MTLD({0:[line_0]}, name = "bundle")
    
    T   = {'name': 'T', 
            'materialType': 'MultiwireConnector', 
            'connectorType': ['Conn_R','Conn_R', 'Conn_R'],
            'resistanceVector' : [50, 50, 50],
            'inductanceVector' : [0.0, 0.0, 0.0],
            'capacitanceVector': [1e+22,1e+22, 1e+22 ]}

    # networks
    ## level 0
    term_0_L  = {line_0: {"connections" : [[0,0,T], [1,1,T], [2,2,T]], "side": "S" , "bundle" : bundle }}
    term_0_R  = {line_0: {"connections" : [[3,0,T], [4,1,T], [5,2,T]], "side": "L" , "bundle" : bundle }}

    terminal_0_left = nw.Network(term_0_L)
    terminal_0_right = nw.Network(term_0_R)

    def pulse(t): return wf.sin_sq_pulse(t, 1.0, 1.0472e9)
    terminal_0_left.connect_to_ground(node=0, R = 50, Vt = pulse)
    terminal_0_left.connect_to_ground(node=1, R = 50)
    terminal_0_left.connect_to_ground(node=2, R = 50)

    terminal_0_right.connect_to_ground(node=3, R = 50)
    terminal_0_right.connect_to_ground(node=4, R = 50)
    terminal_0_right.connect_to_ground(node=5, R = 50)


    terminal_left   = nw.NetworkD({0:terminal_0_left})
    terminal_right  = nw.NetworkD({0:terminal_0_right})

    #manager

    v_probe = bundle.add_probe(position = 10.0, probe_type = "voltage")

    manager = mtln.MTLN([bundle], [terminal_left, terminal_right])
    
    manager.run_until(finalTime, dt = 0.5e-10)

    t, V4, _, V5, _, V6 = np.genfromtxt('python/testData/ngspice/manual/agrawal.txt', usecols=(0,1,2,3,4,5), unpack = True)

    fig, ax = plt.subplots(3,1)
    ax[0].plot(1e9*v_probe.t, v_probe.val[:,0], 'r-.', label = 'V at end of wire 1')
    ax[0].plot(1e9*t, V5, 'g-.', label = 'V at end of wire 1, Spice')
    ax[0].set_ylabel(r'$V (t)\,[V]$')
    ax[0].set_xticks(range(0, 70, 10))
    ax[0].set_ylim(-0.2,0.4)
    ax[0].grid('both')
    ax[0].legend()

    ax[1].plot(1e9*v_probe.t, v_probe.val[:,1], 'r-.', label = 'V at end of wire 2')
    ax[1].plot(1e9*t, V4, 'g-.', label = 'V at end of wire 2, Spice')
    ax[1].set_ylabel(r'$V (t)\,[V]$')
    ax[1].set_xticks(range(0, 70, 10))
    ax[1].set_ylim(-0.2,0.4)
    ax[1].grid('both')
    ax[1].legend()

    ax[2].plot(1e9*v_probe.t, v_probe.val[:,2], 'r-.', label = 'V at end of wire 3')
    ax[2].plot(1e9*t, V6, 'g-.', label = 'V at end of wire 3, Spice')
    ax[2].set_ylabel(r'$V (t)\,[V]$')
    ax[2].set_xlabel(r'$t\,[ns]$')
    ax[2].set_xticks(range(0, 70, 10))
    ax[2].set_ylim(-0.2,0.4)
    ax[2].grid('both')
    ax[2].legend()

    fig.tight_layout()
    plt.show()
   

    
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
    
    T1 = {'name': 'T1', 
              'materialType': 'MutiwireConnector', 
              'connectorType': ['Conn_R','Conn_R'],
              'resistanceVector': [50,50], 
              'inductanceVector': [0.0,0.0], 
              'capacitanceVector': [1e+22,1e22]}
    T2 = {'name': 'T2', 
              'materialType': 'MutiwireConnector', 
              'connectorType': ['Conn_R','Conn_R'],
              'resistanceVector': [102,102], 
              'inductanceVector': [0.0,0.0], 
              'capacitanceVector': [1e+22,1e22]}

    
    term_1  = {line: {"connections" : [[0,0,T1], [1,1,T1]], "side": "S", "bundle" : bundle }}
    term_2  = {line: {"connections" : [[2,0,T2], [3,1,T2]], "side": "L", "bundle" : bundle }}

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
    