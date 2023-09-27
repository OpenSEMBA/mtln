import numpy as np
import matplotlib.pyplot as plt

from src.jsonParser import Parser

import src.waveforms as wf
from src.mtl import Field

# import pickle
# import dill

def test_parse_file():
    file = 'python/testData/parser/cable_bundle_and_two_wires.smb.json'
    p = Parser(file)
    assert '_format' in p.parsed.keys()
    assert '_version' in p.parsed.keys()
    
def test_get_nodes():
    file = 'python/testData/parser/cable_bundle_and_two_wires.smb.json'
    p = Parser(file)
    nodes = p.getNodesInJunction("j1")
    assert(type(nodes) == list)
    assert(len(nodes) == 2)    
    assert(nodes[0] == [2, 5])
    assert(nodes[1] == [4, 7])
    
def test_build_networks():
    file = 'python/testData/parser/cable_bundle_and_two_wires.smb.json'
    p = Parser(file)
    # assert set([2,5,4,7]).issubset(p.networks[0].nodes)
    assert p._networks[0].P1.shape == (4,4)
    assert p._networks[0].P1[0,1] == 0.0
    assert np.abs(p._networks[0].P1[0,0]) == 1e10
    
    assert p._networks[1].P1.shape == (1,1)
    assert p._networks[1].P1[0,0] == -1.0/50.
    
    assert p._networks[2].P1.shape == (2,2)
    assert p._networks[2].P1[0,1] == 0.0
    assert p._networks[2].P1[1,0] == 0.0
    assert p._networks[2].P1[0,0] == -1.0/9.2e-2
    assert p._networks[2].P1[1,1] == -1.0/9.0e-2
    
    assert p._networks[3].P1.shape == (1,1)
    assert p._networks[3].P1[0,0] == -1.0/50

    file = 'python/testData/parser/ribbon_cable_20ns_termination_network.smb.json'
    p = Parser(file)

    assert p._networks[0].P1.shape == (2,2)
    assert p._networks[0].P1[0,1] == 0.0
    assert p._networks[0].P1[1,0] == 0.0
    assert p._networks[0].P1[0,0] == -1.0/50
    assert p._networks[0].P1[1,1] == -1.0/50

    assert p._networks[1].P1.shape == (2,2)
    assert p._networks[1].P1[0,1] == 0.0
    assert p._networks[1].P1[1,0] == 0.0
    assert p._networks[1].P1[0,0] == -1.0/50
    assert p._networks[1].P1[1,1] == -1.0/50
    
    
def test_ribbon_cable_20_ns_termination_network():
    """
     _             _
    | |    c1     | |
    | 2-----------4 |
    | |    c0     | |
    | 1-----------3 |
    |_|           |_|
    T1             T2
    
    """
    file = 'python/testData/parser/ribbon_cable_20ns_termination_network.smb.json'
    p = Parser(file)
    p.run(finalTime = 200e-9)
    assert (np.isclose(np.max(p.probes["v0"].val[:, 0]), 113e-3, atol=1e-3))
    
def test_ribbon_cable_1ns_paul_interconnection_network():
    """
     _             _             _
    | |     1     | |     1     | |
    | 2-----------4-6-----------8 |
    | |     b0    | |     b1    | |
    | 1-----------3-5-----------7 |
    |_|     0     |_|     0     |_|
    term_1(0)     iconn(1)     term_2(2)
    
    """
    file = 'python/testData/parser/ribbon_cable_1ns_paul_interconnection_network.smb.json'
    p = Parser(file)
    p.run(finalTime = 200e-9)
    

    times = [12.5, 25, 40, 55]
    voltages = [120, 95, 55, 32]
    for (t, v) in zip(times, voltages):
        index = np.argmin(np.abs(p.probes["v0"].t - t*1e-9))
        assert np.all(np.isclose(p.probes["v0"].val[index, 0], v*1e-3, atol=10e-3))


def test_ribbon_cable_1ns_R_interconnection_network():
    """
     _             __________             _
    | |     1     |          |     1     | |
    | 2-----------4--R-------6-----------8 |
    | |  bundle_0 |          |  bundle_0 | |
    | 1-----------3--R-------5-----------7 |
    |_|     0     |__________|     0     |_|
    t1                  j1                t2
    
    """
    file = 'python/testData/parser/ribbon_cable_1ns_R_interconnection_network.smb.json'
    p = Parser(file)
    p.run(finalTime=200e-9)

    t0, V0 = np.genfromtxt('python/testData/ngspice/test_ribbon_cable_1ns_R_interconnection_network/V1.txt', delimiter=',', usecols=(0,1), unpack = True)
    t1, V1 = np.genfromtxt('python/testData/ngspice/test_ribbon_cable_1ns_R_interconnection_network/V2.txt', delimiter=',', usecols=(0,1), unpack = True)

    V0_resampled = np.interp(p.probes["v0"].t, t0, V0)
    V1_resampled = np.interp(p.probes["v0"].t, t1, V1)

    assert(np.allclose(V0_resampled[:-1], p.probes["v0"].val[1:,0], atol = 0.01, rtol=0.05))
    assert(np.allclose(V1_resampled[:-1], p.probes["v0"].val[1:,1], atol = 0.01, rtol=0.05))

def test_ribbon_cable_1ns_RV_interconnection_network():
    """
     _             __________             _
    | |     1     |          |     1     | |
    | 2-----------4--R--V35--6-----------8 |
    | |  bundle_0 |          |  bundle_0 | |
    | 1-----------3--R-------5-----------7 |
    |_|     0     |__________|     0     |_|
    t1                  j1                t2
    
    """

    file = 'python/testData/parser/ribbon_cable_1ns_RV_interconnection_network.smb.json'
    p = Parser(file)
    p.run(finalTime=200e-9)


    t0, V0 = np.genfromtxt('python/testData/ngspice/test_ribbon_cable_1ns_RV_interconnection_network/V1.txt', delimiter=',', usecols=(0,1), unpack = True)
    t1, V1 = np.genfromtxt('python/testData/ngspice/test_ribbon_cable_1ns_RV_interconnection_network/V2.txt', delimiter=',', usecols=(0,1), unpack = True)

    V0_resampled = np.interp(p.probes["v0"].t, t0, V0)
    V1_resampled = np.interp(p.probes["v0"].t, t1, V1)

    assert(np.allclose(V0_resampled[:-1], p.probes["v0"].val[1:,0], atol = 0.01, rtol=0.05))
    assert(np.allclose(V1_resampled[:-1], p.probes["v0"].val[1:,1], atol = 0.01, rtol=0.05))

    # plt.plot(1e9*p.probes["v0"].t, p.probes["v0"].val[:,0] ,'r', label = 'Conductor 0')
    # plt.plot(1e9*p.probes["v0"].t, p.probes["v0"].val[:,1] ,'b', label = 'Conductor 1')
    # plt.plot(1e9*t0, V0 ,'g--', label = 'Conductor 0 - NgSpice')
    # plt.plot(1e9*t1, V1 ,'k--', label = 'Conductor 1 - NgSpice')
    # plt.ylabel(r'$V (0, t)\,[V]$')
    # plt.xlabel(r'$t\,[ns]$')
    # plt.xticks(range(0, 200, 50))
    # plt.grid('both')
    # plt.legend()
    # plt.show()


def test_ribbon_cable_1ns_RV_T_network():
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

    file = 'python/testData/parser/ribbon_cable_1ns_RV_T_network.smb.json'
    p = Parser(file)
    p.run(finalTime=200e-9)


    t0, V0 = np.genfromtxt('python/testData/ngspice/test_ribbon_cable_1ns_RV_T_network/V1.txt', delimiter=',', usecols=(0,1), unpack = True)
    t1, V1 = np.genfromtxt('python/testData/ngspice/test_ribbon_cable_1ns_RV_T_network/V2.txt', delimiter=',', usecols=(0,1), unpack = True)

    V0_resampled = np.interp(p.probes["v0"].t, t0, V0)
    V1_resampled = np.interp(p.probes["v0"].t, t1, V1)

    assert(np.allclose(V0_resampled[:-1], p.probes["v0"].val[1:,0], atol = 0.01, rtol=0.05))
    assert(np.allclose(V1_resampled[:-1], p.probes["v0"].val[1:,1], atol = 0.01, rtol=0.05))

    # plt.plot(1e9*p.probes["v0"].t, p.probes["v0"].val[:,0] ,'r', label = 'Conductor 0')
    # plt.plot(1e9*p.probes["v0"].t, p.probes["v0"].val[:,1] ,'b', label = 'Conductor 1')
    # plt.plot(1e9*t0, V0 ,'g--', label = 'Conductor 0 - NgSpice')
    # plt.plot(1e9*t1, V1 ,'k--', label = 'Conductor 1 - NgSpice')
    # plt.ylabel(r'$V (0, t)\,[V]$')
    # plt.xlabel(r'$t\,[ns]$')
    # plt.xticks(range(0, 200, 50))
    # plt.grid('both')
    # plt.legend()
    # plt.show()

def test_1_double_exponential():
        
    """
    External field: double exponential 
     _________              __________
    |         |            |          |          
    | g--R1---0------------1---R2--g  |                
    |_________|     b0     |__________|                    
    """
    
    file = 'python/testData/parser/1.smb.json'
    p = Parser(file)

    distances = np.zeros([1, 51, 3])
    distances[:,:,0] = 0.0508
    p.runWithExternalField(
        finalTime=80e-9, 
        field= Field(wf.double_exp_sp(5.25e4, 4.0e6,4.76e8),wf.null(),wf.null()), 
        distances = distances)

    fig, ax1 = plt.subplots()
    ax1.set_xlabel(r'$t\,[ns]$')
    ax1.set_ylabel(r'$V (t)\,[V]$', color='tab:red')
    ax1.plot(1e9*p.probes["v0"].t, p.probes["v0"].val, color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_xticks(range(0, 80, 20))

    ax2 = ax1.twinx() 
    ax2.set_ylabel(r'$I (t)\,[A]$', color='tab:blue') 
    ax2.plot(1e9*p.probes["i0"].t, p.probes["i0"].val, color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    fig.tight_layout() 
    plt.show()


def test_3_two_conductor_line():
    """
     _             _
    | |     1     | |
    | 1-----------3 |
    | |     0     | |
    | 0-----------2 |
    |_|           |_|
    term_1(0)     term_2(1)
    
    """
    file = 'python/testData/parser/3.smb.json'
    p = Parser(file)
    p.run(finalTime=5e-9)
    
    plt.figure()
    plt.plot(1e9*p.probes["vR"].t, p.probes["vR"].val[:,1], label = 'End2 of C1')
    plt.ylabel(r'$V (t)\,[V]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 6, 1))
    plt.xlim(0,4)
    plt.grid('both')
    plt.legend()

    plt.figure()
    plt.plot(1e9*p.probes["vL"].t, p.probes["vL"].val[:,0], label = 'End1 of C2')
    plt.plot(1e9*p.probes["vR"].t, p.probes["vR"].val[:,0], label = 'End2 of C2')
    plt.ylabel(r'$V (t)\,[V]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 6, 1))
    plt.grid('both')
    plt.legend()
    
    plt.show()

    
def test_4_lines_multilines():
    
    """
     _     l2      _     _     l4       _
    | |     b1    | | 1 | |    b3      | |
    |_2-----------4-6---10-12----------8_|
    t2            | |   | |            t4
                  | |b2 | |
     _      b0    | | 0 | |    b4      _
    | 1-----------3-5---9-11----------7 |
    |_|     l1    |_| l3|_|     l5    |_|
    t1            j1    j2             t3 
    
    """
    file = 'python/testData/parser/4.smb.json'
    p = Parser(file)
    p.run(finalTime=6.46e-9)
    
    fig, ax = plt.subplots(3,1)
    ax[0].plot(1e9*p.probes["v1"].t, p.probes["v1"].val, '-.', label = 'probe 1')
    ax[0].set_ylabel(r'$V (t)\,[V]$')
    ax[0].set_xticks(range(0, 9, 1))
    ax[0].set_ylim(-0.1,0.6)
    ax[0].grid('both')
    ax[0].legend()

    ax[1].plot(1e9*p.probes["v8"].t, p.probes["v8"].val, label = 'probe 4')
    ax[1].set_ylabel(r'$V (t)\,[V]$')
    ax[1].set_xticks(range(0, 9, 1))
    ax[1].set_ylim(-0.1,0.6)
    ax[1].grid('both')
    ax[1].legend()

    ax[2].plot(1e9*p.probes["v7"].t, p.probes["v7"].val, label = 'probe 5')
    ax[2].set_ylabel(r'$V (t)\,[V]$')
    ax[2].set_xlabel(r'$t\,[ns]$')
    ax[2].set_xticks(range(0, 9, 1))
    ax[2].set_ylim(-0.1,0.6)
    ax[2].grid('both')
    ax[2].legend()

    fig.tight_layout()
    plt.show()

    
def test_5_coaxial():
    """
    
      -------------------
     |  \                   \
     R   \                   \
     ·---|                   |
     ·---|                   |
     R   /                   /
     |  /                   /
     --------------------
     |                   |
     R                   R
  ___|___________________|___
     
    """ 
    
    file = 'python/testData/parser/5.smb.json'
    p = Parser(file)

    distances = np.zeros([1, 19, 3])
    distances[:,:,0] = 0.0508
    p.runWithExternalField(
        finalTime=30e-9, 
        field= Field(wf.null(), wf.null(), wf.double_exp_xy_sp(C =5.25e4, a = 4.0e6, b = 4.76e8)), 
        distances = distances)
    
    plt.figure()
    plt.plot(1e9*p.probes["i"].t, p.probes["i"].val[:,0], label = 'Current on shield')
    plt.ylabel(r'$I (t)\,[V]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 40, 10))
    plt.yticks(range(-2, 10, 2))
    plt.grid('both')
    plt.legend()

    plt.figure()
    plt.plot(1e9*p.probes["v"].t, p.probes["v"].val[:,1], label = 'Voltage on inner conductor 1')
    plt.plot(1e9*p.probes["v"].t, p.probes["v"].val[:,2], label = 'Voltage on inner conductor 2')
    plt.ylabel(r'$V (t)\,[V]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 40, 10))
    plt.yticks(range(-1, 4, 1))
    plt.grid('both')
    plt.legend()
    
    plt.show()

def test_6_agrawal():
    file = 'python/testData/parser/6.smb.json'
    p = Parser(file)
    p.run(finalTime=7.0e-8)
    
    fig, ax = plt.subplots(3,1)
    ax[0].plot(1e9*p.probes["v"].t, 1e3*p.probes["v"].val[:,2], '-.', label = 'V at end of wire 1')
    ax[0].set_ylabel(r'$V (t)\,[mV]$')
    ax[0].set_xticks(range(0, 70, 10))
    ax[2].set_yticks(range(-200, 500, 100))
    ax[0].grid('both')
    ax[0].legend()

    ax[1].plot(1e9*p.probes["v"].t, 1e3*p.probes["v"].val[:,1], label = 'V at end of wire 2')
    ax[1].set_ylabel(r'$V (t)\,[mV]$')
    ax[1].set_xticks(range(0, 70, 10))
    ax[2].set_yticks(range(-200, 500, 100))
    ax[1].grid('both')
    ax[1].legend()

    ax[2].plot(1e9*p.probes["v"].t, 1e3*p.probes["v"].val[:,0], label = 'V at end of wire 3')
    ax[2].set_ylabel(r'$V (t)\,[mV]$')
    ax[2].set_xlabel(r'$t\,[ns]$')
    ax[2].set_xticks(range(0, 70, 10))
    ax[2].set_yticks(range(-200, 500, 100))
    ax[2].grid('both')
    ax[2].legend()

    fig.tight_layout()
    plt.show()

    fig.tight_layout()
    plt.show()
    
def test_7_bundles():
    file = 'python/testData/parser/7.smb.json'
    p = Parser(file)
    p.run(finalTime=0.5e-7,dt=0.5e-10)

    plt.figure()
    plt.plot(1e9*p.probes["i_sh21c1"].t, p.probes["i_sh21c1"].val[:,0], label = 'Current on shield')
    plt.ylabel(r'$I (t)\,[A]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 60, 10))
    # plt.yticks(range(0, 1.2, ))
    # plt.ylim(0,1)
    plt.grid('both')
    plt.legend()
    plt.figure()
    
    plt.plot(1e9*p.probes["v_s1s4c2"].t, p.probes["v_s1s4c2"].val[:,1], label = 'Voltage on bundle 4, level 2, conductor 1')
    plt.ylabel(r'$V (t)\,[V]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 60, 10))
    # plt.yticks(range(0, 1.2, ))
    # plt.ylim(0,1)
    plt.grid('both')
    plt.legend()
    
    plt.figure()
    plt.plot(1e9*p.probes["v_s1s2c2"].t, p.probes["v_s1s2c2"].val[:,1], label = 'Voltage on bundle 2, level 2, conductor 1')
    plt.ylabel(r'$V (t)\,[V]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 60, 10))
    # plt.yticks(range(0, 1.2, ))
    # plt.ylim(0,1)
    plt.grid('both')
    plt.legend()
    plt.show()


def test_8_two_conductor_line_with_source():
    """
     _             _
    | |     1     | |
    | 1-----------3 |
    | |     0     | |
    | 0-----------2 |
    |_|           |_|
    term_1(0)     term_2(1)
    
    """
    file = 'python/testData/parser/8.smb.json'
    p = Parser(file)
    p.run(finalTime=5e-9)
    
    plt.figure()
    plt.plot(1e9*p.probes["vR"].t, p.probes["vR"].val[:,1], label = 'End2 of C1')
    plt.ylabel(r'$V (t)\,[V]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 6, 1))
    plt.yticks(range(-3, 4, 1))
    plt.xlim(0,4)
    plt.grid('both')
    plt.legend()

    plt.figure()
    plt.plot(1e9*p.probes["vL"].t, p.probes["vL"].val[:,0], label = 'End1 of C2')
    plt.plot(1e9*p.probes["vR"].t, p.probes["vR"].val[:,0], label = 'End2 of C2')
    plt.ylabel(r'$V (t)\,[V]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 6, 1))
    plt.yticks(range(-3, 4, 1))
    plt.grid('both')
    plt.legend()
    
    plt.show()

    
    
