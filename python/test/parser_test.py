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
    
    file = 'python/testData/parser/test_1_double_exponential.smb.json'
    p = Parser(file)

    distances = np.zeros([1, 51, 3])
    distances[:,:,2] = 0.0508
    p.runWithExternalField(
        finalTime=80e-9, 
        dt = 0,
        field= Field(wf.null(),wf.null(),wf.double_exp_sp(5.25e4, 4.0e6,4.76e8)), 
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
    file = 'python/testData/parser/test_3_two_conductor_line.smb.json'
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
    file = 'python/testData/parser/test_4_multilines.smb.json'
    p = Parser(file)
    p.run(finalTime=6.46e-9, dt = 3.23e-12)
    
    t, V1, V2, V3 = np.genfromtxt('python/testData/test4/ex4voltout', delimiter=',',usecols=(0,1,2,3), unpack = True)
    
    fig, ax = plt.subplots(3,1)
    ax[0].plot(1e9*p.probes["v_s1"].t, p.probes["v_s1"].val, '-.', label = 'probe 1')
    ax[0].plot(1e9*t, V1, '-.', label = 'probe 1 - mHa')
    ax[0].set_ylabel(r'$V (t)\,[V]$')
    ax[0].set_xticks(range(0, 9, 1))
    ax[0].set_ylim(-0.1,0.6)
    ax[0].grid('both')
    ax[0].legend()

    ax[1].plot(1e9*p.probes["v_s5"].t, p.probes["v_s5"].val, label = 'probe 4')
    ax[1].plot(1e9*t, V2, '-.', label = 'probe 2 - mHa')
    ax[1].set_ylabel(r'$V (t)\,[V]$')
    ax[1].set_xticks(range(0, 9, 1))
    ax[1].set_ylim(-0.1,0.6)
    ax[1].grid('both')
    ax[1].legend()

    ax[2].plot(1e9*p.probes["v_s4"].t, p.probes["v_s4"].val, label = 'probe 5')
    ax[2].plot(1e9*t, V3, '-.', label = 'probe 3 - mHa')
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
    
    file = 'python/testData/parser/test_5_coaxial.smb.json'
    p = Parser(file)

    distances = np.zeros([1, 19, 3])
    distances[:,:,0] = 0.0508
    p.runWithExternalField(
        finalTime=30e-9, 
        dt = 0,
        field= Field(wf.null(), wf.null(), wf.double_exp_xy_sp(C =10.5e4, a = 4.0e6, b = 4.76e8)), 
        distances = distances)
    
    t, _, I = np.genfromtxt('python/testData/test5/ex5s1cout', delimiter = ',',usecols=(0,1,2), unpack = True)
    _, _, V = np.genfromtxt('python/testData/test5/ex5vout', delimiter = ',',usecols=(0,1,2), unpack = True)

    
    plt.figure()
    plt.plot(1e9*p.probes["i"].t, p.probes["i"].val[:,0], label = 'Current on shield')
    plt.plot(1e9*t, I, label = 'Result from manual')
    plt.ylabel(r'$I (t)\,[A]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 40, 10))
    plt.grid('both')
    plt.legend()

    plt.figure()
    plt.plot(1e9*p.probes["v"].t, p.probes["v"].val[:,2], label = 'Voltage on inner conductor 2')
    plt.plot(1e9*t, V, label = 'Result from manual')
    plt.ylabel(r'$V (t)\,[V]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 40, 10))
    plt.grid('both')
    plt.legend()
    
    plt.show()

def test_6_agrawal():
    file = 'python/testData/parser/test_6_agrawal.smb.json'
    p = Parser(file)
    p.run(finalTime=7.0e-8, dt = 0.5e-10)
    			# 	"inductanceMatrix": [
                #     [0.8792e-6, 0.6791e-6, 0.6787e-6],
                #     [0.6775e-6, 0.8827e-6, 0.6785e-6],
                #     [0.6760e-6, 0.6774e-6, 0.8829e-6]
                # ],
				# "capacitanceMatrix": [
                #     [75.54e-12, -34.57e-12,-34.92e-12],
                #     [-34.63e-12, 73.87e-12, -33.96e-12],
                #     [-34.29e-12, -34.0e-12, 73.41e-12]
                # ]

    t, V4, _, V5, _, V6 = np.genfromtxt('python/testData/ngspice/manual/agrawal.txt', usecols=(0,1,2,3,4,5), unpack = True)

    fig, ax = plt.subplots(3,1)
    ax[0].plot(1e9*p.probes["v"].t, p.probes["v"].val[:,2], 'r-.', label = 'V at end of wire 1')
    ax[0].plot(1e9*t, V5, 'g-.', label = 'V at end of wire 1, Spice')
    ax[0].set_ylabel(r'$V (t)\,[V]$')
    ax[0].set_xticks(range(0, 70, 10))
    ax[0].set_ylim(-0.2,0.4)
    ax[0].grid('both')
    ax[0].legend()

    ax[1].plot(1e9*p.probes["v"].t, p.probes["v"].val[:,1], 'r-.', label = 'V at end of wire 2')
    ax[1].plot(1e9*t, V4, 'g-.', label = 'V at end of wire 2, Spice')
    ax[1].set_ylabel(r'$V (t)\,[V]$')
    ax[1].set_xticks(range(0, 70, 10))
    ax[1].set_ylim(-0.2,0.4)
    ax[1].grid('both')
    ax[1].legend()

    ax[2].plot(1e9*p.probes["v"].t, p.probes["v"].val[:,1], 'r-.', label = 'V at end of wire 3')
    ax[2].plot(1e9*t, V6, 'g-.', label = 'V at end of wire 3, Spice')
    ax[2].set_ylabel(r'$V (t)\,[V]$')
    ax[2].set_xlabel(r'$t\,[ns]$')
    ax[2].set_xticks(range(0, 70, 10))
    ax[2].set_ylim(-0.2,0.4)
    ax[2].grid('both')
    ax[2].legend()

    fig.tight_layout()
    plt.show()

    
def test_7_bundles():
    # file = 'python/testData/parser/test_7_bundles.smb.json'
    file = 'python/testData/parser/test_7_bundles_single_conductors.smb.json'
    # file = 'python/testData/parser/test_7_bundles_T1_T1.smb.json'
    # file = 'python/testData/parser/test_7_bundles_Level_0.smb.json'
    # file = 'python/testData/parser/test_7_bundles_Level_1.smb.json'
    # file = 'python/testData/parser/test_7_bundles_Level_1_single_bundle.smb.json'

    dt = 0.5e-10
    p = Parser(file)
    p.run(finalTime=50e-9,dt=dt)   

    t, Ish2_0, Ish2_1 = np.genfromtxt('python/testData/bundles_test/ex7s2cout_probes', delimiter=',', usecols=(0,1,2), unpack = True)
    _ ,Ish1_0, Ish1_1 = np.genfromtxt('python/testData/bundles_test/ex7s1cout_probes', delimiter=',', usecols=(0,1,2), unpack = True)
    I1, I2 = np.genfromtxt('python/testData/bundles_test/ex7curout_probes', delimiter=',', usecols=(1,2), unpack = True)
    
    fig, ax = plt.subplots(3,1)
    ax[0].plot(1e9*(p.probes["i_sh21c1"].t-2.0*dt), 1e3*p.probes["i_sh21c1"].val[:,0], label = 'Current on bundle1, outer shield. x = 0.1')
    ax[0].plot(1e9*t, 1e3*Ish2_1, 'r--', label = 'result from manual ')
    ax[0].set_ylabel(r'$I (t)\,[mA]$')
    ax[0].set_xlabel(r'$t\,[ns]$')
    ax[0].set_xticks(range(0, 60, 10))
    ax[0].grid('both')
    ax[0].legend()
    
    ax[1].plot(1e9*(p.probes["i_sh21c1"].t-2.0*dt), 1e6*p.probes["i_sh21c1"].val[:,1], label = 'Current on bundle1, inner shield, x = 0.1')
    ax[1].plot(1e9*t, 1e6*Ish1_1, 'r--', label = 'result from manual ')
    ax[1].set_ylabel(r'$I (t)\,[mA]$')
    ax[1].set_xlabel(r'$t\,[ns]$')
    ax[1].set_xticks(range(0, 60, 10))
    ax[1].grid('both')
    ax[1].legend()
    
    # ax[1].plot(1e9*p.probes["cable_1_terminal_voltage"].t, 1e6*50*p.probes["cable_1_terminal_current"].val[:,8], label = 'cable 1 - s1s2c2 - I')
    # ax[1].plot(1e9*t, 1e6*50*I1, 'r--', label = 'result from manual ')
    # ax[1].set_ylabel(r'$V (t)\,[\mu V]$')
    # ax[1].set_xlabel(r'$t\,[ns]$')
    # ax[1].set_xticks(range(0, 60, 10))
    # ax[1].grid('both')
    # ax[1].legend()

    # ax[2].plot(1e9*p.probes["cable_1_terminal_voltage"].t, -1e6*p.probes["cable_1_terminal_voltage"].val[:,6], label = 'cable 1 - s1s4c2 -V')
    ax[2].plot(1e9*p.probes["cable_1_terminal_voltage"].t, 1e6*50*p.probes["cable_1_terminal_current"].val[:,6], label = 'cable 1 - s1s4c2-I')
    ax[2].plot(1e9*t, 1e6*50*I2, 'r--', label = 'result from manual ')
    ax[2].set_ylabel(r'$V (t)\,[\mu V]$')
    ax[2].set_xlabel(r'$t\,[ns]$')
    ax[2].set_xticks(range(0, 60, 10))
    # ax[2].set_ylim(-120,120)
    ax[2].grid('both')
    ax[2].legend()
    

    plt.show()
    
def test_7_bundles_B1_B2_B3():
    file = 'python/testData/parser/test_7_bundles_single_conductors_B1_B2_B3.smb.json'

    dt = 0.5e-10
    p = Parser(file)
    p.run(finalTime=50e-9,dt=dt)   

    # dt = 0.0

    t, I_b1_sh2_terminal,I_b1_sh2_mid,I_b1_sh2_junction, I_b2_sh2_junction,I_b2_sh2_connector, I_b2_sh2_terminal, I_b3_sh2_junction,I_b3_sh2_terminal =\
        np.genfromtxt('python/testData/bundles_test/B1_B2_B3/shield_2_current_B1_B2_B3', delimiter=',', usecols=(0,1,2,3,4,5,6,7,8), unpack = True)
    
    _, I_b1_sh1_terminal,I_b1_sh1_mid,I_b1_sh1_junction, I_b2_sh1_junction,I_b2_sh1_connector, I_b2_sh1_terminal, I_b3_sh1_junction,I_b3_sh1_terminal =\
        np.genfromtxt('python/testData/bundles_test/B1_B2_B3/shield_1_current_B1_B2_B3', delimiter=',', usecols=(0,1,2,3,4,5,6,7,8), unpack = True)
    
    _, I_b1_s1s2c2_terminal,I_b1_s1s2c2_junction, I_b1_s1s4c2_terminal, I_b1_s1s4c2_junction, I_b2_s1s2c2_junction, I_b2_s1s2c2_connector,I_b2_s1s2c2_terminal =\
        np.genfromtxt('python/testData/bundles_test/B1_B2_B3/cable_current_B1_B2_B3', delimiter=',', usecols=(0,1,2,3,4,5,6,7), unpack = True)
    
    
    # I_b1_cable_1, I_b1_cable_2, I_b3_cable = [], [], []
    # _, I_b1_cable_1[0:2], I_b1_cable_2[0:2], I_b3_cable[0:3] = np.genfromtxt('python/testData/bundles_test/B1_B2_B3/cable_current_B1_B2_B3', delimiter=',', usecols=(1,2), unpack = True)
    
    fig, ax = plt.subplots(2,1)
    fig.suptitle("Bundle S1 - shield 2")
    ax[0].plot(1e9*(p.probes["b1_terminal_current"].t-1.5*dt), 1e3*p.probes["b1_terminal_current"].val[:,0], label = 'Current on bundle1, outer shield. x = 0')
    ax[0].plot(1e9*t, 1e3*I_b1_sh2_terminal, 'r--', label = 'result from manual ')

    # ax[1].plot(1e9*(p.probes["b1_mid_current"].t-1.5*dt), 1e3*p.probes["b1_mid_current"].val[:,0], label = 'Current on bundle1, outer shield. x = 0.1')
    # ax[1].plot(1e9*t, 1e3*I_b1_sh2_mid, 'r--', label = 'result from manual ')

    ax[1].plot(1e9*(p.probes["b1_junction_current"].t-1.5*dt), 1e3*p.probes["b1_junction_current"].val[:,0], label = 'Current on bundle1, outer shield. x = 0.54')
    ax[1].plot(1e9*t, 1e3*I_b1_sh2_junction, 'r--', label = 'result from manual ')

    for i in range(2):
        ax[i].set_ylabel(r'$I (t)\,[mA]$')
        ax[i].set_xlabel(r'$t\,[ns]$')
        ax[i].set_xticks(range(0, 60, 10))
        ax[i].grid('both')
        ax[i].legend()

    fig, ax = plt.subplots(2,1)
    fig.suptitle("Bundle S2 - shield 2")
    ax[0].plot(1e9*(p.probes["b2_junction_current"].t-1.5*dt), 1e3*p.probes["b2_junction_current"].val[:,0], label = 'Current on bundle1, outer shield. x = 0.0')
    ax[0].plot(1e9*t, 1e3*I_b2_sh2_junction, 'r--', label = 'result from manual ')

    # ax[1].plot(1e9*(p.probes["b2_connector_current"].t-1.5*dt), 1e3*p.probes["b2_connector_current"].val[:,0], label = 'Current on bundle2, outer shield. x = 0.3')
    # ax[1].plot(1e9*t, 1e3*I_b2_sh2_connector, 'r--', label = 'result from manual ')

    ax[1].plot(1e9*(p.probes["b2_terminal_current"].t-1.5*dt), 1e3*p.probes["b2_terminal_current"].val[:,0], label = 'Current on bundle2, outer shield. x = 0.343')
    ax[1].plot(1e9*t, 1e3*I_b2_sh2_terminal, 'r--', label = 'result from manual ')


    for i in range(2):
        ax[i].set_ylabel(r'$I (t)\,[mA]$')
        ax[i].set_xlabel(r'$t\,[ns]$')
        ax[i].set_xticks(range(0, 60, 10))
        ax[i].grid('both')
        ax[i].legend()

    fig, ax = plt.subplots(2,1)
    fig.suptitle("Bundle S3 - shield 2")
    ax[0].plot(1e9*(p.probes["b3_junction_current"].t-1.5*dt), 1e3*p.probes["b3_junction_current"].val[:,0], label = 'Current on bundle3, outer shield. x = 0')
    ax[0].plot(1e9*t, 1e3*I_b3_sh2_junction, 'r--', label = 'result from manual ')

    ax[1].plot(1e9*(p.probes["b3_terminal_current"].t-1.5*dt), 1e3*p.probes["b3_terminal_current"].val[:,0], label = 'Current on bundle3, outer shield. x = 0.165')
    ax[1].plot(1e9*t, 1e3*I_b3_sh2_terminal, 'r--', label = 'result from manual ')

    for i in range(2):
        ax[i].set_ylabel(r'$I (t)\,[mA]$')
        ax[i].set_xlabel(r'$t\,[ns]$')
        ax[i].set_xticks(range(0, 60, 10))
        ax[i].grid('both')
        ax[i].legend()
        
    #####################################
    
    fig, ax = plt.subplots(2,1)
    fig.suptitle("Bundle S1 - shield 1")
    ax[0].plot(1e9*(p.probes["b1_terminal_current"].t-1.5*dt), 1e3*p.probes["b1_terminal_current"].val[:,1], label = 'Current on bundle1, outer shield. x = 0')
    ax[0].plot(1e9*t, 1e3*I_b1_sh1_terminal, 'r--', label = 'result from manual ')

    # ax[1].plot(1e9*(p.probes["b1_mid_current"].t-1.5*dt), 1e3*p.probes["b1_mid_current"].val[:,1], label = 'Current on bundle1, outer shield. x = 0.1')
    # ax[1].plot(1e9*t, 1e3*I_b1_sh1_mid, 'r--', label = 'result from manual ')

    ax[1].plot(1e9*(p.probes["b1_junction_current"].t-1.5*dt), 1e3*p.probes["b1_junction_current"].val[:,1], label = 'Current on bundle1, outer shield. x = 0.54')
    ax[1].plot(1e9*t, 1e3*I_b1_sh1_junction, 'r--', label = 'result from manual ')

    for i in range(2):
        ax[i].set_ylabel(r'$I (t)\,[mA]$')
        ax[i].set_xlabel(r'$t\,[ns]$')
        ax[i].set_xticks(range(0, 60, 10))
        ax[i].grid('both')
        ax[i].legend()

    fig, ax = plt.subplots(2,1)
    fig.suptitle("Bundle S2 - shield 1")
    ax[0].plot(1e9*(p.probes["b2_junction_current"].t-1.5*dt), 1e3*p.probes["b2_junction_current"].val[:,1], label = 'Current on bundle2, inner shield. x = 0.0')
    ax[0].plot(1e9*t, 1e3*I_b2_sh1_junction, 'r--', label = 'result from manual ')

    # ax[1].plot(1e9*(p.probes["b2_connector_current"].t-1.5*dt), 1e3*p.probes["b2_connector_current"].val[:,1], label = 'Current on bundle2, inner shield. x = 0.3')
    # ax[1].plot(1e9*t, 1e3*I_b2_sh1_connector, 'r--', label = 'result from manual ')

    ax[1].plot(1e9*(p.probes["b2_terminal_current"].t-1.5*dt), 1e3*p.probes["b2_terminal_current"].val[:,1], label = 'Current on bundle2, inner shield. x = 0.343')
    ax[1].plot(1e9*t, 1e3*I_b2_sh1_terminal, 'r--', label = 'result from manual ')

    for i in range(2):
        ax[i].set_ylabel(r'$I (t)\,[mA]$')
        ax[i].set_xlabel(r'$t\,[ns]$')
        ax[i].set_xticks(range(0, 60, 10))
        ax[i].grid('both')
        ax[i].legend()

    fig, ax = plt.subplots(2,1)
    fig.suptitle("Bundle S3 - shield 1")
    ax[0].plot(1e9*(p.probes["b3_junction_current"].t-1.5*dt), 1e3*p.probes["b3_junction_current"].val[:,1], label = 'Current on bundle3, inner shield. x = 0')
    ax[0].plot(1e9*t, 1e3*I_b3_sh1_junction, 'r--', label = 'result from manual ')

    ax[1].plot(1e9*(p.probes["b3_terminal_current"].t-1.5*dt), 1e3*p.probes["b3_terminal_current"].val[:,1], label = 'Current on bundle3, inner shield. x = 0.165')
    ax[1].plot(1e9*t, 1e3*I_b3_sh1_terminal, 'r--', label = 'result from manual ')

    for i in range(2):
        ax[i].set_ylabel(r'$I (t)\,[mA]$')
        ax[i].set_xlabel(r'$t\,[ns]$')
        ax[i].set_xticks(range(0, 60, 10))
        ax[i].grid('both')
        ax[i].legend()

    #####################################
    
    fig, ax = plt.subplots(2,1)
    fig.suptitle("Bundle S1 - wire s1s2c2")
    ax[0].plot(1e9*(p.probes["b1_terminal_current"].t-1.5*dt), 1e6*p.probes["b1_terminal_current"].val[:,8], label = 'Current on bundle1, wire s1s2c2. x = 0')
    ax[0].plot(1e9*t, 1e6*I_b1_s1s2c2_terminal, 'r--', label = 'result from manual ')

    ax[1].plot(1e9*(p.probes["b1_junction_current"].t-1.5*dt), 1e6*p.probes["b1_junction_current"].val[:,8], label = 'Current on bundle1, wire s1s2c2. x = 0.54')
    ax[1].plot(1e9*t, 1e6*I_b1_s1s2c2_junction, 'r--', label = 'result from manual ')


    for i in range(2):
        ax[i].set_ylabel(r'$I (t)\,[\mu A]$')
        ax[i].set_xlabel(r'$t\,[ns]$')
        ax[i].set_xticks(range(0, 60, 10))
        ax[i].grid('both')
        ax[i].legend()

    fig, ax = plt.subplots(2,1)
    fig.suptitle("Bundle S1 - wire s1s4c2")
    ax[0].plot(1e9*(p.probes["b1_terminal_current"].t-1.5*dt), 1e6*p.probes["b1_terminal_current"].val[:,6], label = 'Current on bundle1, wire s1s4c2. x = 0')
    ax[0].plot(1e9*t, 1e6*I_b1_s1s4c2_terminal, 'r--', label = 'result from manual ')

    ax[1].plot(1e9*(p.probes["b1_junction_current"].t-1.5*dt), 1e6*p.probes["b1_junction_current"].val[:,6], label = 'Current on bundle1, wire s1s4c2. x = 0.54')
    ax[1].plot(1e9*t, 1e6*I_b1_s1s4c2_junction, 'r--', label = 'result from manual ')


    for i in range(2):
        ax[i].set_ylabel(r'$I (t)\,[\mu A]$')
        ax[i].set_xlabel(r'$t\,[ns]$')
        ax[i].set_xticks(range(0, 60, 10))
        ax[i].grid('both')
        ax[i].legend()

    fig, ax = plt.subplots(2,1)
    fig.suptitle("Bundle S2 - wire s1s2c2")
    ax[0].plot(1e9*(p.probes["b2_junction_current"].t-1.5*dt), 1e6*p.probes["b2_junction_current"].val[:,2], label = 'Current on bundle2, wire s1s2c2. x = 0')
    ax[0].plot(1e9*t, 1e6*I_b2_s1s2c2_junction, 'r--', label = 'result from manual ')

    # ax[1].plot(1e9*(p.probes["b2_connector_current"].t-1.5*dt), 1e6*p.probes["b2_connector_current"].val[:,2], label = 'Current on bundle2, wire s1s2c2. x = 0.3')
    # ax[1].plot(1e9*t, 1e6*I_b2_s1s2c2_connector, 'r--', label = 'result from manual ')

    ax[1].plot(1e9*(p.probes["b2_terminal_current"].t-1.5*dt), 1e6*p.probes["b2_terminal_current"].val[:,2], label = 'Current on bundle2, wire s1s2c2. x = 0.342')
    ax[1].plot(1e9*t, 1e6*I_b2_s1s2c2_terminal, 'r--', label = 'result from manual ')

    for i in range(2):
        ax[i].set_ylabel(r'$I (t)\,[\mu A]$')
        ax[i].set_xlabel(r'$t\,[ns]$')
        ax[i].set_xticks(range(0, 60, 10))
        ax[i].grid('both')
        ax[i].legend()


    plt.show()
    
def test_ex7_simplified():
    file = 'python/testData/parser/example7_simplified.smb.json'

    p = Parser(file)
    p.run(finalTime=50e-9,dt=0.5e-10)   

    t, Ish = np.genfromtxt('python/testData/bundles_test/ex7s2cout_simplified.txt', delimiter=',', usecols=(0,1), unpack = True)
    I1, I2 = np.genfromtxt('python/testData/bundles_test/ex7curout_simplified.txt', delimiter=',', usecols=(1,2), unpack = True)
    
    fig, ax = plt.subplots(3,1)
    ax[0].plot(1e9*p.probes["i_sh21c1"].t, 1e3*p.probes["i_sh21c1"].val[:,0], label = 'sh21c1 - shield 0, c1')
    ax[0].plot(1e9*t, 1e3*Ish, 'r--', label = 'result from manual ')
    ax[0].set_ylabel(r'$I (t)\,[mA]$')
    ax[0].set_xlabel(r'$t\,[ns]$')
    ax[0].set_xticks(range(0, 60, 10))
    ax[0].grid('both')
    ax[0].legend()
    
    ax[1].plot(1e9*p.probes["cable_1_terminal_current_start"].t, 1e6*p.probes["cable_1_terminal_current_start"].val[:,2], label = 'cable 1 - s1s2c2 - I')
    ax[1].plot(1e9*t, 1e6*I1, 'r--', label = 'result from manual ')
    ax[1].set_ylabel(r'$I (t)\,[\mu A]$')
    ax[1].set_xlabel(r'$t\,[ns]$')
    ax[1].set_xticks(range(0, 60, 10))
    ax[1].grid('both')
    ax[1].legend()

    ax[2].plot(1e9*p.probes["cable_1_terminal_current_end"].t, 1e6*p.probes["cable_1_terminal_current_end"].val[:,2], label = 'cable 1 - s1s4c2-I')
    ax[2].plot(1e9*t, 1e6*I2, 'r--', label = 'result from manual ')
    ax[2].set_ylabel(r'$I (t)\,[\mu A]$')
    ax[2].set_xlabel(r'$t\,[ns]$')
    ax[2].set_xticks(range(0, 60, 10))
    ax[2].grid('both')
    ax[2].legend()
    

    plt.show()
def test_ex7_simplified_R_R():
    file = 'python/testData/parser/example7_simplified_R_R.smb.json'

    p = Parser(file)
    p.run(finalTime=50e-9,dt=0.5e-10)   

    t, Ish = np.genfromtxt('python/testData/bundles_test/ex7s2cout_simplified_R_R.txt', delimiter=',', usecols=(0,1), unpack = True)
    I1, I2 = np.genfromtxt('python/testData/bundles_test/ex7curout_simplified_R_R.txt', delimiter=',', usecols=(1,2), unpack = True)
    
    # tzz, Ishzz = np.genfromtxt('python/testData/bundles_test/ex7s2cout_simplified_Z_Z.txt', delimiter=',', usecols=(0,1), unpack = True)
    # I1zz, I2zz = np.genfromtxt('python/testData/bundles_test/ex7curout_simplified_Z_Z.txt', delimiter=',', usecols=(1,2), unpack = True)

    
    fig, ax = plt.subplots(3,1)
    ax[0].plot(1e9*p.probes["i_sh21c1"].t, 1e3*p.probes["i_sh21c1"].val[:,0], label = 'sh21c1 - shield 0, c1')
    ax[0].plot(1e9*t, 1e3*Ish, 'r--', label = 'result from manual ')
    # ax[0].plot(1e9*tzz, 1e3*Ishzz, '-.', label = 'result from manual zz')
    ax[0].set_ylabel(r'$I (t)\,[mA]$')
    ax[0].set_xlabel(r'$t\,[ns]$')
    ax[0].set_xticks(range(0, 60, 10))
    ax[0].grid('both')
    ax[0].legend()
    
    ax[1].plot(1e9*p.probes["cable_1_terminal_current_start"].t, 1e6*p.probes["cable_1_terminal_current_start"].val[:,2], label = 'cable 1 - s1s2c2 - I')
    ax[1].plot(1e9*t, 1e6*I1, 'r--', label = 'result from manual ')
    # ax[1].plot(1e9*tzz, 1e6*I1zz, '-.', label = 'result from manual zz ')
    ax[1].set_ylabel(r'$I (t)\,[\mu A]$')
    ax[1].set_xlabel(r'$t\,[ns]$')
    ax[1].set_xticks(range(0, 60, 10))
    ax[1].grid('both')
    ax[1].legend()

    ax[2].plot(1e9*p.probes["cable_1_terminal_current_end"].t, 1e6*p.probes["cable_1_terminal_current_end"].val[:,2], label = 'cable 1 - s1s4c2-I')
    ax[2].plot(1e9*t, 1e6*I2, 'r--', label = 'result from manual ')
    # ax[2].plot(1e9*tzz, 1e6*I2zz, '-.', label = 'result from manual zz')
    ax[2].set_ylabel(r'$I (t)\,[\mu A]$')
    ax[2].set_xlabel(r'$t\,[ns]$')
    ax[2].set_xticks(range(0, 60, 10))
    ax[2].grid('both')
    ax[2].legend()
    

    plt.show()

def test_ex7_simplified_Z_Z():
    file = 'python/testData/parser/example7_simplified_Z_Z.smb.json'

    p = Parser(file)
    p.run(finalTime=50e-9,dt=0.5e-10)   

    # trr, Ishrr = np.genfromtxt('python/testData/bundles_test/ex7s2cout_simplified_R_R.txt', delimiter=',', usecols=(0,1), unpack = True)
    # I1rr, I2rr = np.genfromtxt('python/testData/bundles_test/ex7curout_simplified_R_R.txt', delimiter=',', usecols=(1,2), unpack = True)

    t, Ish = np.genfromtxt('python/testData/bundles_test/ex7s2cout_simplified_Z_Z.txt', delimiter=',', usecols=(0,1), unpack = True)
    I1, I2 = np.genfromtxt('python/testData/bundles_test/ex7curout_simplified_Z_Z.txt', delimiter=',', usecols=(1,2), unpack = True)
    
    fig, ax = plt.subplots(3,1)
    ax[0].plot(1e9*p.probes["i_sh21c1"].t, 1e3*p.probes["i_sh21c1"].val[:,0], label = 'sh21c1 - shield 0, c1')
    ax[0].plot(1e9*t, 1e3*Ish, 'r--', label = 'result from manual ')
    # ax[0].plot(1e9*trr, 1e3*Ishrr, '-.', label = 'result from manual rr')
    ax[0].set_ylabel(r'$I (t)\,[mA]$')
    ax[0].set_xlabel(r'$t\,[ns]$')
    ax[0].set_xticks(range(0, 60, 10))
    ax[0].grid('both')
    ax[0].legend()
    
    ax[1].plot(1e9*p.probes["cable_1_terminal_current_start"].t, 1e6*p.probes["cable_1_terminal_current_start"].val[:,2], label = 'cable 1 - s1s2c2 - I')
    ax[1].plot(1e9*t, 1e6*I1, 'r--', label = 'result from manual ')
    # ax[1].plot(1e9*trr, 1e6*I1rr, '-.', label = 'result from manual rr ')
    ax[1].set_ylabel(r'$I (t)\,[\mu A]$')
    ax[1].set_xlabel(r'$t\,[ns]$')
    ax[1].set_xticks(range(0, 60, 10))
    ax[1].grid('both')
    ax[1].legend()

    ax[2].plot(1e9*p.probes["cable_1_terminal_current_end"].t, 1e6*p.probes["cable_1_terminal_current_end"].val[:,2], label = 'cable 1 - s1s4c2-I')
    ax[2].plot(1e9*t, 1e6*I2, 'r--', label = 'result from manual ')
    # ax[2].plot(1e9*trr, 1e6*I2rr, '-.', label = 'result from manual rr ')
    ax[2].set_ylabel(r'$I (t)\,[\mu A]$')
    ax[2].set_xlabel(r'$t\,[ns]$')
    ax[2].set_xticks(range(0, 60, 10))
    ax[2].grid('both')
    ax[2].legend()
    

    plt.show()
    
def test_e_drive():
    file = 'python/testData/parser/E_drive.smb.json'

    p = Parser(file)
    p.run(finalTime=50e-9,dt=0.5e-10)

    plt.figure()
    
    plt.plot(1e9*p.probes["i_sh21c1"].t, 1e3*p.probes["i_sh21c1"].val[:,0], label = 'sh21c1 - shield 0, c1')
    plt.ylabel(r'$I (t)\,[mA]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 60, 10))
    plt.grid('both')
    plt.legend()

    plt.show()

def test_1_bundle_3_levels_1_1_2():
    # file = 'python/testData/parser/test_1_bundle_3_levels_1_1_2.smb.json'
    file = 'python/testData/parser/test_7_bundles_Level_0.smb.json'
    # file = 'python/testData/parser/test_7_bundles_Level_1.smb.json'
    # file = 'python/testData/parser/test_7_bundles_Level_1_single_bundle.smb.json'

    p = Parser(file)
    p.run(finalTime=50e-9,dt=0.5e-10)
    # p.run(finalTime=0.5e-7,dt=0.5e-11)

    plt.figure()

    # plt.plot(1e9*p.probes["v_1"].t, p.probes["v_1"].val[:,0], label = 'Voltage at source, node 1, terminal')
    plt.plot(1e9*p.probes["v_2"].t, p.probes["v_2"].val[:,0], '--', label = 'Voltage on junction, node 2')
    plt.plot(1e9*p.probes["v_3"].t, p.probes["v_3"].val[:,0], '-.', label = 'Voltage on junction, node 3')
    plt.plot(1e9*p.probes["v_4"].t, p.probes["v_4"].val[:,0], ':',label = 'Voltage on junction, node 4')
    plt.ylabel(r'$I (t)\,[A]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 55, 5))
    # plt.yticks(range(0, 1.2, ))
    # plt.ylim(0,1)
    plt.grid('both')
    plt.legend()

    # plt.plot(1e9*p.probes["v_2"].t, p.probes["v_2"].val[:,0], '--', label = 'Voltage on junction, node 2')
    # plt.plot(1e9*p.probes["v_3"].t, p.probes["v_3"].val[:,0], '-.', label = 'Voltage on junction, node 3')
    # plt.plot(1e9*p.probes["v_4"].t, p.probes["v_4"].val[:,0], ':',label = 'Voltage on junction, node 4')
    # plt.ylabel(r'$V (t)\,[V]$')
    # plt.xlabel(r'$t\,[ns]$')
    # plt.xticks(range(0, 60, 5))
    # plt.yticks(range(0, 1.2, ))
    # plt.ylim(0,1)
    # plt.grid('both')
    # plt.legend()
    
    # ax[2].set_ylabel(r'$V (t)\,[V]$')
    # ax[2].set_xlabel(r'$t\,[ns]$')
    # ax[2].set_xticks(range(0, 60, 10))
    # # plt.yticks(range(0, 1.2, ))
    # # plt.ylim(0,1)
    # ax[2].grid('both')
    # ax[2].legend()

    # ax[3].set_ylabel(r'$V (t)\,[V]$')
    # ax[3].set_xlabel(r'$t\,[ns]$')
    # ax[3].set_xticks(range(0, 60, 10))
    # # plt.yticks(range(0, 1.2, ))
    # # plt.ylim(0,1)
    # ax[3].grid('both')
    # ax[3].legend()
    
    plt.tight_layout()
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
    file = 'python/testData/parser/test_8_two_conductor_line_with_source.smb.json'
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

def test_bundles_R_line_C():
    file = 'python/testData/parser/R_line_C.smb.json'

    p = Parser(file)
    p.run(finalTime=50e-9,dt=0.5e-10)


    t0, V2, V3 = np.genfromtxt('python/testData/ngspice/nw_validation/C_termination.txt', usecols=(0,1,3), unpack = True)

    plt.figure()
    
    plt.plot(1e9*p.probes["v_left"].t, p.probes["v_left"].val, 'r',label = 'Voltage, left')
    plt.plot(1e9*t0, V2, 'g-.' ,label = 'Voltage, left, Spice')
    plt.plot(1e9*p.probes["v_right"].t, p.probes["v_right"].val, 'b', label = 'Voltage, right')
    plt.plot(1e9*t0, V3, 'k-.', label = 'Voltage, right, Spice')
    plt.ylabel(r'$V (t)\,[V]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 60, 10))
    plt.grid('both')
    plt.legend()

    plt.show()


def test_bundles_R_line_LCpRs():
    file = 'python/testData/parser/R_line_LCp_Rs.smb.json'

    p = Parser(file)
    p.run(finalTime=50e-9,dt=0.5e-10)


    t0, V2, V3 = np.genfromtxt('python/testData/ngspice/nw_validation/LCp_Rs_termination.txt', usecols=(0,1,3), unpack = True)

    plt.figure()
    
    plt.plot(1e9*p.probes["v_left"].t, p.probes["v_left"].val, 'r',label = 'Voltage, left')
    plt.plot(1e9*t0, V2, 'g-.' ,label = 'Voltage, left, Spice')
    plt.plot(1e9*p.probes["v_right"].t, p.probes["v_right"].val, 'b', label = 'Voltage, right')
    plt.plot(1e9*t0, V3, 'k-.', label = 'Voltage, right, Spice')
    plt.ylabel(r'$V (t)\,[V]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 60, 10))
    plt.grid('both')
    plt.legend()

    plt.show()

def test_bundles_R_MTL_LCpRs():
    file = 'python/testData/parser/R_MTL_LCp_Rs.smb.json'

    p = Parser(file)
    p.run(finalTime=50e-9,dt=0.5e-10)


    t0, V3, V6 = np.genfromtxt('python/testData/ngspice/nw_validation/MTL_LCp_Rs_termination.txt', usecols=(0,3,7), unpack = True)

    fig, ax = plt.subplots(2,1)
   
    ax[0].plot(1e9*p.probes["v_right"].t, p.probes["v_right"].val[:,0], 'r',label = 'Voltage, right, line 0')
    ax[0].plot(1e9*t0, V3, 'g-.' ,label = 'Voltage, right, line 0, Spice')
    ax[1].plot(1e9*p.probes["v_right"].t, p.probes["v_right"].val[:,1], 'b', label = 'Voltage, right, line 1')
    ax[1].plot(1e9*t0, V6, 'k-.', label = 'Voltage, right, line 1 Spice')
    ax[0].set_ylabel(r'$V (t)\,[V]$')
    ax[1].set_ylabel(r'$V (t)\,[V]$')
    ax[1].set_xlabel(r'$t\,[ns]$')
    ax[0].set_xticks(range(0, 60, 10))
    ax[1].set_xticks(range(0, 60, 10))
    ax[0].grid('both')
    ax[1].grid('both')
    ax[0].legend()
    ax[1].legend()

    plt.show()

def test_5_coaxial_v11():
    file = 'python/testData/parser/coaxial_tests/test_5_coaxial_v11.smb.json'
    p = Parser(file)
    dt = 0.5e-10
    p.run(finalTime=30e-9, dt=dt)

    t, Is, Im, Ie = np.genfromtxt('python/testData/bundles_test/coaxial_v11/coaxial_v11_ex5s1cout.txt', delimiter=',', usecols=(0,1,2,3), unpack = True)
    tV, Vs, Vm, Ve = np.genfromtxt('python/testData/bundles_test/coaxial_v11/coaxial_v11_ex5vout.txt', delimiter=',', usecols=(0,1,2,3), unpack = True)

    # Is_re = np.interp(p.probes["i_start"].t, t, Is)
    
    # t = t[2:]
    # Is = Is[0:-2]
    
    fig, ax = plt.subplots(2,1)
    ax[0].plot(1e9*(p.probes["i_start"].t-2.0*dt), p.probes["i_start"].val[:,0], 'b',label = 'Current on shield. x = 0')
    ax[0].plot(1e9*t, Is, 'b--',label = 'mHarness: Current on shield. x = 0')

    # ax[0].plot(1e9*(p.probes["v_mid"].t), p.probes["i_mid"].val[:,0], 'b',label = 'Current on shield. x = 0.5')
    # ax[0].plot(1e9*t, Im, 'b--',label = 'mHarness: Current on shield. x = 0.5')

    ax[0].plot(1e9*(p.probes["i_end"].t-2.0*dt), p.probes["i_end"].val[:,0], 'g',label = 'Current on shield. x = 0.54')
    ax[0].plot(1e9*t, Ie, 'g--',label = 'mHarness: Current on shield. x = 0.54')
    ax[0].set_ylabel(r'$I (t)\,[A]$')
    ax[0].set_xlabel(r'$t\,[ns]$')
    ax[0].grid('both')
    ax[0].legend()


    ax[1].plot(1e9*(p.probes["v_start"].t-1.5*dt), p.probes["v_start"].val[:,1], 'b', label = 'Voltage on inner conductor 1. x = 0')
    ax[1].plot(1e9*t, Vs, 'b--',label = 'mHarness: Voltage on inner conductor 1. x = 0')
    # ax[1].plot(1e9*(p.probes["v_mid"].t), p.probes["v_mid"].val[:,1], 'b', label = 'Voltage on inner conductor 1. x = 0.5')
    # ax[1].plot(1e9*t, Vm, 'b--',label = 'mHarness: Voltage on inner conductor 1. x = 0.5')
    ax[1].plot(1e9*(p.probes["v_end"].t-1.5*dt), p.probes["v_end"].val[:,1], 'g', label = 'Voltage on inner conductor 1. x = 0.54')
    ax[1].plot(1e9*t, Ve, 'g--',label = 'mHarness: Voltage on inner conductor 1. x = 0.54')
    ax[1].set_ylabel(r'$V (t)\,[V]$')
    ax[1].set_xlabel(r'$t\,[ns]$')
    ax[1].grid('both')
    ax[1].legend()
    
    plt.show()

def test_5_coaxial_v11_pin_voltage():
    file = 'python/testData/parser/coaxial_tests/test_5_coaxial_v11_pin.smb.json'
    p = Parser(file)

    p.run(finalTime=30e-9, dt=0.5e-10)

    t, Is, Im, Ie = np.genfromtxt('python/testData/bundles_test/coaxial_v11/coaxial_v11_pin_ex5s1cout.txt', delimiter=',', usecols=(0,1,2,3), unpack = True)
    tV, Vs, Vm, Ve = np.genfromtxt('python/testData/bundles_test/coaxial_v11/coaxial_v11_pin_ex5vout.txt', delimiter=',', usecols=(0,1,2,3), unpack = True)

    # Is_re = np.interp(p.probes["i_start"].t, t, Is)
    
    # t = t[2:]
    # Is = Is[0:-2]
    
    fig, ax = plt.subplots(2,1)
    ax[0].plot(1e9*(p.probes["v_start"].t-0.25e-10), p.probes["i_start"].val[:,0], 'b',label = 'Current on shield. x = 0')
    ax[0].plot(1e9*t, Is, 'b--',label = 'mHarness: Current on shield. x = 0')

    # ax[0].plot(1e9*(p.probes["v_mid"].t), p.probes["i_mid"].val[:,0], 'b',label = 'Current on shield. x = 0.5')
    # ax[0].plot(1e9*t, Im, 'b--',label = 'mHarness: Current on shield. x = 0.5')

    ax[0].plot(1e9*(p.probes["v_end"].t-0.25e-10), p.probes["i_end"].val[:,0], 'g',label = 'Current on shield. x = 0.54')
    ax[0].plot(1e9*t, Ie, 'g--',label = 'mHarness: Current on shield. x = 0.54')
    ax[0].set_ylabel(r'$I (t)\,[A]$')
    ax[0].set_xlabel(r'$t\,[ns]$')
    ax[0].grid('both')
    ax[0].legend()


    ax[1].plot(1e9*(p.probes["v_start"].t), p.probes["v_start"].val[:,1], 'b', label = 'Voltage on inner conductor 1. x = 0')
    ax[1].plot(1e9*t, Vs, 'b--',label = 'mHarness: Voltage on inner conductor 1. x = 0')
    # ax[1].plot(1e9*(p.probes["v_mid"].t), p.probes["v_mid"].val[:,1], 'b', label = 'Voltage on inner conductor 1. x = 0.5')
    # ax[1].plot(1e9*t, Vm, 'b--',label = 'mHarness: Voltage on inner conductor 1. x = 0.5')
    ax[1].plot(1e9*(p.probes["v_end"].t), p.probes["v_end"].val[:,1], 'g', label = 'Voltage on inner conductor 1. x = 0.54')
    ax[1].plot(1e9*t, Ve, 'g--',label = 'mHarness: Voltage on inner conductor 1. x = 0.54')
    ax[1].set_ylabel(r'$V (t)\,[V]$')
    ax[1].set_xlabel(r'$t\,[ns]$')
    ax[1].grid('both')
    ax[1].legend()
    
    plt.show()

def test_5_coaxial_v12():
    file = 'python/testData/parser/coaxial_tests/test_5_coaxial_v12.smb.json'
    p = Parser(file)

    p.run(finalTime=30e-9, dt=0.5e-10)

    t, Is, Im, Ie = np.genfromtxt('python/testData/bundles_test/coaxial_v12/coaxial_v12_ex5s1cout.txt', delimiter=',', usecols=(0,1,2,3), unpack = True)
    _, Vs, Vm, Ve = np.genfromtxt('python/testData/bundles_test/coaxial_v12/coaxial_v12_ex5vout.txt', delimiter=',', usecols=(0,1,2,3), unpack = True)

    dt0 = 0.0
    dt1 = 0.0
    # dt0 = 1e-10
    # dt1 = 1.492537313432836e-10
    
    fig, ax = plt.subplots(2,1)
    ax[0].plot(1e9*(p.probes["i_start"].t-dt0), p.probes["i_start"].val[:,0], 'b',label = 'Current on shield. x = 0')
    ax[0].plot(1e9*t, Is, 'b--',label = 'mHarness: Current on shield. x = 0')
    
    ax[0].plot(1e9*(p.probes["i_end"].t-dt0), p.probes["i_end"].val[:,0], 'g',label = 'Current on shield. x = 0.54')
    ax[0].plot(1e9*t, Ie, 'g--',label = 'mHarness: Current on shield. x = 0.54')

    ax[0].set_ylabel(r'$I (t)\,[A]$')
    ax[0].set_xlabel(r'$t\,[ns]$')
    ax[0].grid('both')
    ax[0].legend()

    ax[1].plot(1e9*(p.probes["v_start"].t-dt1), p.probes["v_start"].val[:,1], 'b', label = 'Voltage on inner conductor 1. x = 0')
    ax[1].plot(1e9*t, Vs, 'b--',label = 'mHarness: Voltage on inner conductor 1. x = 0')
    
    ax[1].plot(1e9*(p.probes["v_end"].t-dt1), p.probes["v_end"].val[:,1], 'g', label = 'Voltage on inner conductor 1. x = 0.54')
    ax[1].plot(1e9*t, Ve, 'g--',label = 'mHarness: Voltage on inner conductor 1. x = 0.54')
    
    ax[1].set_ylabel(r'$V (t)\,[V]$')
    ax[1].set_xlabel(r'$t\,[ns]$')
    ax[1].grid('both')
    ax[1].legend()
    
    plt.show()

def test_5_coaxial_v13():
    file = 'python/testData/parser/coaxial_tests/test_5_coaxial_v13.smb.json'
    p = Parser(file)

    p.run(finalTime=30e-9, dt=0.5e-10)

    t, Is, Im, Ie = np.genfromtxt('python/testData/bundles_test/coaxial_v13/coaxial_v13_ex5s1cout.txt', delimiter=',', usecols=(0,1,2,3), unpack = True)
    _, Vs, Vm, Ve = np.genfromtxt('python/testData/bundles_test/coaxial_v13/coaxial_v13_ex5vout.txt', delimiter=',', usecols=(0,1,2,3), unpack = True)

    dt0 = 0.0
    dt1 = 0.0
    # dt0 = 1e-10
    # dt1 = 1.492537313432836e-10
    
    fig, ax = plt.subplots(2,1)
    ax[0].plot(1e9*(p.probes["i_start"].t-dt0), p.probes["i_start"].val[:,0], 'b',label = 'Current on shield. x = 0')
    ax[0].plot(1e9*t, Is, 'b--',label = 'mHarness: Current on shield. x = 0')
    
    ax[0].plot(1e9*(p.probes["i_end"].t-dt0), p.probes["i_end"].val[:,0], 'g',label = 'Current on shield. x = 0.54')
    ax[0].plot(1e9*t, Ie, 'g--',label = 'mHarness: Current on shield. x = 0.54')

    ax[0].set_ylabel(r'$I (t)\,[A]$')
    ax[0].set_xlabel(r'$t\,[ns]$')
    ax[0].grid('both')
    ax[0].legend()

    ax[1].plot(1e9*(p.probes["v_start"].t-dt1), p.probes["v_start"].val[:,1], 'b', label = 'Voltage on inner conductor 1. x = 0')
    ax[1].plot(1e9*t, Vs, 'b--',label = 'mHarness: Voltage on inner conductor 1. x = 0')
    
    ax[1].plot(1e9*(p.probes["v_end"].t-dt1), p.probes["v_end"].val[:,1], 'g', label = 'Voltage on inner conductor 1. x = 0.54')
    ax[1].plot(1e9*t, Ve, 'g--',label = 'mHarness: Voltage on inner conductor 1. x = 0.54')
    
    ax[1].set_ylabel(r'$V (t)\,[V]$')
    ax[1].set_xlabel(r'$t\,[ns]$')
    ax[1].grid('both')
    ax[1].legend()
    
    plt.show()

def test_5_coaxial_v21():
    file = 'python/testData/parser/coaxial_tests/v21_two_shields_1_coaxial.smb.json'
    p = Parser(file)
    p.run(finalTime=50e-9,dt=0.5e-10)   


    t, I = np.genfromtxt('python/testData/bundles_test/coaxial_v21/ex7s2cout_1', delimiter=',', usecols=(0,1), unpack = True)
    _ ,I1, I2   = np.genfromtxt('python/testData/bundles_test/coaxial_v21/ex7curout_1', delimiter=',', usecols=(0,1,2), unpack = True)

    
    fig, ax = plt.subplots(3,1)
    ax[0].plot(1e9*p.probes["i_sh21c1"].t, 1e3*p.probes["i_sh21c1"].val[:,0], label = 'Current on shield. x = 0')
    # ax[0].plot(1e9*p.probes["v_sh21c1"].t, -10*p.probes["v_sh21c1"].val[:,0], label = 'Current on shield from V. x = 0')
    ax[0].plot(1e9*t, 1e3*I, 'r--', label = 'result from manual ')
    ax[0].set_ylabel(r'$I (t)\,[mA]$')
    ax[0].set_xlabel(r'$t\,[ns]$')
    ax[0].set_xticks(range(0, 60, 10))
    ax[0].grid('both')
    ax[0].legend()
    
    ax[1].plot(1e9*p.probes["cable_1_terminal_current_start"].t, 1e6*50*p.probes["cable_1_terminal_current_start"].val[:,3], label = 'Voltage on wire. x = 0')
    # ax[1].plot(1e9*p.probes["cable_1_terminal_voltage_start"].t, -1e6*p.probes["cable_1_terminal_voltage_start"].val[:,2], label = 'Voltage on wire. x = 0')
    # ax[1].plot(1e9*p.probes["cable_1_terminal_voltage_start"].t, -1e6*p.probes["cable_1_terminal_voltage_start"].val[:,2], label = 'Voltage')
    ax[1].plot(1e9*t, 1e6*50*I1, 'r--', label = 'result from manual ')
    ax[1].set_ylabel(r'$V (t)\,[\mu V]$')
    ax[1].set_xlabel(r'$t\,[ns]$')
    ax[1].set_xticks(range(0, 60, 10))
    ax[1].grid('both')
    ax[1].legend()

    ax[2].plot(1e9*p.probes["cable_1_terminal_current_end"].t, 1e6*50*p.probes["cable_1_terminal_current_end"].val[:,3], label = 'Voltage on wire. x = 0.54')
    # ax[2].plot(1e9*p.probes["cable_1_terminal_voltage_end"].t, 1e6*p.probes["cable_1_terminal_voltage_end"].val[:,2], label = 'Voltage')
    ax[2].plot(1e9*t, 1e6*50*I2, 'r--', label = 'result from manual ')
    ax[2].set_ylabel(r'$V (t)\,[\mu V]$')
    ax[2].set_xlabel(r'$t\,[ns]$')
    ax[2].set_xticks(range(0, 60, 10))
    ax[2].grid('both')
    ax[2].legend()
    

    plt.show()
    
def test_5_coaxial_v21_LC_computed():
    """
    LC and computed and used as input
    Probes on shield 2, shield 1 and wires
    """
    
    
    file = 'python/testData/parser/coaxial_tests/v21_two_shields_1_coaxial_LC_computed.smb.json'
    p = Parser(file)
    p.run(finalTime=50e-9,dt=0.5e-10)   


    t, Is2_0, Is2_1, Is2_3, Is2_54 = np.genfromtxt('python/testData/bundles_test/coaxial_v21/ex7s2cout_LC_computed', delimiter=',', usecols=(0,1,2,3,4), unpack = True)
    _, Is1_0, Is1_1, Is1_3, Is1_54 = np.genfromtxt('python/testData/bundles_test/coaxial_v21/ex7s1cout_LC_computed', delimiter=',', usecols=(0,1,2,3,4), unpack = True)
    _ ,Iw1_0, Iw1_1, Iw1_3, Iw1_54   = np.genfromtxt('python/testData/bundles_test/coaxial_v21/ex7curout_LC_computed', delimiter=',', usecols=(0,1,2,3,4), unpack = True)

    
    fig, ax = plt.subplots(3,1)
    ax[0].plot(1e9*p.probes["I_0"].t, 1e3*p.probes["I_0"].val[:,0], label = 'Current on outer shield x = 0')
    ax[0].plot(1e9*t, 1e3*Is2_0, 'r--', label = 'result from manual ')
    ax[0].set_ylabel(r'$I (t)\,[mA]$')
    ax[0].set_xlabel(r'$t\,[ns]$')
    ax[0].set_xticks(range(0, 60, 10))
    ax[0].grid('both')
    ax[0].legend()
    
    ax[1].plot(1e9*p.probes["I_0"].t, 1e6*p.probes["I_0"].val[:,1], label = 'Current in inner shield x = 0')
    ax[1].plot(1e9*t, 1e6*Is1_0, 'r--', label = 'result from manual ')
    ax[1].set_ylabel(r'$I (t)\,[mA]$')
    ax[1].set_xlabel(r'$t\,[ns]$')
    ax[1].set_xticks(range(0, 60, 10))
    ax[1].grid('both')
    ax[1].legend()

    ax[2].plot(1e9*p.probes["I_0"].t, 1e6*50*p.probes["I_0"].val[:,2], label = 'Voltage on wire. x = 0')
    ax[2].plot(1e9*t, 1e6*50*Iw1_0, 'r--', label = 'result from manual ')
    ax[2].set_ylabel(r'$V (t)\,[\mu V]$')
    ax[2].set_xlabel(r'$t\,[ns]$')
    ax[2].set_xticks(range(0, 60, 10))
    ax[2].grid('both')
    ax[2].legend()
    

    plt.show()
    
def test_5_coaxial_v21_LC_computed_w_connectors():
    """
    LC and computed and used as input
    Probes on shield 2, shield 1 and wires
    """
    
    
    file = 'python/testData/parser/coaxial_tests/v21_two_shields_1_coaxial_LC_computed_w_connectors.smb.json'
    p = Parser(file)
    p.run(finalTime=50e-9,dt=0.5e-10)   


    t, Is2_0, Is2_1, Is2_3, Is2_54 = np.genfromtxt('python/testData/bundles_test/coaxial_v21/ex7s2cout_LC_computed_w_connectors', delimiter=',', usecols=(0,1,2,3,4), unpack = True)
    _, Is1_0, Is1_1, Is1_3, Is1_54 = np.genfromtxt('python/testData/bundles_test/coaxial_v21/ex7s1cout_LC_computed_w_connectors', delimiter=',', usecols=(0,1,2,3,4), unpack = True)
    _ ,Iw1_0, Iw1_1, Iw1_3, Iw1_54   = np.genfromtxt('python/testData/bundles_test/coaxial_v21/ex7curout_LC_computed_w_connectors', delimiter=',', usecols=(0,1,2,3,4), unpack = True)

    
    fig, ax = plt.subplots(3,1)
    ax[0].plot(1e9*p.probes["I_0"].t, 1e3*p.probes["I_0"].val[:,0], label = 'Current on outer shield x = 0')
    ax[0].plot(1e9*t, 1e3*Is2_0, 'r--', label = 'result from manual ')
    ax[0].set_ylabel(r'$I (t)\,[mA]$')
    ax[0].set_xlabel(r'$t\,[ns]$')
    ax[0].set_xticks(range(0, 60, 10))
    ax[0].grid('both')
    ax[0].legend()
    
    ax[1].plot(1e9*p.probes["I_0"].t, 1e6*p.probes["I_0"].val[:,1], label = 'Current in inner shield x = 0')
    ax[1].plot(1e9*t, 1e6*Is1_0, 'r--', label = 'result from manual ')
    ax[1].set_ylabel(r'$I (t)\,[mA]$')
    ax[1].set_xlabel(r'$t\,[ns]$')
    ax[1].set_xticks(range(0, 60, 10))
    ax[1].grid('both')
    ax[1].legend()

    ax[2].plot(1e9*p.probes["I_0"].t, 1e6*50*p.probes["I_0"].val[:,2], label = 'Voltage on wire. x = 0')
    ax[2].plot(1e9*t, 1e6*50*Iw1_0, 'r--', label = 'result from manual ')
    ax[2].set_ylabel(r'$V (t)\,[\mu V]$')
    ax[2].set_xlabel(r'$t\,[ns]$')
    ax[2].set_xticks(range(0, 60, 10))
    ax[2].grid('both')
    ax[2].legend()
    

    plt.show()
    
def test_5_coaxial_v21_LC_computed_w_connectors_large():
    """
    LC and computed and used as input
    Probes on shield 2, shield 1 and wires
    """
    
    
    file = 'python/testData/parser/coaxial_tests/v21_two_shields_1_coaxial_LC_computed_w_connectors_large.smb.json'
    p = Parser(file)
    p.run(finalTime=50e-9,dt=0.5e-10)   


    t, Is2_0, Is2_1, Is2_3, Is2_54 = np.genfromtxt('python/testData/bundles_test/coaxial_v21/ex7s2cout_LC_computed_w_connectors_large', delimiter=',', usecols=(0,1,2,3,4), unpack = True)
    _, Is1_0, Is1_1, Is1_3, Is1_54 = np.genfromtxt('python/testData/bundles_test/coaxial_v21/ex7s1cout_LC_computed_w_connectors_large', delimiter=',', usecols=(0,1,2,3,4), unpack = True)
    _ ,Iw1_0, Iw1_1, Iw1_3, Iw1_54 = np.genfromtxt('python/testData/bundles_test/coaxial_v21/ex7curout_LC_computed_w_connectors_large', delimiter=',', usecols=(0,1,2,3,4), unpack = True)

    
    fig, ax = plt.subplots(3,1)
    ax[0].plot(1e9*p.probes["I_0"].t, 1e3*p.probes["I_0"].val[:,0], label = 'Current on outer shield x = 0')
    ax[0].plot(1e9*t, 1e3*Is2_0, 'r--', label = 'result from manual ')
    ax[0].set_ylabel(r'$I (t)\,[mA]$')
    ax[0].set_xlabel(r'$t\,[ns]$')
    ax[0].set_xticks(range(0, 60, 10))
    ax[0].grid('both')
    ax[0].legend()
    
    ax[1].plot(1e9*p.probes["I_0"].t, 1e6*p.probes["I_0"].val[:,1], label = 'Current in inner shield x = 0')
    ax[1].plot(1e9*t, 1e6*Is1_0, 'r--', label = 'result from manual ')
    ax[1].set_ylabel(r'$I (t)\,[\mu A]$')
    ax[1].set_xlabel(r'$t\,[ns]$')
    ax[1].set_xticks(range(0, 60, 10))
    ax[1].grid('both')
    ax[1].legend()

    ax[2].plot(1e9*p.probes["I_0"].t, 1e6*p.probes["I_0"].val[:,2], label = 'Current on wire 1. x = 0')
    ax[2].plot(1e9*t, 1e6*Iw1_0, 'r--', label = 'result from manual ')
    ax[2].set_ylabel(r'$I (t)\,[\mu A]$')
    ax[2].set_xlabel(r'$t\,[ns]$')
    ax[2].set_xticks(range(0, 60, 10))
    ax[2].grid('both')
    ax[2].legend()

    fig, ax = plt.subplots(3,1)
    ax[0].plot(1e9*p.probes["I_054"].t, 1e3*p.probes["I_054"].val[:,0], label = 'Current on outer shield x = 0.540')
    ax[0].plot(1e9*t, 1e3*Is2_54, 'r--', label = 'result from manual ')
    ax[0].set_ylabel(r'$I (t)\,[mA]$')
    ax[0].set_xlabel(r'$t\,[ns]$')
    ax[0].set_xticks(range(0, 60, 10))
    ax[0].grid('both')
    ax[0].legend()
    
    ax[1].plot(1e9*p.probes["I_054"].t, 1e6*p.probes["I_054"].val[:,1], label = 'Current in inner shield x = 0.54')
    ax[1].plot(1e9*t, 1e6*Is1_54, 'r--', label = 'result from manual ')
    ax[1].set_ylabel(r'$I (t)\,[\mu A]$')
    ax[1].set_xlabel(r'$t\,[ns]$')
    ax[1].set_xticks(range(0, 60, 10))
    ax[1].grid('both')
    ax[1].legend()

    ax[2].plot(1e9*p.probes["I_054"].t, 1e6*p.probes["I_054"].val[:,2], label = 'Current on wire 1. x = 0.54')
    ax[2].plot(1e9*t, 1e6*Iw1_54, 'r--', label = 'result from manual ')
    ax[2].set_ylabel(r'$A (t)\,[\mu A]$')
    ax[2].set_xlabel(r'$t\,[ns]$')
    ax[2].set_xticks(range(0, 60, 10))
    ax[2].grid('both')
    ax[2].legend()


    plt.show()
    

def test_5_coaxial_v21_moved_source():
    file = 'python/testData/parser/coaxial_tests/v21_two_shields_1_coaxial.smb.json'
    p = Parser(file)
    p.run(finalTime=50e-9,dt=0.5e-10)   


    t, I = np.genfromtxt('python/testData/bundles_test/coaxial_v21/two_shields_1_coaxial_moved_source_ex7s2cout', delimiter=',', usecols=(0,1), unpack = True)
    _ ,I1, I2   = np.genfromtxt('python/testData/bundles_test/coaxial_v21/two_shields_1_coaxial_moved_source_ex7curout', delimiter=',', usecols=(0,1,2), unpack = True)

    
    fig, ax = plt.subplots(3,1)
    ax[0].plot(1e9*p.probes["i_sh21c1"].t, 1e3*p.probes["i_sh21c1"].val[:,0], label = 'sh21c1 - shield 0, c1')
    ax[0].plot(1e9*t, 1e3*I, 'r--', label = 'result from manual ')
    ax[0].set_ylabel(r'$I (t)\,[mA]$')
    ax[0].set_xlabel(r'$t\,[ns]$')
    ax[0].set_xticks(range(0, 60, 10))
    ax[0].grid('both')
    ax[0].legend()
    
    ax[1].plot(1e9*p.probes["cable_1_terminal_voltage_start"].t, 1e6*50*p.probes["cable_1_terminal_current_start"].val[:,2], label = 'cable 1 - s1s2c2 - I')
    ax[1].plot(1e9*t, 1e6*50*I1, 'r--', label = 'result from manual ')
    ax[1].set_ylabel(r'$V (t)\,[\mu V]$')
    ax[1].set_xlabel(r'$t\,[ns]$')
    ax[1].set_xticks(range(0, 60, 10))
    ax[1].grid('both')
    ax[1].legend()

    ax[2].plot(1e9*p.probes["cable_1_terminal_voltage_end"].t, 1e6*50*p.probes["cable_1_terminal_current_end"].val[:,2], label = 'cable 1 - s1s4c2-I')
    ax[2].plot(1e9*t, 1e6*50*I2, 'r--', label = 'result from manual ')
    ax[2].set_ylabel(r'$V (t)\,[\mu V]$')
    ax[2].set_xlabel(r'$t\,[ns]$')
    ax[2].set_xticks(range(0, 60, 10))
    ax[2].grid('both')
    ax[2].legend()
    

    plt.show()
    
def test_5_coaxial_v21_moved_source_and_probes():
    file = 'python/testData/parser/coaxial_tests/v21_two_shields_1_coaxial_moved_source_and_probes.smb.json'
    p = Parser(file)
    p.run(finalTime=50e-9,dt=0.5e-10)   

    dt = 0.0
    
    t, I = np.genfromtxt('python/testData/bundles_test/coaxial_v21/two_shields_1_coaxial_moved_source_and_probes_ex7s2cout', delimiter=',', usecols=(0,1), unpack = True)
    _ ,I1, I2   = np.genfromtxt('python/testData/bundles_test/coaxial_v21/two_shields_1_coaxial_moved_source_and_probes_ex7curout', delimiter=',', usecols=(0,1,2), unpack = True)

    
    fig, ax = plt.subplots(3,1)
    ax[0].plot(1e9*(p.probes["i_sh21c1"].t-dt), 1e3*p.probes["i_sh21c1"].val[:,0], label = 'Current on shield, x = 0.1')
    ax[0].plot(1e9*t, 1e3*I, 'r--', label = 'result from manual ')
    ax[0].set_ylabel(r'$I (t)\,[mA]$')
    ax[0].set_xlabel(r'$t\,[ns]$')
    ax[0].set_xticks(range(0, 60, 10))
    ax[0].grid('both')
    ax[0].legend()
    
    ax[1].plot(1e9*(p.probes["voltage_0.03"].t-2*dt), 1e6*50*p.probes["current_0.03"].val[:,2], label = 'Voltage on wire. x = 0.03')
    ax[1].plot(1e9*t, 1e6*50*I1, 'r--', label = 'result from manual ')
    ax[1].set_ylabel(r'$V (t)\,[\mu V]$')
    ax[1].set_xlabel(r'$t\,[ns]$')
    ax[1].set_xticks(range(0, 60, 10))
    ax[1].grid('both')
    ax[1].legend()

    ax[2].plot(1e9*p.probes["voltage_0.51"].t, 1e6*50*p.probes["current_0.51"].val[:,2], label = 'Voltage on wire. x = 0.51')
    ax[2].plot(1e9*t, 1e6*50*I2, 'r--', label = 'result from manual ')
    ax[2].set_ylabel(r'$V (t)\,[\mu V]$')
    ax[2].set_xlabel(r'$t\,[ns]$')
    ax[2].set_xticks(range(0, 60, 10))
    ax[2].grid('both')
    ax[2].legend()
    
    ax[0].set_title(("localized efield drive x = 0.3-0.42"))

    plt.show()
        