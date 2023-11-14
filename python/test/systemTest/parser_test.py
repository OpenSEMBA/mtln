import numpy as np
import matplotlib.pyplot as plt

from src.jsonParser import Parser

import src.waveforms as wf
from src.mtl import Field

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
    file = 'python/testData/parser/Paul/ribbon_cable_20ns_termination_network.smb.json'
    p = Parser(file)
    p.run(finalTime = 200e-9)

    # plt.plot(1e9*p.probes["v0"].t, 1e3*p.probes["v0"].val)
    # plt.ylabel(r'$V_1 (0, t)\,[mV]$')
    # plt.xlabel(r'$t\,[ns]$')
    # plt.xticks(range(0, 200, 50))
    # plt.grid('both')
    # plt.show()

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
    file = 'python/testData/parser/Paul/ribbon_cable_1ns_paul_interconnection_network.smb.json'
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
    file = 'python/testData/parser/Paul/ribbon_cable_1ns_R_interconnection_network.smb.json'
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

    file = 'python/testData/parser/Paul/ribbon_cable_1ns_RV_interconnection_network.smb.json'
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

    file = 'python/testData/parser/Paul/ribbon_cable_1ns_RV_T_network.smb.json'
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
    
    file = 'python/testData/parser/manual/test_1_double_exponential.smb.json'
    p = Parser(file)

    distances = np.zeros([1, 51, 3])
    distances[:,:,0] = 0.0508
    p.runWithExternalField(
        finalTime=80e-9, 
        dt = 1.9e-10,
        field= Field(wf.double_exp_sp(2*5.25e4, 4.0e6,4.76e8),wf.null(),wf.null()), 
        distances = distances)

    t0, V = np.genfromtxt('python/testData/manual/test1/ex1voltout', delimiter=',', usecols=(0,1), unpack = True)
    t1, I = np.genfromtxt('python/testData/manual/test1/ex1curout' , delimiter=',', usecols=(0,1), unpack = True)


    fig, ax1 = plt.subplots()
    ax1.set_xlabel(r'$t\,[ns]$')
    ax1.set_ylabel(r'$V (t)\,[V]$', color='tab:red')
    ax1.plot(1e9*p.probes["v0"].t, p.probes["v0"].val, color='tab:red')
    ax1.plot(1e9*t0, V, color='tab:orange')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_xticks(range(0, 80, 20))

    ax2 = ax1.twinx() 
    ax2.set_ylabel(r'$I (t)\,[A]$', color='tab:blue') 
    ax2.plot(1e9*p.probes["i0"].t, p.probes["i0"].val, color='tab:blue')
    ax2.plot(1e9*t1, I, color='tab:green')
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
    file = 'python/testData/parser/manual/test_3_two_conductor_line.smb.json'
    p = Parser(file)
    p.run(finalTime=5e-9)
    
    t0, V = np.genfromtxt('python/testData/manual/test3/ex3voltout', delimiter=',', usecols=(0,1), unpack = True)
    t1, I = np.genfromtxt('python/testData/manual/test3/ex3curout' , delimiter=',', usecols=(0,1), unpack = True)

    
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
    file = 'python/testData/parser/manual/test_4_multilines.smb.json'
    p = Parser(file)
    p.run(finalTime=6.46e-9, dt = 3.23e-12)
    
    t, V1, V2, V3 = np.genfromtxt('python/testData/manual/test4/ex4voltout', delimiter=',',usecols=(0,1,2,3), unpack = True)
    
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
    
    file = 'python/testData/parser/manual/test_5_coaxial.smb.json'
    p = Parser(file)

    distances = np.zeros([1, 19, 3])
    distances[:,:,0] = 0.0508
    p.runWithExternalField(
        finalTime=30e-9, 
        dt = 0.5e-10,
        field= Field(wf.null(), wf.null(), wf.double_exp_xy_sp(C =10.5e4, a = 4.0e6, b = 4.76e8)), 
        distances = distances)
    
    t, _, I = np.genfromtxt('python/testData/manual/test5/ex5s1cout', delimiter = ',',usecols=(0,1,2), unpack = True)
    _, _, V = np.genfromtxt('python/testData/manual/test5/ex5vout', delimiter = ',',usecols=(0,1,2), unpack = True)

    
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
    file = 'python/testData/parser/manual/test_6_agrawal.smb.json'
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
    tm, V4m, V5m, V6m =    np.genfromtxt('python/testData/manual/test6/ex6voltout', delimiter = ',',usecols=(0,1,2,3), unpack = True)

    fig, ax = plt.subplots(3,1)
    ax[0].plot(1e9*p.probes["v"].t, p.probes["v"].val[:,2], 'r-.', label = 'V at end of wire 1')
    ax[0].plot(1e9*t, V5, 'g-.', label = 'V at end of wire 1, Spice')
    ax[0].plot(1e9*tm, V4m, 'b-.', label = 'V at end of wire 1, manual')
    ax[0].set_ylabel(r'$V (t)\,[V]$')
    ax[0].set_xticks(range(0, 70, 10))
    ax[0].set_ylim(-0.2,0.4)
    ax[0].grid('both')
    ax[0].legend()

    ax[1].plot(1e9*p.probes["v"].t, p.probes["v"].val[:,1], 'r-.', label = 'V at end of wire 2')
    ax[1].plot(1e9*t, V4, 'g-.', label = 'V at end of wire 2, Spice')
    ax[1].plot(1e9*tm, V5m, 'b-.', label = 'V at end of wire 2, manual')
    ax[1].set_ylabel(r'$V (t)\,[V]$')
    ax[1].set_xticks(range(0, 70, 10))
    ax[1].set_ylim(-0.2,0.4)
    ax[1].grid('both')
    ax[1].legend()

    ax[2].plot(1e9*p.probes["v"].t, p.probes["v"].val[:,1], 'r-.', label = 'V at end of wire 3')
    ax[2].plot(1e9*t, V6, 'g-.', label = 'V at end of wire 3, Spice')
    ax[2].plot(1e9*tm, V6m, 'b-.', label = 'V at end of wire 3, manual')
    ax[2].set_ylabel(r'$V (t)\,[V]$')
    ax[2].set_xlabel(r'$t\,[ns]$')
    ax[2].set_xticks(range(0, 70, 10))
    ax[2].set_ylim(-0.2,0.4)
    ax[2].grid('both')
    ax[2].legend()

    fig.tight_layout()
    plt.show()

   
def test_7_bundles():
    file = 'python/testData/parser/manual/test_7_bundles_single_conductors.smb.json'

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

def test_bug():
    file = 'python/testData/parser/manual/test_7_bundles_single_conductors_B1_B2_B3_l0.smb.json'

    dt = 0.5e-10
    p = Parser(file)
    p.run(finalTime=50e-9,dt=dt)   

    t, I_b1_sh2_terminal,I_b1_sh2_mid,I_b1_sh2_junction, I_b2_sh2_junction,I_b2_sh2_connector, I_b2_sh2_terminal, I_b3_sh2_junction,I_b3_sh2_terminal =\
        np.genfromtxt('python/testData/bundles_test/B1_B2_B3/shield_2_current_B1_B2_B3', delimiter=',', usecols=(0,1,2,3,4,5,6,7,8), unpack = True)
    
    _, I_b1_sh1_terminal,I_b1_sh1_mid,I_b1_sh1_junction, I_b2_sh1_junction,I_b2_sh1_connector, I_b2_sh1_terminal, I_b3_sh1_junction,I_b3_sh1_terminal =\
        np.genfromtxt('python/testData/bundles_test/B1_B2_B3/shield_1_current_B1_B2_B3', delimiter=',', usecols=(0,1,2,3,4,5,6,7,8), unpack = True)
    
    _, I_b1_s1s2c2_terminal,I_b1_s1s2c2_junction, I_b1_s1s4c2_terminal, I_b1_s1s4c2_junction, I_b2_s1s2c2_junction, I_b2_s1s2c2_connector,I_b2_s1s2c2_terminal =\
        np.genfromtxt('python/testData/bundles_test/B1_B2_B3/cable_current_B1_B2_B3', delimiter=',', usecols=(0,1,2,3,4,5,6,7), unpack = True)


    fig, ax = plt.subplots(2,1)
    fig.suptitle("Bundle S1 - shield 2")
    ax[0].plot(1e9*(p.probes["b1_terminal_current"].t-1.5*dt), 1e3*p.probes["b1_terminal_current"].val[:,0], label = 'Current on bundle1, outer shield. x = 0')
    ax[1].plot(1e9*(p.probes["b1_junction_current"].t-1.5*dt), 1e3*p.probes["b1_junction_current"].val[:,0], label = 'Current on bundle1, outer shield. x = 0.54')

    ax[0].plot(1e9*t, 1e3*I_b1_sh2_terminal, 'r--', label = 'result from manual ')
    ax[1].plot(1e9*t, 1e3*I_b1_sh2_junction, 'r--', label = 'result from manual ')

    for i in range(2):
        ax[i].set_ylabel(r'$I (t)\,[mA]$')
        ax[i].set_xlabel(r'$t\,[ns]$')
        ax[i].set_xticks(range(0, 60, 10))
        ax[i].grid('both')
        ax[i].legend()

    fig, ax = plt.subplots(2,1)
    fig.suptitle("Bundle S2 - shield 2")
    ax[0].plot(1e9*(p.probes["b2_junction_current"].t-1.5*dt), 1e3*p.probes["b2_junction_current"].val[:,0], label = 'Current on bundle2, outer shield. x = 0.0')
    ax[1].plot(1e9*(p.probes["b2_terminal_current"].t-1.5*dt), 1e3*p.probes["b2_terminal_current"].val[:,0], label = 'Current on bundle2, outer shield. x = 0.343')


    for i in range(2):
        ax[i].set_ylabel(r'$I (t)\,[mA]$')
        ax[i].set_xlabel(r'$t\,[ns]$')
        ax[i].set_xticks(range(0, 60, 10))
        ax[i].grid('both')
        ax[i].legend()

    fig, ax = plt.subplots(2,1)
    fig.suptitle("Bundle S3 - shield 2")
    ax[0].plot(1e9*(p.probes["b3_junction_current"].t-1.5*dt), 1e3*p.probes["b3_junction_current"].val[:,0], label = 'Current on bundle3, outer shield. x = 0')
    ax[1].plot(1e9*(p.probes["b3_terminal_current"].t-1.5*dt), 1e3*p.probes["b3_terminal_current"].val[:,0], label = 'Current on bundle3, outer shield. x = 0.165')

    for i in range(2):
        ax[i].set_ylabel(r'$I (t)\,[mA]$')
        ax[i].set_xlabel(r'$t\,[ns]$')
        ax[i].set_xticks(range(0, 60, 10))
        ax[i].grid('both')
        ax[i].legend()
        
    plt.show()
    
def test_7_bundles_B1_B2_B3():
    file = 'python/testData/parser/manual/test_7_bundles_single_conductors_B1_B2_B3.smb.json'

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
    file = 'python/testData/parser/manual/test_8_two_conductor_line_with_source.smb.json'
    p = Parser(file)
    p.run(finalTime=5e-9)
    
    tm, V1, V2, V3 =    np.genfromtxt('python/testData/manual/test8/ex8voltout', delimiter = ',',usecols=(0,1,2,3), unpack = True)
    
    
    plt.figure()
    plt.plot(1e9*p.probes["vR"].t, p.probes["vR"].val[:,1], label = 'End2 of C1')
    plt.plot(1e9*tm, V1, '--', label = 'manual')
    plt.ylabel(r'$V (t)\,[V]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 6, 1))
    # plt.yticks(range(-3, 4, 1))
    plt.xlim(0,4)
    plt.grid('both')
    plt.legend()

    plt.figure()
    plt.plot(1e9*p.probes["vL"].t, p.probes["vL"].val[:,0], label = 'End1 of C2')
    plt.plot(1e9*p.probes["vR"].t, p.probes["vR"].val[:,0], label = 'End2 of C2')
    plt.plot(1e9*tm, V2,  '--', label = 'manual')
    plt.plot(1e9*tm, V3,  '--', label = 'manual')
    plt.ylabel(r'$V (t)\,[V]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 6, 1))
    # plt.yticks(range(-3, 4, 1))
    plt.grid('both')
    plt.legend()
    
    plt.show()

def test_bundles_R_line_C():
    file = 'python/testData/parser/non_R_terminations/R_line_C.smb.json'

    p = Parser(file)
    p.run(finalTime=50e-9,dt=0.5e-10)


    t0, V2, V3 = np.genfromtxt('python/testData/ngspice/non_R_terminations/C_termination.txt', usecols=(0,1,3), unpack = True)

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
    file = 'python/testData/parser/non_R_terminations/R_line_LCp_Rs.smb.json'

    p = Parser(file)
    p.run(finalTime=50e-9,dt=0.5e-10)


    t0, V2, V3 = np.genfromtxt('python/testData/ngspice/non_R_terminations/LCp_Rs_termination.txt', usecols=(0,1,3), unpack = True)

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
    file = 'python/testData/parser/non_R_terminations/R_MTL_LCp_Rs.smb.json'

    p = Parser(file)
    p.run(finalTime=50e-9,dt=0.5e-10)


    t0, V3, V6 = np.genfromtxt('python/testData/ngspice/non_R_terminations/MTL_LCp_Rs_termination.txt', usecols=(0,3,7), unpack = True)

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

