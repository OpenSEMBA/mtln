import numpy as np
import matplotlib.pyplot as plt

import sympy as sp

import src.mtl as mtl
from src.networkExtraction import *

import src.waveforms as wf

from src.jsonParser import Parser

import skrf as rf
from skrf.media import Coaxial


def test_cables_panel_with_empty_dispersive():
    # Gets L and C matrices from SACAMOS cable_panel_4cm.bundle
    L = np.array( \
        [[  7.92796549E-07,  1.25173387E-07,  4.84953816E-08],
         [  1.25173387E-07,  1.01251901E-06,  1.25173387E-07],
         [  4.84953816E-08,  1.25173387E-07,  1.00276097E-06]])
    C = np.array( \
        [[  1.43342565E-11, -1.71281372E-12, -4.79422869E-13],
         [ -1.71281372E-12,  1.13658354E-11, -1.33594804E-12],
         [ -4.79422869E-13, -1.33594804E-12,  1.12858157E-11]])

    G = np.zeros([3,3])
    R = np.zeros([3,3])
    # Models MTL amd extracts S11.
    length = 398e-3
    Zs = np.ones([1, 3]) * 50.0
    Zl = Zs
    line = mtl.MTL(l=L, c=C, g=G, r=R, length=length, Zs=Zs, Zl=Zl)
    
    poles = np.array([])    
    residues = np.array([])    
    D = 0
    E = 0
    line.connectors.add_dispersive_connector(position = 200e-3, 
                                conductor=0,
                                d=D,
                                e=E,
                                poles=poles, 
                                residues=residues)

    finalTime = 300e-9
    
    line_ntw = extract_2p_network(line, fMin=1e7, fMax=1e9, finalTime=finalTime)

    
    p1p2 = rf.subnetwork(
        rf.Network(
        'python/testData/cable_panel/experimental_measurements/Ch1P1Ch2P2-SParameters-Segmented.s2p'), [0])
    p1p2 = p1p2.interpolate(line_ntw.frequency)

    
    # Asserts correlation with [S11| measurements.
    R = np.corrcoef(np.abs(p1p2.s[:,0,0]), np.abs(line_ntw.s[:,0,0]))
    assert(R[0,1] >= 0.96)

    plt.figure()
    p1p2.plot_s_mag(label='measurements')
    line_ntw.plot_s_mag(label='mtl')
    plt.grid()
    plt.legend()
    plt.xlim(1e7, 1e9)
    plt.xscale('log')
    plt.show()

def test_line_lumped_R():
    
    file = 'python/testData/dispersive/line_lumped_R.smb.json'
    p = Parser(file)
    dt = min([b.dt for b in p.bundles])
    p.run(finalTime=1e-6)
    
    t, V, I = np.genfromtxt('python/testData/dispersive/line_lumped_R.txt', usecols=(0,1,3), unpack = True)


    V_resampled = np.interp(p.probes["v1"].t, t, V)
    I_resampled = np.interp(p.probes["i1"].t, t, I)

    assert(np.allclose(V_resampled[:-1], p.probes["v1"].val[1:,0], atol=5e-4))
    assert(np.allclose(I_resampled[:-1], p.probes["i1"].val[1:,0], atol=5e-5))

    # plt.figure()
    # plt.plot(1e6*p.probes["v1"].t, p.probes["v1"].val[:,0], label = 'Voltage on line end')
    # plt.plot(1e6*t, V, label = 'Voltage on line end - Spice')
    # plt.plot(1e6*p.probes["v1"].t, V_resampled, label = 'Voltage on line end - Spice - resampled')
    # plt.ylabel(r'$V (t)\,[V]$')
    # plt.xlabel(r'$t\,[us]$')
    # plt.grid('both')
    # plt.legend()

    # plt.figure()
    # plt.plot(1e6*p.probes["i1"].t, 1e3*p.probes["i1"].val[:,0], label = 'Current on line end')
    # plt.plot(1e6*t, 1e3*I, label = 'Current on line end - Spice')
    # plt.ylabel(r'$I (t)\,[mA]$')
    # plt.xlabel(r'$t\,[us]$')
    # plt.grid('both')
    # plt.legend()

    # plt.show()

    # I = np.fft.fftshift(np.fft.fft(p.probes["i1"].val[:,0]))
    # If = np.fft.fftshift(np.fft.fftfreq(len(p.probes["i1"].val[:,0]))/dt)
    # V = np.fft.fftshift(np.fft.fft(p.probes["v1"].val[:,0]-p.probes["v0"].val[:,0]))
    # Vf = np.fft.fftshift(np.fft.fftfreq(len(p.probes["v1"].val[:,0]))/dt)
    
    
    # n = np.where(Vf>=0)[0][0]
    # plt.xlabel(r'$f\,[Hz]$')
    # plt.loglog(Vf[n:], np.abs(V/I)[n:], label = 'Z')
    # plt.grid('both')
    # plt.ylim(0, 500)

    
def test_line_lumped_RLs():
    
    file = 'python/testData/dispersive/line_lumped_RLs.smb.json'
    p = Parser(file)
    dt = min([b.dt for b in p.bundles])
    p.run(finalTime=1e-6)
    
    t, V, I = np.genfromtxt('python/testData/dispersive/line_lumped_RLs.txt', usecols=(0,1,3), unpack = True)


    V_resampled = np.interp(p.probes["v1"].t, t, V)
    I_resampled = np.interp(p.probes["i1"].t, t, I)

    assert(np.allclose(V_resampled[:-1], p.probes["v1"].val[1:,0], atol=5e-2))
    assert(np.allclose(I_resampled[:-1], p.probes["i1"].val[1:,0], atol=5e-3))

    # plt.figure()
    # plt.plot(1e6*p.probes["v1"].t, p.probes["v1"].val[:,0], label = 'Voltage on line end')
    # plt.plot(1e6*t, V, label = 'Voltage on line end - Spice')
    # plt.plot(1e6*p.probes["v1"].t, V_resampled, label = 'Voltage on line end - Spice - resampled')
    # plt.ylabel(r'$V (t)\,[V]$')
    # plt.xlabel(r'$t\,[us]$')
    # plt.grid('both')
    # plt.legend()

    # plt.figure()
    # plt.plot(1e6*p.probes["i1"].t, 1e3*p.probes["i1"].val[:,0], label = 'Current on line end')
    # plt.plot(1e6*t, 1e3*I, label = 'Current on line end - Spice')
    # plt.ylabel(r'$I (t)\,[mA]$')
    # plt.xlabel(r'$t\,[us]$')
    # plt.grid('both')
    # plt.legend()

    # plt.show()

    # I = np.fft.fftshift(np.fft.fft(p.probes["i1"].val[:,0]))
    # If = np.fft.fftshift(np.fft.fftfreq(len(p.probes["i1"].val[:,0]))/dt)
    # V = np.fft.fftshift(np.fft.fft(p.probes["v1"].val[:,0]-p.probes["v0"].val[:,0]))
    # Vf = np.fft.fftshift(np.fft.fftfreq(len(p.probes["v1"].val[:,0]))/dt)
    
    
    # n = np.where(Vf>=0)[0][0]
    # plt.xlabel(r'$f\,[Hz]$')
    # plt.loglog(Vf[n:], np.abs(V/I)[n:], label = 'Z')
    # plt.grid('both')
    # plt.ylim(0, 500)

def test_line_lumped_RCp():
    
    file = 'python/testData/dispersive/line_lumped_RCp.smb.json'
    p = Parser(file)
    dt = min([b.dt for b in p.bundles])
    p.run(finalTime=1e-6)
    
    t, V, I = np.genfromtxt('python/testData/dispersive/line_lumped_RCp.txt', usecols=(0,1,3), unpack = True)


    V_resampled = np.interp(p.probes["v1"].t, t, V)
    I_resampled = np.interp(p.probes["i1"].t, t, I)

    assert(np.allclose(V_resampled[:-1], p.probes["v1"].val[1:,0], atol=5e-4))
    assert(np.allclose(I_resampled[:-1], p.probes["i1"].val[1:,0], atol=5e-5))

    # plt.figure()
    # plt.plot(1e6*p.probes["v1"].t, p.probes["v1"].val[:,0], label = 'Voltage on line end')
    # plt.plot(1e6*t, V, label = 'Voltage on line end - Spice')
    # plt.plot(1e6*p.probes["v1"].t, V_resampled, label = 'Voltage on line end - Spice - resampled')
    # plt.ylabel(r'$V (t)\,[V]$')
    # plt.xlabel(r'$t\,[us]$')
    # plt.grid('both')
    # plt.legend()

    # plt.figure()
    # plt.plot(1e6*p.probes["i1"].t, 1e3*p.probes["i1"].val[:,0], label = 'Current on line end')
    # plt.plot(1e6*t, 1e3*I, label = 'Current on line end - Spice')
    # plt.ylabel(r'$I (t)\,[mA]$')
    # plt.xlabel(r'$t\,[us]$')
    # plt.grid('both')
    # plt.legend()

    # plt.show()

    # I = np.fft.fftshift(np.fft.fft(p.probes["i1"].val[:,0]))
    # If = np.fft.fftshift(np.fft.fftfreq(len(p.probes["i1"].val[:,0]))/dt)
    # V = np.fft.fftshift(np.fft.fft(p.probes["v1"].val[:,0]-p.probes["v0"].val[:,0]))
    # Vf = np.fft.fftshift(np.fft.fftfreq(len(p.probes["v1"].val[:,0]))/dt)
    
    
    # n = np.where(Vf>=0)[0][0]
    # plt.xlabel(r'$f\,[Hz]$')
    # plt.loglog(Vf[n:], np.abs(V/I)[n:], label = 'Z')
    # plt.grid('both')
    # plt.ylim(0, 500)

def test_line_lumped_complex_pole():
    
    file = 'python/testData/dispersive/line_lumped_complex_pole.smb.json'
    p = Parser(file)
    dt = min([b.dt for b in p.bundles])
    p.run(finalTime=1e-6)
    
    t, V, I = np.genfromtxt('python/testData/dispersive/line_lumped_complex_pole.txt', usecols=(0,1,3), unpack = True)


    V_resampled = np.interp(p.probes["v1"].t, t, V)
    I_resampled = np.interp(p.probes["i1"].t, t, I)

    # assert(np.allclose(V_resampled[:-1], p.probes["v1"].val[1:,0], atol=5e-4))
    # assert(np.allclose(I_resampled[:-1], p.probes["i1"].val[1:,0], atol=5e-5))

    plt.figure()
    plt.plot(1e6*p.probes["v1"].t, p.probes["v1"].val[:,0], label = 'Voltage on line end')
    plt.plot(1e6*t, V,  '--', label = 'Voltage on line end - Spice')
    # plt.plot(1e6*p.probes["v1"].t, V_resampled, label = 'Voltage on line end - Spice - resampled')
    plt.ylabel(r'$V (t)\,[V]$')
    plt.xlabel(r'$t\,[us]$')
    plt.grid('both')
    plt.legend()

    plt.figure()
    plt.plot(1e6*p.probes["i1"].t, 1e3*p.probes["i1"].val[:,0], label = 'Current on line end')
    plt.plot(1e6*t, 1e3*I, '--', label = 'Current on line end - Spice')
    plt.ylabel(r'$I (t)\,[mA]$')
    plt.xlabel(r'$t\,[us]$')
    plt.grid('both')
    plt.legend()

    plt.show()

    # I = np.fft.fftshift(np.fft.fft(p.probes["i1"].val[:,0]))
    # If = np.fft.fftshift(np.fft.fftfreq(len(p.probes["i1"].val[:,0]))/dt)
    # V = np.fft.fftshift(np.fft.fft(p.probes["v1"].val[:,0]-p.probes["v0"].val[:,0]))
    # Vf = np.fft.fftshift(np.fft.fftfreq(len(p.probes["v1"].val[:,0]))/dt)
    
    
    # n = np.where(Vf>=0)[0][0]
    # plt.xlabel(r'$f\,[Hz]$')
    # plt.loglog(Vf[n:], np.abs(V/I)[n:], label = 'Z')
    # plt.grid('both')
    # plt.ylim(0, 500)

