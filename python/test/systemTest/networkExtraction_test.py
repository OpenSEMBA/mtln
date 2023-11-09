from src.probes import *
import pickle
import numpy as np
import matplotlib.pyplot as plt

import sympy as sp
from scipy.constants import epsilon_0, mu_0, speed_of_light

import src.mtl as mtl
from src.networkExtraction import *

import src.waveforms as wf

import skrf as rf
from skrf.media import DistributedCircuit

EXPERIMENTAL_DATA = 'python/testData/cable_panel/experimental_measurements/'


EXPERIMENTAL_DATA = 'python/testData/cable_panel/experimental_measurements/'
MTLN_RUNS_DATA = 'python/testData/cable_panel/mtln_runs'


def test_port_s_extraction():
    ''' S parmeter extraction from a single-conductor MTL '''
    L0 = 0.25e-6
    C0 = 100e-12
    length = 400.0
    Zs = 150.0
    Zl = 150.0

    fMin = 0.01e6
    fMax = 1e6
    finalTime = 25e-6

    def gauss(t):
        spread = 1/fMax/2.0
        delay = 8*spread
        return np.exp(- (t-delay)**2 / (2*spread**2))

    line = mtl.MTL(l=L0, c=C0, length=length, Zs=Zs, Zl=Zl)
    line.add_voltage_source(position=line.r[0], conductor=0, magnitude=gauss)
    p1 = line.add_port_probe(terminal=0)
    p2 = line.add_port_probe(terminal=1)
    line.run_until(finalTime)
    f, s = extract_s_reciprocal((p1, 0), (p2, 0))

    fq = rf.Frequency.from_f(f=f[(f >= fMin) & (f < fMax)], unit='Hz')
    s = s[(f >= fMin) & (f < fMax)]
    nw_mtln = rf.Network(frequency=fq, s=s)

    media = DistributedCircuit(fq, C=C0, L=L0)
    nw_skrf = media.line(length - line.dz/2.0, 'm',
                         name='line', embed=True, z0=[Zs, Zl])

    R_S11 = np.corrcoef(np.abs(nw_mtln.s[:, 0, 0]), np.abs(nw_skrf.s[:, 0, 0]))
    assert (np.real(R_S11[0, 1]) > 0.99999)
    R_S21 = np.corrcoef(np.abs(nw_mtln.s[:, 1, 0]), np.abs(nw_skrf.s[:, 1, 0]))
    assert (np.real(R_S21[0, 1]) > 0.999)

    # plt.figure()
    # nw_skrf.plot_s_mag(m=0, n=0, label='S11 skrf')
    # nw_mtln.plot_s_mag(m=0, n=0, label='S11 mtln')
    # nw_skrf.plot_s_mag(m=1, n=0, label='S21 skrf')
    # nw_mtln.plot_s_mag(m=1, n=0, label='S21 mtln')
    # plt.legend()
    # plt.show()


def test_cables_panel_s_extraction():
    fn = open(MTLN_RUNS_DATA + '/cable_panel_ports.pkl', 'rb')
    pS = pickle.load(fn)
    pL = pickle.load(fn)

    fMin = 10e6
    fMax = 1e9

    f, s_p12 = extract_s_reciprocal((pS, 0), (pL, 0))
    s_p12 = s_p12[(f >= fMin) & (f < fMax), :]

    _, s_p14 = extract_s_reciprocal((pS, 0), (pL, 1))
    s_p14 = s_p14[(f >= fMin) & (f < fMax), :]

    _, s_p15 = extract_s_reciprocal((pS, 0), (pS, 2))
    s_p15 = s_p15[(f >= fMin) & (f < fMax), :]

    fq = rf.Frequency.from_f(f=f[(f >= fMin) & (f < fMax)], unit='Hz')
    mtln_12 = rf.Network(frequency=fq, s=s_p12)
    mtln_14 = rf.Network(frequency=fq, s=s_p14)
    mtln_15 = rf.Network(frequency=fq, s=s_p15)

    inta_12 = rf.Network(
        EXPERIMENTAL_DATA + 'Ch1P1Ch2P2-SParameters-Segmented.s2p').interpolate(fq)
    inta_14 = rf.Network(
        EXPERIMENTAL_DATA + 'Ch1P1Ch2P4-SParameters-Segmented.s2p').interpolate(fq)
    inta_26 = rf.Network(
        EXPERIMENTAL_DATA + 'Ch1P2Ch2P6-SParameters-Segmented.s2p').interpolate(fq)

    R_S11 = np.corrcoef(np.abs(mtln_12.s[:, 0, 0]), np.abs(inta_12.s[:, 0, 0]))
    R_S21 = np.corrcoef(np.abs(mtln_12.s[:, 1, 0]), np.abs(inta_12.s[:, 1, 0]))
    assert (R_S11[0, 1] > 0.96)
    assert (R_S21[0, 1] > 0.94)

    # plt.figure()
    # mtln_12.plot_s_db(m=0,n=0,label='S11 mtln')
    # inta_12.plot_s_db(m=0,n=0,label='S11 INTA')
    # mtln_12.plot_s_db(m=1,n=0,label='S21 mtln')
    # inta_12.plot_s_db(m=1,n=0,label='S21 INTA')
    # plt.xscale('log')
    # plt.ylim(-40,0)

    # plt.figure()
    # mtln_14.plot_s_db(m=0,n=0,label='S11 mtln')
    # inta_14.plot_s_db(m=0,n=0,label='S11 INTA')
    # mtln_14.plot_s_db(m=1,n=0,label='S41 mtln')
    # inta_14.plot_s_db(m=1,n=0,label='S41 INTA')
    # plt.xscale('log')
    # plt.ylim(-40,0)

    # plt.figure()
    # mtln_15.plot_s_db(m=0,n=0,label='S11 mtln')
    # inta_26.plot_s_db(m=0,n=0,label='S22 INTA')
    # mtln_15.plot_s_db(m=1,n=0,label='S15 mtln')
    # inta_26.plot_s_db(m=1,n=0,label='S62 INTA')
    # plt.xscale('log')
    # plt.ylim(-60,0)

    # plt.show()


def test_extract_network_paul_8_6_150ohm_load():

    L0 = 0.25e-6
    C0 = 100e-12
    length = 400.0
    Zs = 150.0
    Zl = 150.0

    line = mtl.MTL(l=L0, c=C0, length=length, Zs=Zs, Zl=Zl)
    line_ntw = extract_2p_network(
        line, fMin=0.01e6, fMax=1e6, finalTime=250e-6)

    media = DistributedCircuit(line_ntw.frequency, C=C0, L=L0)
    skrf_ntw = media.line(length - line.dz/2.0, 'm',
                          name='line', embed=True, z0=[Zs, Zl])

    R_S11 = np.corrcoef(
        np.abs(line_ntw.s[:, 0, 0]), np.abs(skrf_ntw.s[:, 0, 0]))
    R_S22 = np.corrcoef(
        np.abs(line_ntw.s[:, 1, 1]), np.abs(skrf_ntw.s[:, 1, 1]))

    assert (np.real(R_S11[0, 1]) > 0.9999)
    assert (np.real(R_S22[0, 1]) > 0.9999)

    # plt.figure()
    # skrf_ntw.plot_s_db(m=0, n=0, label='S11 skrf')
    # line_ntw.plot_s_db(m=0, n=0, label='S11 mtl')
    # skrf_ntw.plot_s_db(m=1, n=1, label='S22 skrf')
    # line_ntw.plot_s_db(m=1, n=1, label='S22 mtl')
    # plt.grid()
    # plt.legend()

    # plt.figure()
    # skrf_tl.plot_s_deg(m=0, n=0, label='skrf')
    # line_ntw.plot_s_deg(m=0, n=0, label='mtl')
    # plt.grid()
    # plt.legend()

    # plt.show()


def test_cables_panel_experimental_comparison():
    # Gets L and C matrices from SACAMOS cable_panel_4cm.bundle
    L = np.array(
        [[7.92796549E-07,  1.25173387E-07,  4.84953816E-08],
         [1.25173387E-07,  1.01251901E-06,  1.25173387E-07],
         [4.84953816E-08,  1.25173387E-07,  1.00276097E-06]])
    C = np.array(
        [[1.43342565E-11, -1.71281372E-12, -4.79422869E-13],
         [-1.71281372E-12,  1.13658354E-11, -1.33594804E-12],
         [-4.79422869E-13, -1.33594804E-12,  1.12858157E-11]])




    length = 398e-3
    Zs = np.ones([1, 3]) * 50.0
    Zl = Zs
    line = mtl.MTL(l=L, c=C, length=length, Zs=Zs, Zl=Zl)

    finalTime = 3000e-9
    line.dt = line.dt
    mtln_ntw = extract_2p_network(
        line,
        fMin=1e7, fMax=1e9, finalTime=finalTime,
        p1=("S", 0),
        p2=("L", 0)
    )

    meas_ntw = rf.Network(
        EXPERIMENTAL_DATA + 'Ch1P1Ch2P2-SParameters-Segmented.s2p'
    ).interpolate(mtln_ntw.frequency)

    R_S11 = np.corrcoef(np.abs(meas_ntw.s[:, 0, 0]), np.abs(mtln_ntw.s[:, 0, 0]))
    assert (R_S11[0, 1] > 0.96)

    plt.figure()
    meas_ntw.plot_s_db(m=0, n=0, label='meas.')
    mtln_ntw.plot_s_db(m=0, n=0, label='mtln')
    plt.grid()
    plt.legend()
    plt.xlim(1e7, 1e9)
    plt.xscale('log')
    plt.show()
    
def test_cables_panel_ferrite_experimental_comparison_S12():
    # Gets L and C matrices from SACAMOS cable_panel_4cm.bundle
    L = np.array(
        [[7.92796549E-07,  1.25173387E-07,  4.84953816E-08],
         [1.25173387E-07,  1.01251901E-06,  1.25173387E-07],
         [4.84953816E-08,  1.25173387E-07,  1.00276097E-06]])
    C = np.array(
        [[1.43342565E-11, -1.71281372E-12, -4.79422869E-13],
         [-1.71281372E-12,  1.13658354E-11, -1.33594804E-12],
         [-4.79422869E-13, -1.33594804E-12,  1.12858157E-11]])

    length = 398e-3
    Zs = np.ones([1, 3]) * 50.0
    Zl = Zs
    # line = mtl.MTL(l=L, c=C, length=length, Zs=Zs, Zl=Zl)

    line = mtl.MTL_losses(l=L, c=C, g=0.0, r=0.0, length=length,Zs=Zs, Zl=Zl)

    poles    = np.array([-1.48535971e+10       +0.j       ,
                         -2.29904404e+08       +0.j       ,
                         -6.73258119e+07       +0.j       ,
                         -1.21476603e+08       +0.j       ,
                         -7.40969289e+05+44929341.8519292j,
                         -7.40969289e+05-44929341.8519292j])

    residues = np.array([-9.32340577e+11      +0.j        ,
                          7.81078516e+09      +0.j        ,
                          9.49567177e+08      +0.j        ,
                          -2.28755419e+10      +0.j        ,
                          0.5*(2.98804764e+05+1349305.87476208j),
                          0.5*(2.98804764e+05-1349305.87476208j)])
    D = 203.08780084
    E = 0
    line.add_dispersive_connector(position = 0.5*length, 
                                conductor=0,
                                d=D/line.dx,
                                e=E/line.dx,
                                poles=poles, 
                                residues=residues/line.dx)
    finalTime = 3000e-9
    mtln_ntw = extract_2p_network(
        line,
        fMin=1e7, fMax=1e9, finalTime=finalTime,
        p1=("S", 0),
        p2=("L", 0)
    )
    mtln_ntw.write_touchstone('mtln_no_ferrite_p12.s2p')
    meas_ntw = rf.Network(
        EXPERIMENTAL_DATA + 'Ch1P1Ch2P2-SParameters-7427009-Middle-Segmented.s2p'
    ).interpolate(mtln_ntw.frequency)

    # R_S11 = np.corrcoef(np.abs(meas_ntw.s[:, 0, 0]), np.abs(mtln_ntw.s[:, 0, 0]))
    # assert (R_S11[0, 1] > 0.96)

    plt.figure()
    plt.title('finalTime = '+str(finalTime))
    meas_ntw.plot_s_db(m=0, n=0, label='meas.')
    mtln_ntw.plot_s_db(m=0, n=0, label='mtln')
    plt.grid()
    plt.legend()
    plt.xlim(1e7, 1e9)
    plt.xscale('log')
    plt.savefig('mtln_no_ferrite_p12.png')

def test_cables_panel_ferrite_experimental_comparison_S14():
    # Gets L and C matrices from SACAMOS cable_panel_4cm.bundle
    L = np.array(
        [[7.92796549E-07,  1.25173387E-07,  4.84953816E-08],
         [1.25173387E-07,  1.01251901E-06,  1.25173387E-07],
         [4.84953816E-08,  1.25173387E-07,  1.00276097E-06]])
    C = np.array(
        [[1.43342565E-11, -1.71281372E-12, -4.79422869E-13],
         [-1.71281372E-12,  1.13658354E-11, -1.33594804E-12],
         [-4.79422869E-13, -1.33594804E-12,  1.12858157E-11]])

    length = 398e-3
    Zs = np.ones([1, 3]) * 50.0
    Zl = Zs
    # line = mtl.MTL(l=L, c=C, length=length, Zs=Zs, Zl=Zl)

    line = mtl.MTL_losses(l=L, c=C, g=0.0, r=0.0, length=length,Zs=Zs, Zl=Zl)

    poles    = np.array([-1.48535971e+10       +0.j       ,
                         -2.29904404e+08       +0.j       ,
                         -6.73258119e+07       +0.j       ,
                         -1.21476603e+08       +0.j       ,
                         -7.40969289e+05+44929341.8519292j,
                         -7.40969289e+05-44929341.8519292j])

    residues = np.array([-9.32340577e+11      +0.j        ,
                          7.81078516e+09      +0.j        ,
                          9.49567177e+08      +0.j        ,
                          -2.28755419e+10      +0.j        ,
                          0.5*(2.98804764e+05+1349305.87476208j),
                          0.5*(2.98804764e+05-1349305.87476208j)])
    D = 203.08780084
    E = 0
    line.add_dispersive_connector(position = 0.5*length, 
                                conductor=0,
                                d=D/line.dx,
                                e=E/line.dx,
                                poles=poles, 
                                residues=residues/line.dx)
    finalTime = 3000e-9
    mtln_ntw = extract_2p_network(
        line,
        fMin=1e7, fMax=1e9, finalTime=finalTime,
        p1=("S", 0),
        p2=("L", 1)
    )
    mtln_ntw.write_touchstone('mtln_no_ferrite_p14.s2p')
    meas_ntw = rf.Network(
        EXPERIMENTAL_DATA + 'Ch1P1Ch2P2-SParameters-7427009-Middle-Segmented.s2p'
    ).interpolate(mtln_ntw.frequency)

    # R_S11 = np.corrcoef(np.abs(meas_ntw.s[:, 0, 0]), np.abs(mtln_ntw.s[:, 0, 0]))
    # assert (R_S11[0, 1] > 0.96)

    plt.figure()
    plt.title('finalTime = '+str(finalTime))
    meas_ntw.plot_s_db(m=1, n=0, label='meas.')
    mtln_ntw.plot_s_db(m=1, n=0, label='mtln')
    plt.grid()
    plt.legend()
    plt.xlim(1e7, 1e9)
    plt.xscale('log')
    plt.savefig('mtln_no_ferrite_p14.png')

def test_cables_panel_ferrite_experimental_comparison_S26():
    # Gets L and C matrices from SACAMOS cable_panel_4cm.bundle
    L = np.array(
        [[7.92796549E-07,  1.25173387E-07,  4.84953816E-08],
         [1.25173387E-07,  1.01251901E-06,  1.25173387E-07],
         [4.84953816E-08,  1.25173387E-07,  1.00276097E-06]])
    C = np.array(
        [[1.43342565E-11, -1.71281372E-12, -4.79422869E-13],
         [-1.71281372E-12,  1.13658354E-11, -1.33594804E-12],
         [-4.79422869E-13, -1.33594804E-12,  1.12858157E-11]])

    length = 398e-3
    Zs = np.ones([1, 3]) * 50.0
    Zl = Zs
    # line = mtl.MTL(l=L, c=C, length=length, Zs=Zs, Zl=Zl)

    line = mtl.MTL_losses(l=L, c=C, g=0.0, r=0.0, length=length,Zs=Zs, Zl=Zl)

    poles    = np.array([-1.48535971e+10       +0.j       ,
                         -2.29904404e+08       +0.j       ,
                         -6.73258119e+07       +0.j       ,
                         -1.21476603e+08       +0.j       ,
                         -7.40969289e+05+44929341.8519292j,
                         -7.40969289e+05-44929341.8519292j])

    residues = np.array([-9.32340577e+11      +0.j        ,
                          7.81078516e+09      +0.j        ,
                          9.49567177e+08      +0.j        ,
                          -2.28755419e+10      +0.j        ,
                          0.5*(2.98804764e+05+1349305.87476208j),
                          0.5*(2.98804764e+05-1349305.87476208j)])
    D = 203.08780084
    E = 0
    line.add_dispersive_connector(position = 0.5*length, 
                                conductor=0,
                                d=D/line.dx,
                                e=E/line.dx,
                                poles=poles, 
                                residues=residues/line.dx)
    finalTime = 3000e-9
    mtln_ntw = extract_2p_network(
        line,
        fMin=1e7, fMax=1e9, finalTime=finalTime,
        p1=("L", 0),
        p2=("L", 2)
    )
    mtln_ntw.write_touchstone('mtln_no_ferrite_p26.s2p')
    meas_ntw = rf.Network(
        EXPERIMENTAL_DATA + 'Ch1P1Ch2P2-SParameters-7427009-Middle-Segmented.s2p'
    ).interpolate(mtln_ntw.frequency)

    # R_S11 = np.corrcoef(np.abs(meas_ntw.s[:, 0, 0]), np.abs(mtln_ntw.s[:, 0, 0]))
    # assert (R_S11[0, 1] > 0.96)

    plt.figure()
    plt.title('finalTime = '+str(finalTime))
    meas_ntw.plot_s_db(m=1, n=0, label='meas.')
    mtln_ntw.plot_s_db(m=1, n=0, label='mtln')
    plt.grid()
    plt.legend()
    plt.xlim(1e7, 1e9)
    plt.xscale('log')
    plt.savefig('mtln_no_ferrite_p26.png')
