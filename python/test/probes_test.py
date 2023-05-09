import numpy as np
import matplotlib.pyplot as plt
import pickle

import src.mtl as mtl
from src.probes import *

import skrf as rf
from skrf.media import DistributedCircuit

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
    line.add_voltage_source(position=line.x[0], conductor=0, magnitude=gauss)
    p1 = line.add_port_probe(terminal=0)
    p2 = line.add_port_probe(terminal=1)
    line.run_until(finalTime)
    f, s = Port.extract_s_reciprocal(p1, p2) 
    
    fq = rf.Frequency.from_f(f=f[(f >= fMin) & (f < fMax)], unit='Hz')
    s = s[(f >= fMin) & (f < fMax)]
    nw_mtln = rf.Network(frequency=fq, s=s)

    media = DistributedCircuit(fq, C=C0, L=L0)
    nw_skrf = media.line(length - line.dx/2.0, 'm',
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

    f, s_p12 = Port.extract_s_reciprocal(pS, pL, 0, 0)
    s_p12 = s_p12[(f>=fMin)&(f<fMax),:]
    
    _, s_p14 = Port.extract_s_reciprocal(pS, pL, 0, 1)
    s_p14 = s_p14[(f>=fMin)&(f<fMax),:]

    _, s_p15 = Port.extract_s_reciprocal(pS, pS, 0, 2)
    s_p15 = s_p15[(f>=fMin)&(f<fMax),:]
    
    fq = rf.Frequency.from_f(f=f[(f>=fMin)&(f<fMax)], unit='Hz')
    mtln_12 = rf.Network(frequency=fq, s=s_p12)
    mtln_14 = rf.Network(frequency=fq, s=s_p14)
    mtln_15 = rf.Network(frequency=fq, s=s_p15)
    
    inta_12 = rf.Network(EXPERIMENTAL_DATA + 'Ch1P1Ch2P2-SParameters-Segmented.s2p').interpolate(fq)
    inta_14 = rf.Network(EXPERIMENTAL_DATA + 'Ch1P1Ch2P4-SParameters-Segmented.s2p').interpolate(fq)
    inta_26 = rf.Network(EXPERIMENTAL_DATA + 'Ch1P2Ch2P6-SParameters-Segmented.s2p').interpolate(fq)
    
    R_S11 = np.corrcoef(np.abs(mtln_12.s[:,0,0]), np.abs(inta_12.s[:,0,0]))
    R_S21 = np.corrcoef(np.abs(mtln_12.s[:,1,0]), np.abs(inta_12.s[:,1,0]))
    assert(R_S11[0,1] > 0.96)
    assert(R_S21[0,1] > 0.94)

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

    plt.figure()
    mtln_15.plot_s_db(m=0,n=0,label='S11 mtln')
    inta_26.plot_s_db(m=0,n=0,label='S22 INTA')
    mtln_15.plot_s_db(m=1,n=0,label='S15 mtln')
    inta_26.plot_s_db(m=1,n=0,label='S62 INTA')
    plt.xscale('log')
    plt.ylim(-60,0)

    plt.show()

    
