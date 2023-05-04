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

    line_f, line_s = Port.extract_s(p1, p2)
    fq = rf.Frequency.from_f(
        f=line_f[(line_f >= fMin) & (line_f < fMax)], unit='Hz')
    line_s = line_s[(line_f >= fMin) & (line_f < fMax)]
    nw_mtln = rf.Network(frequency=fq, s=line_s)

    media = DistributedCircuit(fq, C=C0, L=L0)
    nw_skrf = media.line(length - line.dx/2.0, 'm',
                         name='line', embed=True, z0=[Zs, Zl])

    R_S11 = np.corrcoef(np.abs(nw_mtln.s[:, 0, 0]), np.abs(nw_skrf.s[:, 0, 0]))
    R_S21 = np.corrcoef(np.abs(nw_mtln.s[:, 1, 0]), np.abs(nw_skrf.s[:, 1, 0]))

    assert (np.real(R_S11[1, 1]) > 0.99999)
    assert (np.real(R_S21[1, 1]) > 0.99999)

    # plt.figure()
    # nw_skrf.plot_s_mag(m=0, n=0, label='S11 skrf')
    # nw_mtln.plot_s_mag(m=0, n=0, label='S11 mtln from s')
    # nw_skrf.plot_s_mag(m=1, n=0, label='S21 skrf')
    # nw_mtln.plot_s_mag(m=1, n=0, label='S21 mtln extract network')
    # plt.legend()
    # plt.show()


def test_cables_panel_s_extraction():
    fn = open(MTLN_RUNS_DATA + '/cable_panel_ports.pkl', 'rb')
    pS = pickle.load(fn)
    pL = pickle.load(fn)

    fMin = 10e6
    fMax = 1e9

    line_f, line_s = Port.extract_s(pS, pL)
    line_s = line_s[(line_f>=fMin)&(line_f<fMax),:]
    fq = rf.Frequency.from_f(f=line_f[(line_f>=fMin)&(line_f<fMax)], unit='Hz')
    mtln = rf.Network(frequency=fq, s=line_s)
    
    inta = rf.Network(EXPERIMENTAL_DATA + 'Ch1P1Ch2P2-SParameters-Segmented.s2p')
    inta = inta.interpolate(mtln.frequency)

    R_S11 = np.corrcoef(np.abs(mtln.s[:,0,0]), np.abs(inta.s[:,0,0]))
    R_S21 = np.corrcoef(np.abs(mtln.s[:,1,0]), np.abs(inta.s[:,1,0]))
    assert(R_S11[0,1] > 0.96)
    assert(R_S21[0,1] > 0.94)

    # plt.figure()
    # mtln.plot_s_db(m=0,n=0,label='S11 mtln')
    # inta.plot_s_db(m=0,n=0,label='S11 INTA')
    # mtln.plot_s_db(m=1,n=0,label='S21 mtln')
    # inta.plot_s_db(m=1,n=0,label='S21 INTA')
    # plt.xscale('log')
    # plt.ylim(-40,0)

    # plt.figure()
    # mtln.plot_s_deg(m=0,n=0,label='S11 mtln')
    # inta.plot_s_deg(m=0,n=0,label='S11 INTA')
    # mtln.plot_s_deg(m=1,n=0,label='S21 mtln')
    # inta.plot_s_deg(m=1,n=0,label='S21 INTA')
    # plt.xscale('log')

    # plt.show()

    
