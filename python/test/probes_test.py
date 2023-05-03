import numpy as np
import matplotlib.pyplot as plt

import src.mtln as mtln
from src.probes import *

import skrf as rf
from skrf.media import DistributedCircuit

def test_port_probe_s_extraction():

    L0 = 0.25e-6
    C0 = 100e-12
    length = 400.0
    Zs = 150.0
    Zl = 150.0
    
    fMin=0.01e6
    fMax=1e6
    finalTime = 25e-6
    
    def gauss(t):
        spread = 1/fMax/2.0
        delay = 8*spread
        return np.exp(- (t-delay)**2 / (2*spread**2))

    line = mtln.MTL(l=L0, c=C0, length=length, Zs=Zs, Zl=Zl)
    line.add_voltage_source(position=line.x[0], conductor=0, magnitude=gauss)
    p1 = line.add_port_probe(terminal=0, conductor=0)
    p2 = line.add_port_probe(terminal=1, conductor=0)
    line.run_until(finalTime)
    
    line_f, line_s = PortProbe.extract_s(p1, p2)
    fq = rf.Frequency.from_f(f=line_f[(line_f >= fMin) & (line_f < fMax)], unit='Hz')
    line_s = line_s[(line_f >= fMin) & (line_f < fMax)]    
    nw_mtln = rf.Network(frequency=fq, s=line_s)
    
    media = DistributedCircuit(fq, C=C0, L=L0)
    nw_skrf = media.line(length - line.dx/2.0, 'm', name='line', embed=True, z0=[Zs, Zl])

    R_S11  = np.corrcoef(np.abs(nw_mtln.s[:,0,0]), np.abs(nw_skrf.s[:,0,0]))
    R_S21  = np.corrcoef(np.abs(nw_mtln.s[:,1,0]), np.abs(nw_skrf.s[:,1,0]))

    assert(np.real(R_S11[1,1]) > 0.99999)
    assert(np.real(R_S21[1,1] ) > 0.99999)
    
    plt.figure()    
    nw_skrf.plot_s_mag(m=0, n=0, label='S11 skrf')
    nw_mtln.plot_s_mag(m=0, n=0, label='S11 mtln from s')
    nw_skrf.plot_s_mag(m=1, n=0, label='S21 skrf')
    nw_mtln.plot_s_mag(m=1, n=0, label='S21 mtln extract network')
    plt.legend()
    plt.show()