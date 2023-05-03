import numpy as np
import matplotlib.pyplot as plt

import src.mtln as mtln
from src.probes import *

import skrf as rf
from skrf.media import DistributedCircuit

def test_s_and_z_extraction():

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
    
    line_f, line_z = PortProbe.extract_z(p1)
    line_z = line_z[(line_f >= fMin) & (line_f < fMax)]

    line_f, line_s = PortProbe.extract_s(p1, p2)
    line_s = line_s[(line_f >= fMin) & (line_f < fMax)]

    fq = rf.Frequency.from_f(f=line_f[(line_f >= fMin) & (line_f < fMax)], unit='Hz')
    
    nw_mtln_from_z = rf.Network(frequency=fq, z=line_z)
    nw_mtln_from_s = rf.Network(frequency=fq, s=line_s)
    
    media = DistributedCircuit(fq, C=C0, L=L0)
    nw_skrf = media.line(length - line.dx/2.0, 'm', name='line', embed=True, z0=[Zs, Zl])
    
    plt.figure()    
    nw_skrf.plot_s_mag(m=0, n=0, label='S11 skrf')
    nw_mtln_from_s.plot_s_mag(m=0, n=0, label='S11 mtln from s')
    nw_mtln_from_z.plot_s_mag(m=0, n=0, label='S11 mtln from z')
    # nw_mtln.plot_s_mag(m=0, n=0, label='S11 mtln extract network')
    # nw_skrf.plot_s_mag(m=1, n=0, label='S21 skrf')
    # nw_mtln.plot_s_mag(m=1, n=0, label='S21 mtln extract network')

    # # nw_skrf.plot_s_mag(m=1, n=0, label='skrf')
    # nw_mtln_from_s.plot_s_mag(m=1, n=0, label='mtln from s')

    plt.legend()

    plt.show()

    # R_S11_fromS_fromZ = np.corrcoef(nw_mtln_from_s.s[:,0,0], nw_mtln_from_z.s[:,0,0])
    # R_S11_fromS_skrf  = np.corrcoef(nw_mtln_from_s.s[:,0,0], nw_skrf.s[:,0,0])
    # R_S21_fromS_skrf  = np.corrcoef(nw_mtln_from_s.s[:,1,0], nw_skrf.s[:,1,0])

    # assert(np.real(R_S11_fromS_fromZ[1,1]) > 0.99999)
    # assert(np.real(R_S11_fromS_skrf[1,1] ) > 0.99999)
    # assert(np.real(R_S21_fromS_skrf[1,1] ) > 0.99999)