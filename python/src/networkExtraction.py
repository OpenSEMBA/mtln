import numpy as np
import skrf as rf

from .probes import *
from .mtl import *

def extract_s_reciprocal(port_1, port_2):
    ''' 
    Extracts s-parameters. Only valid parameters are the ones related to the illuminated port.
    Reference: https://en.wikipedia.org/wiki/Scattering_parameters 
    '''
    p1 = port_1[0]
    cond1 = port_1[1]
    f, a1, b1 = p1.get_incident_and_reflected_power_wave(cond1)
    p2 = port_2[0]
    cond2 = port_2[1]
    _, a2, b2 = p2.get_incident_and_reflected_power_wave(cond2)
    s = np.zeros((len(f), 2, 2), dtype=complex)
    s[:, 0, 0] = b1/a1
    s[:, 1, 0] = b2/a1
    s[:, 0, 1] = b1/a2
    s[:, 1, 1] = b2/a2

    return f, s

def getSolvedProbesFromLine(line, p1, p2, magnitude, finalTime):
    p1_term = p1[0]
    p2_term = p2[0]
    p1_cond = p1[1]
    p2_cond = p2[1]

    line1 = line.create_clean_copy()
    if p1_term == "S":
        line1.add_voltage_source(
            position=line1.x[0], conductor=p1_cond, magnitude=magnitude)
    else:
        line1.add_voltage_source(
            position=line1.x[-1], conductor=p1_cond, magnitude=magnitude)
    pS_probe, pL_probe = line1.add_port_probes()
    if p1_term == "S":
        p1_probe = pS_probe
    else:
        p1_probe = pL_probe
    if p2_term == "S":
        p2_probe = pS_probe
    else:
        p2_probe = pL_probe
    line1.run_until(finalTime)
    
    return extract_s_reciprocal((p1_probe, p1_cond), (p2_probe, p2_cond))


def extract_2p_network(
        line, fMin, fMax, finalTime,
        p1=("S", 0), p2=("L", 0)):

    assert p1 != p2


    spread = 1/fMax/2.0
    delay = 8*spread

    def gauss(t):
        return np.exp(- (t-delay)**2 / (2*spread**2))

    f, s_line1 = getSolvedProbesFromLine(line, p1, p2, gauss, finalTime)
    f, s_line2 = getSolvedProbesFromLine(line, p2, p1, gauss, finalTime)

    s = np.zeros((len(f), 2, 2), dtype=complex)
    s[:, 0, 0] = s_line1[:, 0, 0]
    s[:, 1, 0] = s_line1[:, 1, 0]
    s[:, 0, 1] = s_line2[:, 1, 0]
    s[:, 1, 1] = s_line2[:, 0, 0]
    fq = rf.Frequency.from_f(f[(f >= fMin) & (f < fMax)], unit='Hz')
    return rf.Network(frequency=fq, s=s[(f >= fMin) & (f < fMax), :, :])
