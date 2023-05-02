import numpy as np
import matplotlib.pyplot as plt

import sympy as sp
from scipy.constants import epsilon_0, mu_0, speed_of_light

import src.mtln as mtln

import src.waveforms as wf

import skrf as rf
from skrf.media import DistributedCircuit

def test_mtln_2_tubes_short():
    """ 
    Analogous in Ch. 8 of Paul Clayton, 
    Analysis of Multiconductor Transmission Lines. 2007. 
    """
    """
    Two mtl with half lenght, shorted at the connection point
    """
    # line = mtl.MTL(l=0.25e-6, c=100e-12, length=400.0, Zs=150)
    # finalTime = 18e-6

    # def magnitude(t): return wf.square_pulse(t, 100, 6e-6)
    # line.add_voltage_source(position=0.0, conductor=0, magnitude=magnitude)
    # v_probe = line.add_probe(position=0.0, conductor=0, type='voltage')
    # i_probe = line.add_probe(position=400.0, conductor=0, type='current')

    # for t in line.get_time_range(finalTime):
    #     line.step()

    # xticks = range(int(np.floor(min(1e6*i_probe.t))),
    #                int(np.ceil(max(1e6*i_probe.t))+1))

    # start_times = [0.1, 4.1, 6.1, 8.1, 10.1, 12.1, 14.1, 16.1]
    # end_times = [3.9, 5.9, 7.9, 9.9, 11.9, 13.9, 15.9, 18.9]
    # check_voltages = [25, -12.5, -37.5, -18.75, 18.75, 9.375, -9.375, -4.6875]
    # for (t_start, t_end, v) in zip(start_times, end_times, check_voltages):
    #     start = np.argmin(np.abs(v_probe.t - t_start*1e-6))
    #     end = np.argmin(np.abs(v_probe.t - t_end*1e-6))
    #     assert np.all(np.isclose(v_probe.val[start:end], v))

    # plt.figure()
    # plt.plot(1e6*v_probe.t, v_probe.val)
    # plt.ylabel(r'$V (0, t)\,[V]$')
    # plt.xlabel(r'$t\,[\mu s]$')
    # plt.grid('both')

    # plt.figure()
    # plt.plot(1e6*i_probe.t, i_probe.val )
    # plt.ylabel(r'$I (L, t)\,[A]$')
    # plt.xlabel(r'$t\,[\mu s]$')
    # plt.xticks(xticks)
    # plt.grid('both')

    # plt.show()

def test_mtln_2_tubes_R():
    pass