import numpy as np
import matplotlib.pyplot as plt

import sympy as sp
from scipy.constants import epsilon_0, mu_0, speed_of_light

import src.mtl as mtl

import src.waveforms as wf

import skrf as rf
from skrf.media import DistributedCircuit

EXPERIMENTAL_DATA = 'python/testData/cable_panel/experimental_measurements/'

def fail():
    assert(True==False)

def test_get_phase_velocities():
    v = mtl.MTL(l=0.25e-6, c=100e-12, nx=100).get_phase_velocities()
    assert v.shape == (100,1)
    assert np.isclose(2e8, v[:][0])

    v = mtl.MTL(l=0.25e-6, c=100e-12, nx=50).get_phase_velocities()
    assert v.shape == (50,1)
    assert np.isclose(2e8, v[:][0])


def test_coaxial_line_paul_8_6_square():
    """ 
    Described in Ch. 8 of Paul Clayton, 
    Analysis of Multiconductor Transmission Lines. 2004. 
    """

    line = mtl.MTL(l=0.25e-6, c=100e-12, length=400.0, Zs=150)
    finalTime = 18e-6

    def magnitude(t): return wf.square_pulse(t, 100, 6e-6)
    line.add_voltage_source(position=0.0, conductor=0, magnitude=magnitude)
    v_probe = line.add_probe(position=0.0, type='voltage')
    i_probe = line.add_probe(position=400.0, type='current')

    line.run_until(finalTime)

    xticks = range(int(np.floor(min(1e6*i_probe.t))),
                   int(np.ceil(max(1e6*i_probe.t))+1))

    start_times = [0.1, 4.1, 6.1, 8.1, 10.1, 12.1, 14.1, 16.1]
    end_times = [3.9, 5.9, 7.9, 9.9, 11.9, 13.9, 15.9, 18.9]
    check_voltages = [25, -12.5, -37.5, -18.75, 18.75, 9.375, -9.375, -4.6875]
    for (t_start, t_end, v) in zip(start_times, end_times, check_voltages):
        start = np.argmin(np.abs(v_probe.t - t_start*1e-6))
        end = np.argmin(np.abs(v_probe.t - t_end*1e-6))
        assert np.all(np.isclose(v_probe.val[start:end], v))

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


def test_symmetry_in_voltage_excitation():
    """ 
    Test results are identical when exciting from S or from L.
    """
    def magnitude(t): return wf.square_pulse(t, 100, 6e-6)
    finalTime = 18e-6

    line_vs = mtl.MTL(l=0.25e-6, c=100e-12, length=400.0, Zs=150, Zl=150)
    line_vs.add_voltage_source(position=0.0, conductor=0, magnitude=magnitude)
    vs_probe = line_vs.add_probe(position=0.0, type='voltage')
    line_vs.run_until(finalTime)

    line_vl = mtl.MTL(l=0.25e-6, c=100e-12, length=400.0, Zs=150, Zl=150)
    line_vl.add_voltage_source(position=400.0, conductor=0, magnitude=magnitude)
    vl_probe = line_vl.add_probe(position=400.0, type='voltage')
    line_vl.run_until(finalTime)

    assert np.all(vl_probe.val == vs_probe.val)
    
    # plt.figure()
    # plt.plot(vs_probe.t, vs_probe.val)
    # plt.plot(vl_probe.t, vl_probe.val)
    # plt.show()


def test_coaxial_line_paul_8_6_triangle():
    """ 
    Described in Ch. 8 of Paul Clayton, 
    Analysis of Multiconductor Transmission Lines. 2004. 
    """

    line = mtl.MTL(l=0.25e-6, c=100e-12, length=400.0, Zs=150.0)
    finalTime = 18e-6

    def magnitude(t): return wf.triangle_pulse(t, 100, 6e-6)
    line.add_voltage_source(position=0.0, conductor=0, magnitude=magnitude)
    v_probe = line.add_probe(position=0.0, type='voltage')

    line.run_until(finalTime)

    times = [4.0, 5.9, 6.1, 8.0, 10.1, 12]
    voltages = [16.67, 12.5, -12.5, -25, 6.25, 12.5]
    for (t, v) in zip(times, voltages):
        index = np.argmin(np.abs(v_probe.t - t*1e-6))
        assert np.all(np.isclose(v_probe.val[index], v, atol=0.5))

    # xticks = range(int(np.floor(min(1e6*v_probe.t))), int(np.ceil(max(1e6*v_probe.t))+1))

    # plt.plot(1e6*v_probe.t, v_probe.val)
    # plt.ylabel(r'$V (0, t)\,[V]$')
    # plt.xlabel(r'$t\,[\mu s]$')
    # plt.grid('both')
    # plt.show()


def test_ribbon_cable_20ns_paul_9_3():
    """
    Described in Ch. 9.3.1 "Ribbon Cables" of Paul Clayton
    Analysis of Multiconductor Transmission Lines. 2007. 
    """
    l = np.zeros([2, 2])
    l[0] = [0.7485*1e-6, 0.5077*1e-6]
    l[1] = [0.5077*1e-6, 1.0154*1e-6]
    c = np.zeros([2, 2])
    c[0] = [37.432*1e-12, -18.716*1e-12]
    c[1] = [-18.716*1e-12, 24.982*1e-12]

    Zs, Zl = np.zeros([1, 2]), np.zeros([1, 2])
    Zs[:] = [50, 50]
    Zl[:] = [50, 50]

    line = mtl.MTL(l=l, c=c, length=2.0, nx=2, Zs=Zs, Zl=Zl)
    finalTime = 200e-9

    def magnitude(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=20e-9, fall_time=20e-9, f0=1e6, D=0.5)
    line.add_voltage_source(position=0.0, conductor=1, magnitude=magnitude)
    v_probe = line.add_probe(position=0.0, type='voltage')

    line.run_until(finalTime)

    # From Paul's book:
    # "The crosstalk waveform rises to a peak of around 110 mV [...]"
    assert (np.isclose(np.max(v_probe.val[:, 0]), 113e-3, atol=1e-3))

    # plt.plot(1e9*v_probe.t, 1e3*v_probe.val)
    # plt.ylabel(r'$V_1 (0, t)\,[mV]$')
    # plt.xlabel(r'$t\,[ns]$')
    # plt.xticks(range(0, 200, 50))
    # plt.grid('both')
    # plt.show()


def test_ribbon_cable_1ns_paul_9_3():
    """
    Described in Ch. 9.3.1 "Ribbon Cables" of Paul Clayton
    Analysis of Multiconductor Transmission Lines. 2007. 
    """
    l = np.zeros([2, 2])
    l[0] = [0.7485*1e-6, 0.5077*1e-6]
    l[1] = [0.5077*1e-6, 1.0154*1e-6]
    c = np.zeros([2, 2])
    c[0] = [37.432*1e-12, -18.716*1e-12]
    c[1] = [-18.716*1e-12, 24.982*1e-12]

    Zs, Zl = np.zeros([1, 2]), np.zeros([1, 2])
    Zs[:] = [50, 50]
    Zl[:] = [50, 50]

    line = mtl.MTL(l=l, c=c, length=2.0, nx=100, Zs=Zs, Zl=Zl)
    finalTime = 200e-9

    def magnitude(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=1e-9, fall_time=1e-9, f0=1e6, D=0.5)
    line.add_voltage_source(position=0.0, conductor=1, magnitude=magnitude)
    v_probe = line.add_probe(position=0.0, type='voltage')

    line.run_until(finalTime)

    times = [12.5, 25, 40, 55]
    voltages = [120, 95, 55, 32]
    for (t, v) in zip(times, voltages):
        index = np.argmin(np.abs(v_probe.t - t*1e-9))
        assert np.all(np.isclose(v_probe.val[index, 0], v*1e-3, atol=10e-3))

    # plt.plot(1e9*v_probe.t, 1e3*v_probe.val)
    # plt.ylabel(r'$V_1 (0, t)\,[mV]$')
    # plt.xlabel(r'$t\,[ns]$')
    # plt.xticks(range(0, 200, 50))
    # plt.grid('both')
    # plt.show()


def test_lumped_dispersive():
    """
    Uses lossy mtl class with dispersive lumped connector
    """
    l = np.zeros([2, 2])
    l[0] = [0.7485*1e-6, 0.5077*1e-6]
    l[1] = [0.5077*1e-6, 1.0154*1e-6]
    c = np.zeros([2, 2])
    c[0] = [37.432*1e-12, -18.716*1e-12]
    c[1] = [-18.716*1e-12, 24.982*1e-12]

    Zs, Zl = np.zeros([1, 2]), np.zeros([1, 2])
    Zs[:] = [50, 50]
    Zl[:] = [50, 50]

    g = np.zeros([2, 2])
    r = np.zeros([2, 2])
    line = mtl.MTL_losses(l=l, c=c, g=g, r=r, length=2.0, nx=4, Zs=Zs, Zl=Zl)
    finalTime = 200e-9

    poles = np.array((-1e6,-1e9))
    residues = np.array((1e5, 1e7))
    line.add_dispersive_connector(position = 1.0, conductor=0,d=1,e=1,poles=poles, residues=residues)

    def magnitude(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=1e-9, fall_time=1e-9, f0=1e6, D=0.5)
    line.add_voltage_source(position=0.0, conductor=1, magnitude=magnitude)
    v_probe = line.add_probe(position=0.0, type='voltage')

    line.run_until(finalTime)

    plt.plot(1e9*v_probe.t, 1e3*v_probe.val)
    plt.ylabel(r'$V_1 (0, t)\,[mV]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xticks(range(0, 200, 50))
    plt.grid('both')
    plt.show()

    # times = [12.5, 25, 40, 55]
    # voltages = [120, 95, 55, 32]
    # for (t, v) in zip(times, voltages):
    #     index = np.argmin(np.abs(v_probe.t - t*1e-9))
    #     assert np.all(np.isclose(v_probe.val[index,0], v*1e-3, atol=10e-3))


def test_lumped_dispersive():
    """
    Uses lossy mtl class with dispersive lumped connector
    """
    l = np.zeros([2, 2])
    l[0] = [0.7485*1e-6, 0.5077*1e-6]
    l[1] = [0.5077*1e-6, 1.0154*1e-6]
    c = np.zeros([2, 2])
    c[0] = [37.432*1e-12, -18.716*1e-12]
    c[1] = [-18.716*1e-12, 24.982*1e-12]

    Zs, Zl = np.zeros([1, 2]), np.zeros([1, 2])
    Zs[:] = [50, 50]
    Zl[:] = [50, 50]

    g = np.zeros([2, 2])
    r = np.zeros([2, 2])
    line = mtl.MTL_losses(l=l, c=c, g=g, r=r, length=2.0, nx=4, Zs=Zs, Zl=Zl)
    finalTime = 200e-9

    poles = np.array((-1e6,-1e9))
    residues = np.array((1e5, 1e7))
    line.add_dispersive_connector(position = 1.0, conductor=0,d=1,e=1,poles=poles, residues=residues)

    def magnitude(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=1e-9, fall_time=1e-9, f0=1e6, D=0.5)
    line.add_voltage_source(position=0.0, conductor=1, magnitude=magnitude)
    v_probe = line.add_probe(position=0.0, type='voltage')

    line.run_until(finalTime)

    # plt.plot(1e9*v_probe.t, 1e3*v_probe.val[:,0])
    # plt.ylabel(r'$V_1 (0, t)\,[mV]$')
    # plt.xlabel(r'$t\,[ns]$')
    # plt.xticks(range(0, 200, 50))
    # plt.grid('both')
    # plt.show()

    fail()
    # times = [12.5, 25, 40, 55]
    # voltages = [120, 95, 55, 32]
    # for (t, v) in zip(times, voltages):
    #     index = np.argmin(np.abs(v_probe.t - t*1e-9))
    #     assert np.all(np.isclose(v_probe.val[index.0], v*1e-3, atol=10e-3))


def test_ribbon_cable_1ns_paul_9_3_lossless_lossy():
    """
    Described in Ch. 9.3.1 "Ribbon Cables" of Paul Clayton
    Analysis of Multiconductor Transmission Lines. 2007. 
    """
    """
    Uses lossy mtl class with no losses
    """
    l = np.zeros([2, 2])
    l[0] = [0.7485*1e-6, 0.5077*1e-6]
    l[1] = [0.5077*1e-6, 1.0154*1e-6]
    c = np.zeros([2, 2])
    c[0] = [37.432*1e-12, -18.716*1e-12]
    c[1] = [-18.716*1e-12, 24.982*1e-12]

    Zs, Zl = np.zeros([1, 2]), np.zeros([1, 2])
    Zs[:] = [50, 50]
    Zl[:] = [50, 50]

    g = np.zeros([2, 2])
    r = np.zeros([2, 2])
    line = mtl.MTL_losses(l=l, c=c, g=g, r=r, length=2.0, nx=100, Zs=Zs, Zl=Zl)
    finalTime = 200e-9

    def magnitude(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=1e-9, fall_time=1e-9, f0=1e6, D=0.5)
    line.add_voltage_source(position=0.0, conductor=1, magnitude=magnitude)
    v_probe = line.add_probe(position=0.0, type='voltage')

    line.run_until(finalTime)

    # plt.plot(1e9*v_probe.t, 1e3*v_probe.val)
    # plt.ylabel(r'$V_1 (0, t)\,[mV]$')
    # plt.xlabel(r'$t\,[ns]$')
    # plt.xticks(range(0, 200, 50))
    # plt.grid('both')
    # plt.show()

    times = [12.5, 25, 40, 55]
    voltages = [120, 95, 55, 32]
    for (t, v) in zip(times, voltages):
        index = np.argmin(np.abs(v_probe.t - t*1e-9))
        assert np.all(np.isclose(v_probe.val[index,0], v*1e-3, atol=10e-3))


def test_pcb_paul_9_3_2():
    """
    Described in Ch. 9.3.2 "Printed Circuit Boards" of Paul Clayton
    Analysis of Multiconductor Transmission Lines. 2007. 
    """
    l = np.zeros([2, 2])
    l[0] = [1.10515*1e-6, 0.690613*1e-6]
    l[1] = [0.690613*1e-6, 1.38123*1e-6]
    c = np.zeros([2, 2])
    c[0] = [40.5985*1e-12, -20.2992*1e-12]
    c[1] = [-20.2992*1e-12, 29.7378*1e-12]

    Zs, Zl = np.zeros([1, 2]), np.zeros([1, 2])
    Zs[:] = [50, 50]
    Zl[:] = [50, 50]

    line = mtl.MTL(l=l, c=c, length=0.254, nx=2, Zs=Zs, Zl=Zl)
    finalTime = 40e-9

    def magnitude(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=6.25e-9, fall_time=6.25e-9, f0=1e6, D=0.5)
    line.add_voltage_source(position=0.0, conductor=1, magnitude=magnitude)
    v_probe = line.add_probe(position=0.0, type='voltage')

    line.run_until(finalTime)

    times = [5, 10, 15, 20]
    voltages = [80, 62.5, 23, 8]
    for (t, v) in zip(times, voltages):
        index = np.argmin(np.abs(v_probe.t - t*1e-9))
        assert np.all(np.isclose(v_probe.val[index, 0], v*1e-3, atol=10e-3))

    # plt.plot(1e9*v_probe.t, 1e3*v_probe.val)
    # plt.ylabel(r'$V_1 (0, t)\,[mV]$')
    # plt.xlabel(r'$t\,[\mu s]$')
    # plt.xticks(range(0, 40, 5))
    # plt.grid('both')
    # plt.show()


def test_dispersive_R():
    
    length = 500e-3
    Zs, Zl = 50.0, 50.0

    line = mtl.MTL_losses(l=0.25e-6, c=100e-12, g=0.0, r=0.0, length=length, nx = 20,Zs=Zs, Zl=Zl)

    R = 100
    L = 0

    # poles = np.array([])    
    # residues = np.array([])    
    # D = R/line.dx # R pul
    # E = L/line.dx # L pul

    # line.add_dispersive_connector(position = 250e-3, 
    #                             conductor=0,
    #                             d=D,
    #                             e=E,
    #                             poles=poles, 
    #                             residues=residues)
    
    line.add_resistance_at_point(position = 250e-3, conductor = 0, resistance = R)
    finalTime = 3000.0e-9

    def magnitude(t): return wf.trapezoidal_wave(
        t, A=1, rise_time=1e-9, fall_time=1e-9, f0=1e5, D=0.5)
    line.add_voltage_source(position=0.0, conductor=0, magnitude=magnitude)

    port_probe_0 = line.add_port_probe(0)
    port_probe_1 = line.add_port_probe(1)

    v_probe_L = line.add_probe(220e-3,"voltage")
    v_probe_R = line.add_probe(280e-3,"voltage")

    line.run_until(finalTime)

    plt.plot(1e9*port_probe_1.i0.t, (v_probe_L.val[:,0]-v_probe_R.val[:,0])/port_probe_1.i0.val[:,0], label ='delta V /I')
    plt.ylabel(r'$Z (t)\,[\Omega]$')
    plt.xlabel(r'$t\,[ns]$')
    plt.xlim(10, finalTime*1e9)
    plt.grid('both')
    plt.legend()
    plt.show()

def test_dispersive_connector_R_freq():
    
    length = 500e-3
    Zs, Zl = 50.0, 50.0

    line = mtl.MTL_losses(l=0.25e-6, c=100e-12, g=0.0, r=0.0, length=length, nx = 20,Zs=Zs, Zl=Zl)

    R = 100
    L = 0
    
    # poles = np.array([])    
    # residues = np.array([])    
    # D = R/line.dx # R pul
    # E = L/line.dx # L pul
    # line.add_dispersive_connector(position = 250e-3, 
    #                             conductor=0,
    #                             d=D,
    #                             e=E,
    #                             poles=poles, 
    #                             residues=residues)
    
    # line.add_resistance_at_point(position = 250e-3, conductor = 0, resistance = R)
    finalTime = 3000.0e-9


    line_ntw = line.extract_2p_network(fMin=1e5, fMax=1e9, finalTime=finalTime)
    plt.figure()
    line_ntw.plot_z_re(0,0,label='real 00')
    line_ntw.plot_z_re(0,1,label='real 01')

    plt.xscale('log')
    plt.grid()
    plt.legend()
    plt.xlim(1e5, 1e9)
    plt.show()

    
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
    line = mtl.MTL_losses(l=L, c=C, g=G, r=R, length=length, Zs=Zs, Zl=Zl)
    
    poles = np.array([])    
    residues = np.array([])    
    D = 0
    E = 0
    line.add_dispersive_connector(position = 200e-3, 
                                conductor=0,
                                d=D,
                                e=E,
                                poles=poles, 
                                residues=residues)

    finalTime = 300e-9
    line_ntw = line.extract_2p_network(fMin=1e7, fMax=1e9, finalTime=finalTime)
    
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


def test_wire_over_ground_incident_E_paul_11_3_6_50ns():
    """
    Described in Ch. 11.3.2 "Computed results" of Paul Clayton
    Analysis of Multiconductor Transmission Lines. 2007. 
    Computes the induced voltage at the left end of the line
    when excited by an incident external field with rise time 50 ns
    """

    wire_radius = 0.254e-3
    wire_h = 0.02
    wire_separation = 2.*wire_h
    l = (mu_0/(2*np.pi))*np.arccosh(wire_separation/wire_radius)
    c = 2*np.pi*epsilon_0/np.arccosh(wire_separation/wire_radius)

    nx, finalTime, rise_time, fall_time = 10, 100e-9, 50e-9, 50e-9

    line = mtl.MTL(l=l, c=c, length=1.0, nx=nx, Zs=500, Zl=1000)

    x, z, t = sp.symbols('x z t')
    magnitude = wf.trapezoidal_wave_sp(
        A=1, rise_time=rise_time, fall_time=fall_time, f0=1e6, D=0.5)

    def e_x(x, z, t): return (x+z+t)*0
    def e_z(x, z, t): return magnitude
    line.add_external_field(e_x, e_z, ref_distance=0.0,
                            distances=np.array([wire_separation]))

    v_probe = line.add_probe(position=0.0, type='voltage')

    line.run_until(finalTime)

    times = [3.5, 7, 1.0, 25, 53, 56.6, 59.8, 80]
    voltages = [-1.61, -0.78, -0.99, -0.87, 0.75, -0.1315, 0.1, -0.015]
    for (t, v) in zip(times, voltages):
        index = np.argmin(np.abs(v_probe.t - t*1e-9))
        assert np.all(np.isclose(v_probe.val[index, 0], v*1e-3, atol=2.5e-3))

    # plt.plot(1e9*probe.v0.t, 1e3*probe.v0.val, label='port')
    # plt.plot(1e9*v_probe.t, 1e3*v_probe.val, label='v probe')
    # plt.ylabel(r'$V_1 (0, t)\,[mV]$')
    # plt.xlabel(r'$t\,[ns]$')
    # plt.xticks(range(0, int(finalTime*1e9), 5))
    # plt.grid('both')
    # plt.legend()
    # plt.show()


def test_wire_over_ground_incident_E_paul_11_3_6_10ns():
    """
    Described in Ch. 11.3.2 "Computed results" of Paul Clayton
    Analysis of Multiconductor Transmission Lines. 2007. 
    Computes the induced voltage at the left end of the line
    when excited by an incident external field with rise time 10 ns
    """

    wire_radius = 0.254e-3
    wire_h = 0.02
    wire_separation = 2.*wire_h
    l = (mu_0/(2*np.pi))*np.arccosh(wire_separation/wire_radius)
    c = 2*np.pi*epsilon_0/np.arccosh(wire_separation/wire_radius)

    nx, finalTime, rise_time, fall_time = 10, 40e-9, 10e-9, 10e-9

    line = mtl.MTL(l=l, c=c, length=1.0, nx=nx, Zs=500, Zl=1000)

    x, z, t = sp.symbols('x z t')
    magnitude = wf.trapezoidal_wave_sp(
        A=1, rise_time=rise_time, fall_time=fall_time, f0=1e6, D=0.5)

    def e_x(x, z, t): return (x+z+t)*0
    def e_z(x, z, t): return magnitude
    line.add_external_field(e_x, e_z, ref_distance=0.0,
                            distances=np.array([wire_separation]))

    v_probe = line.add_probe(position=0.0, type='voltage')

    line.run_until(finalTime)

    times = [3.4, 6.8, 9.9, 16.7, 20, 23.3, 35]
    voltages = [-8.2, -3.8, -4.8, -0.55, 0.52, -0.019, 6e-3]
    for (t, v) in zip(times, voltages):
        index = np.argmin(np.abs(v_probe.t - t*1e-9))
        assert np.all(np.isclose(v_probe.val[index, 0], v*1e-3, atol=1.5e-3))

    # plt.plot(1e9*probe.v0.t, 1e3*probe.v0.val, label='port')
    # plt.plot(1e9*v_probe.t, 1e3*v_probe.val, label='v probe')
    # plt.ylabel(r'$V_1 (0, t)\,[mV]$')
    # plt.xlabel(r'$t\,[ns]$')
    # plt.xticks(range(0, int(finalTime*1e9), 5))
    # plt.grid('both')
    # plt.legend()
    # plt.show()


def test_wire_over_ground_incident_E_paul_11_3_6_1ns():
    """
    Described in Ch. 11.3.2 "Computed results" of Paul Clayton
    Analysis of Multiconductor Transmission Lines. 2007. 
    Computes the induced voltage at the left end of the line
    when excited by an incident external field with rise time 1 ns
    """

    wire_radius = 0.254e-3
    wire_h = 0.02
    wire_separation = 2.*wire_h
    l = (mu_0/(2*np.pi))*np.arccosh(wire_separation/wire_radius)
    c = 2*np.pi*epsilon_0/np.arccosh(wire_separation/wire_radius)

    nx, finalTime, rise_time, fall_time = 50, 30e-9, 1e-9, 1e-9

    line = mtl.MTL(l=l, c=c, length=1.0, nx=nx, Zs=500, Zl=1000)

    x, z, t = sp.symbols('x z t')
    magnitude = wf.trapezoidal_wave_sp(
        A=1, rise_time=rise_time, fall_time=fall_time, f0=1e6, D=0.5)

    def e_x(x, z, t): return (x+z+t)*0
    def e_z(x, z, t): return magnitude
    line.add_external_field(e_x, e_z, ref_distance=0.0,
                            distances=np.array([wire_separation]))

    v_probe = line.add_probe(position=0.0, type='voltage')

    line.run_until(finalTime)

    times = [3, 5, 8.5, 12, 15, 19, 25]
    voltages = [-24, 12.9, -3.2, 1.5, -0.6, 0.08, -0.38]
    for (t, v) in zip(times, voltages):
        index = np.argmin(np.abs(v_probe.t - t*1e-9))
        assert np.all(np.isclose(v_probe.val[index, 0], v*1e-3, atol=2.5e-3))

    # plt.plot(1e9*probe.v0.t, 1e3*probe.v0.val, label='port')
    # plt.plot(1e9*v_probe.t, 1e3*v_probe.val, label='v probe')
    # plt.ylabel(r'$V_1 (0, t)\,[mV]$')
    # plt.xlabel(r'$t\,[ns]$')
    # plt.xticks(range(0, int(finalTime*1e9), 5))
    # plt.grid('both')
    # plt.legend()
    # plt.show()
