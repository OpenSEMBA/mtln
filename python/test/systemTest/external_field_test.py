import numpy as np
import matplotlib.pyplot as plt

import src.mtl as mtl

import src.waveforms as wf

from scipy.constants import epsilon_0, mu_0, speed_of_light
import sympy as sp

import pytest

@pytest.mark.skip(reason="planewave is not used. outdated implementation")
def test_external_field():
    """
    Described in Ch. 11.3.2 "Computed results" of Paul Clayton
    Analysis of Multiconductor Transmission Lines. 2007. 
    Computes the induced voltage at the left end of the line
    when excited by an incident external field with rise time 50 ns
    """

    def magnitude(t): return wf.trapezoidal_wave(t, A=1, rise_time=50e-9, fall_time=50e-9, f0=1e6, D=0.5)

    pw = mtl.PlaneWave(field = magnitude, polar = 0, azimuth = 0, polarization= 0)
    assert(pw.ex == 0)
    assert(pw.ez == 1)

    wire_radius = 0.254e-3
    wire_h = 0.02
    wire_separation = 2.*wire_h
    l = (mu_0/(2*np.pi))*np.arccosh(wire_separation/wire_radius)
    c = 2*np.pi*epsilon_0/np.arccosh(wire_separation/wire_radius)

    line = mtl.MTL(l=l, c=c, length=1.0, ndiv=10, Zs=500, Zl=1000)
    line.add_planewave(pw, np.array([0.5]), np.array([0.0]))

    #compute new field terms
    #write equations
    #compute terminals


    nx, finalTime, rise_time, fall_time = 10, 100e-9, 50e-9, 50e-9

    line = mtl.MTL(l=l, c=c, length=1.0, ndiv=nx, Zs=500, Zl=1000)

    x, z, t = sp.symbols('x z t')
    magnitude = wf.trapezoidal_wave_sp(
        A=1, rise_time=rise_time, fall_time=fall_time, f0=1e6, D=0.5)

    def e_x(x, z, t): return (x+z+t)*0
    def e_z(x, z, t): return magnitude
    line.add_external_field(e_x, e_z, ref_distance=0.0,
                            distances=np.array([wire_separation]))

    v_probe = line.add_probe(position=0.0, probe_type='voltage')

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


@pytest.mark.skip(reason="trapezoidal wave implementation has changed. Probably, outdated test")
def test_wire_over_ground_incident_E_paul_11_3_6_50ns_planewave():
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

    line = mtl.MTL(l=l, c=c, length=1.0, ndiv=nx, Zs=500, Zl=1000)

    x, z, t = sp.symbols('x z t')
    magnitude = wf.trapezoidal_wave_sp(A=1, rise_time=rise_time, fall_time=fall_time, f0=1e6, D=0.5)

    def e_x(x, z, t): return (x+z+t)*0
    def e_z(x, z, t): return magnitude
    line.add_external_field(e_x, e_z, ref_distance=0.0,
                            distances=np.array([wire_separation]))

    v_probe = line.add_probe(position=0.0, probe_type='voltage')

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

def test_wire_over_ground_incident_E_paul_11_3_6_50ns_external_field():
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

    line = mtl.MTL(l=l, c=c, length=1.0, ndiv=nx, Zs=500, Zl=1000)


    A, v, x0 = 1.0, np.max(line.get_phase_velocities()), rise_time
    ex = sp.Function('ex')
    ex = wf.null()
    ey = sp.Function('ey')
    ey = wf.null()
    ez = sp.Function('ez')
    ez = wf.trapezoidal_wave_sp(A=1, rise_time=rise_time, fall_time=fall_time, f0=1e6, D=0.5, v = v)
    
    field = mtl.Field(ex,ey,ez)

    distances = np.zeros([1, line.u.shape[0], 3])
    distances[:,:,0] = wire_separation

    line.add_external_field(field, distances)

    v_probe = line.add_probe(position=0.0, probe_type='voltage')

    line.run_until(finalTime)

    # plt.plot(1e9*v_probe.t, 1e3*v_probe.val, label='v probe')
    # plt.ylabel(r'$V_1 (0, t)\,[mV]$')
    # plt.xlabel(r'$t\,[ns]$')
    # plt.xticks(range(0, int(finalTime*1e9), 5))
    # plt.grid('both')
    # plt.legend()
    # plt.show()

    times = [3.5, 7, 25, 53, 56.6, 59.8, 80]
    voltages = [-1.4, -0.7, -0.8 ,0.45, -0.1315, 0.1, -0.015]
    for (t, v) in zip(times, voltages):
        index = np.argmin(np.abs(v_probe.t - t*1e-9))
        assert np.all(np.isclose(v_probe.val[index, 0]*1e3, v, atol=0.25))



def test_wire_over_ground_incident_E_paul_11_3_6_10ns_external_field():
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

    line = mtl.MTL(l=l, c=c, length=1.0, ndiv=nx, Zs=500, Zl=1000)

    A, v, x0 = 1.0, np.max(line.get_phase_velocities()), rise_time
    ex = sp.Function('ex')
    ex = wf.null()
    ey = sp.Function('ey')
    ey = wf.null()
    ez = sp.Function('ez')
    ez = wf.trapezoidal_wave_sp(A=1, rise_time=rise_time, fall_time=fall_time, f0=1e6, D=0.5, v=v)
    
    field = mtl.Field(ex,ey,ez)

    distances = np.zeros([1, line.u.shape[0], 3])
    distances[:,:,0] = wire_separation

    line.add_external_field(field, distances)

    v_probe = line.add_probe(position=0.0, probe_type='voltage')

    line.run_until(finalTime)

    # plt.plot(1e9*v_probe.t, 1e3*v_probe.val, label='v probe')
    # plt.ylabel(r'$V_1 (0, t)\,[mV]$')
    # plt.xlabel(r'$t\,[ns]$')
    # plt.xticks(range(0, int(finalTime*1e9), 5))
    # plt.grid('both')
    # plt.legend()
    # plt.show()

    times = [3.4, 6.8, 9.9, 16.7, 20, 23.3, 35]
    voltages = [-8.2, -3.8, -4.8, -0.55, 0.52, -0.019, 6e-3]
    for (t, v) in zip(times, voltages):
        index = np.argmin(np.abs(v_probe.t - t*1e-9))
        assert np.all(np.isclose(v_probe.val[index, 0], v*1e-3, atol=1.5e-3))




def test_wire_over_ground_incident_E_paul_11_3_6_1ns_external_field():
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

    line = mtl.MTL(l=l, c=c, length=1.0, ndiv=nx, Zs=500, Zl=1000)

    A, v, x0 = 1.0, np.max(line.get_phase_velocities()), rise_time
    ex = sp.Function('ex')
    ex = wf.null()
    ey = sp.Function('ey')
    ey = wf.null()
    ez = sp.Function('ez')
    ez = wf.trapezoidal_wave_x_sp(A=1, rise_time=rise_time, fall_time=fall_time, f0=1e6, D=0.5, v=v)
    
    field = mtl.Field(ex,ey,ez)
    distances = np.zeros([1, line.u.shape[0], 3])
    distances[:,:,0] = wire_separation

    line.add_external_field(field, distances)

    v_probe = line.add_probe(position=0.0, probe_type='voltage')

    line.run_until(finalTime)

    # plt.plot(1e9*v_probe.t, 1e3*v_probe.val, label='v probe')
    # plt.ylabel(r'$V_1 (0, t)\,[mV]$')
    # plt.xlabel(r'$t\,[ns]$')
    # plt.xticks(range(0, int(finalTime*1e9), 5))
    # plt.grid('both')
    # plt.legend()
    # plt.show()

    times = [3, 5, 8.5, 12, 15, 19, 25]
    voltages = [-24, 12.9, -3.2, 1.5, -0.6, 0.08, -0.38]
    for (t, v) in zip(times, voltages):
        index = np.argmin(np.abs(v_probe.t - t*1e-9))
        assert np.all(np.isclose(v_probe.val[index, 0], v*1e-3, atol=2.5e-3))


@pytest.mark.skip(reason="Generic external field not implemented")
def test_wire_over_ground_incident_E_transversal_paul_12_4_100ns():

    """
    Described in Ch. 12.4 "Computed results" of Paul Clayton
    Analysis of Multiconductor Transmission Lines. 2007. 
    """
    l = np.array([
        [0.7485e-6, 0.2408e-6],
        [0.2408e-6, 0.7485e-6]])

    c = np.array([
        [24.982e-12, -6.266e-12],
        [-6.266e-12, 24.982e-12]])


    nx, finalTime = 5, 200e-9
    rise_time, fall_time = 100e-9,100e-9
    line = mtl.MTL(l=l, c=c, length=2.0, ndiv=nx, Zs=[500,500], Zl=[500,500])

    wire_separation = 0.00127
    e_x = wf.ramp_pulse_x_sp(A=1, x0=rise_time, v=3.0e8)
    e_y = wf.null()
    e_z = wf.null()
    
    field = mtl.Field(e_x,e_y,e_z)
    distances = np.zeros([2, line.u.shape[0], 3])
    distances[0,:,0] = -wire_separation
    distances[1,:,0] = wire_separation
    
    line.add_external_field(field, distances)

    v_probe = line.add_probe(position=np.array([0.0, 0.0, 0.0]), probe_type='voltage')

    line.run_until(finalTime)

    # times = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    # voltages = []
    # for (t, v) in zip(times, voltages):
    #     index = np.argmin(np.abs(v_probe.t - t*1e-9))
    #     assert np.all(np.isclose(v_probe.val[index, 0], v*1e-6, atol=2.5e-3))

    # plt.plot(1e9*probe.v0.t, 1e3*probe.v0.val, label='port')
    plt.plot(1e9*v_probe.t, 1e6*v_probe.val[:,0], label='V1')
    plt.ylabel(r'$V_1 (0, t)\,[\mu V]$')
    plt.xlabel(r'$t\,[ns]$')
    # plt.xticks(range(0, int(finalTime*1e9), 5))
    plt.grid('both')
    plt.legend()
    plt.show()
    

@pytest.mark.skip(reason="Generic external field not implemented")
def test_wire_over_ground_incident_E_transversal_paul_12_4_10ns():

    """
    Described in Ch. 12.4 "Computed results" of Paul Clayton
    Analysis of Multiconductor Transmission Lines. 2007. 
    """
    l = np.array([
        [0.7485e-6, 0.2408e-6],
        [0.2408e-6, 0.7485e-6]])

    c = np.array([
        [24.982e-12, -6.266e-12],
        [-6.266e-12, 24.982e-12]])

    nx, finalTime = 10, 100e-9
    rise_time, fall_time = 10e-9, 10e-9

    line = mtl.MTL(l=l, c=c, length=2.0, ndiv=nx, Zs=[500,500], Zl=[500,500])

    wire_separation = 0.00127

    e_y = wf.null()
    e_z = wf.null()
    e_x = wf.ramp_pulse_x_sp(A=1, x0=rise_time)
    
    field = mtl.Field(e_x,e_y,e_z)
    distances = np.zeros([2, line.u.shape[0], 3])
    distances[0,:,0] = -wire_separation
    distances[1,:,0] = wire_separation
    
    line.add_external_field(field, distances)
    v_probe = line.add_probe(position=0.0, probe_type='voltage')
    line.run_until(finalTime)

    # times = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    # voltages = []
    # for (t, v) in zip(times, voltages):
    #     index = np.argmin(np.abs(v_probe.t - t*1e-9))
    #     assert np.all(np.isclose(v_probe.val[index, 0], v*1e-6, atol=2.5e-3))

    # plt.plot(1e9*probe.v0.t, 1e3*probe.v0.val, label='port')
    plt.plot(1e9*v_probe.t, 1e3*v_probe.val[:,1], label='V1')
    plt.ylabel(r'$V_1 (0, t)\,[mV]$')
    plt.xlabel(r'$t\,[ns]$')
    # plt.xticks(range(0, int(finalTime*1e9), 5))
    plt.grid('both')
    plt.legend()
    plt.show()
    
@pytest.mark.skip(reason="Generic external field not implemented")
def test_wire_over_ground_incident_E_transversal_paul_12_4_1ns():

    """
    Described in Ch. 12.4 "Computed results" of Paul Clayton
    Analysis of Multiconductor Transmission Lines. 2007. 
    """
    l = np.array([
        [0.7485e-6, 0.2408e-6],
        [0.2408e-6, 0.7485e-6]])

    c = np.array([
        [24.982e-12, -6.266e-12],
        [-6.266e-12, 24.982e-12]])

    nx, finalTime = 100, 100e-9
    rise_time, fall_time = 1e-9, 1e-9

    line = mtl.MTL(l=l, c=c, length=2.0, ndiv=nx, Zs=[500,500], Zl=[500,500])

    wire_separation = 0.00127

    e_y = wf.null()
    e_z = wf.null()
    e_x = wf.ramp_pulse_x_sp(A=1, x0=rise_time)
    
    field = mtl.Field(e_x,e_y,e_z)
    distances = np.zeros([2, line.u.shape[0], 3])
    distances[0,:,0] = -wire_separation
    distances[1,:,0] = wire_separation
    
    line.add_external_field(field, distances)

    v_probe = line.add_probe(position=0.0, probe_type='voltage')

    line.run_until(finalTime)

    # times = [10, 25, 40, 55]
    # voltages = [1]
    # for (t, v) in zip(times, voltages):
    #     index = np.argmin(np.abs(v_probe.t - t*1e-9))
    #     assert np.all(np.isclose(v_probe.val[index, 0], v*1e-6, atol=2.5e-3))

    # plt.plot(1e9*probe.v0.t, 1e3*probe.v0.val, label='port')
    plt.plot(1e9*v_probe.t, 1e6*v_probe.val, label='v probe')
    plt.ylabel(r'$V_1 (0, t)\,[\mu V]$')
    plt.xlabel(r'$t\,[ns]$')
    # plt.xticks(range(0, int(finalTime*1e9), 5))
    plt.grid('both')
    plt.legend()
    plt.show()
