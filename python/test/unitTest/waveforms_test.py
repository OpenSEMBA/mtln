import numpy as np

import sympy as sp
from scipy.constants import epsilon_0, mu_0

import src.waveforms as wf

def test_trapezoidal_pulse_sp():

    wire_radius = 2.54e-3
    wire_height = 2e-2
    l = (mu_0/(2*np.pi))*np.arccosh(wire_height/wire_radius)
    c = 2*np.pi*epsilon_0/np.arccosh(wire_height/wire_radius)
    x, z, t = sp.symbols('x z t')
    A, rise_time, fall_time= 1, 10e-9, 10e-9
    D, f0 = 0.5, 20e6
    v = 1.0/np.sqrt(l*c)
    magnitude = wf.trapezoidal_wave_sp(A, rise_time, fall_time, f0, D, v=v)
    def p(x,z,t): return magnitude

    m = sp.Function('m')
    m = p
    pulse = sp.lambdify(t, m(x,z,t).subs(x,0.00).subs(z, 0))

    plateu_duration = D/f0 - 0.5 * (rise_time + fall_time)
       
    assert (float(pulse(0.5*rise_time)) == 0.5*A)
    assert (float(pulse(1.1*rise_time)) == A)
    assert (float(pulse(rise_time + plateu_duration + 0.5*fall_time)) == 0.5*A)

def test_trapezoidal_pulse():
    A = 1
    rise_time = 10e-9
    fall_time = 10e-9
    f0 = 20e6
    D = 0.5
    def magnitude(t): return wf.trapezoidal_wave(
        t, A, rise_time, fall_time, f0, D)
    plateu_duration = D/f0 - 0.5 * (rise_time + fall_time)

    assert (magnitude(0.5*rise_time) == 0.5*A)
    assert (magnitude(1.1*rise_time) == A)
    assert (magnitude(rise_time + plateu_duration + 0.5*fall_time) == 0.5*A)


