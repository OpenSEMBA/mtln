import numpy as np
import sympy as sp

def gaussian(x, x0, s0):
    return np.exp( - (x-x0)**2 / (2*s0**2) )

def gaussian_2(x, A, x0, s0):
    return A*np.exp( - ((x-x0)/s0)**2 )

def square_pulse(x, A, x0):
    return A*(x <= x0)*(x >= 0)

def triangle_pulse(x, A, x0):
    return A*(x/x0)*(x <= x0)*(x >= 0)

def sin_sq_pulse(x, A, w):
    return (A*(np.sin(w*x))**2)*(x >= 0)*(x < np.pi/w)

def trapezoidal_wave_x_sp(A, rise_time, fall_time, f0, D, v):
    x, t = sp.symbols('x t')

    mod_time = sp.Mod(t + x/v, 1/f0) 

    cte_time = D/f0 - 0.5 * (rise_time + fall_time)
    t1, t2, t3 = rise_time, rise_time + cte_time, rise_time + cte_time + fall_time

    f1 = sp.Piecewise((A*(mod_time/rise_time), mod_time <= t1), 
                        (A, (mod_time > t1) & (mod_time <= t2)), 
                        (-A*mod_time/fall_time + A*(rise_time + cte_time + fall_time)/fall_time, (mod_time > t2) & (mod_time <= t3)),
                        (0, mod_time >= t3))

    return f1

def trapezoidal_wave_sp(A, rise_time, fall_time, f0, D, v):
    x, t = sp.symbols('x t')

    mod_time = sp.Mod(t + x/v, 1/f0) 

    cte_time = D/f0 - 0.5 * (rise_time + fall_time)
    t1, t2, t3 = rise_time, rise_time + cte_time, rise_time + cte_time + fall_time

    f1 = sp.Piecewise((A*(mod_time/rise_time), mod_time <= t1), 
                        (A, (mod_time > t1) & (mod_time <= t2)), 
                        (-A*mod_time/fall_time + A*(rise_time + cte_time + fall_time)/fall_time, (mod_time > t2) & (mod_time <= t3)),
                        (0, mod_time >= t3))

    return f1
    
def trapezoidal_wave(x, A, rise_time, fall_time, f0, D):
    mod_time = x % (1/f0)
    cte_time = D/f0 - 0.5 * (rise_time + fall_time)
    
    t1, t2, t3 = rise_time, rise_time + cte_time, rise_time + cte_time + fall_time
    
    return A*(mod_time/rise_time)*(mod_time <= t1) + \
           A*(mod_time >= t1 )*(mod_time <= t2) + \
           (-A*mod_time/fall_time + A*(rise_time + cte_time + fall_time)/fall_time)*(mod_time >= t2)*(mod_time <= t3)

def ramp_pulse(x, A, x0):
    return A*(x/x0)*(x <= x0)*(x >= 0) + A*(x >= x0)

def ramp_pulse_x_sp(A, x0):
    x,y,z,t,v = sp.symbols('x y z t v')
    f1 = sp.Piecewise((A*(t/x0), (t+x/v <= x0)), (A, (t+x/v >= x0)))
    return f1

def ramp_pulse_sp(A, x0):
    t = sp.symbols('t')
    f1 = sp.Piecewise((A*(t/x0), (t <= x0)), (A, (t >= x0)))
    return f1

def ramp_pulse_y_sp(A, x0, v):
    x,y,z,t = sp.symbols('x y z t')
    A_sp, x0_sp, v_sp = sp.symbols('A_sp x0_sp v')
    f1 = sp.Piecewise((A*(t/x0), (t+y/v <= x0)), (A, (t+y/v >= x0)))
    return f1

def ramp_pulse_z_sp(A, x0, v):
    x,y,z,t = sp.symbols('x y z t')
    f1 = sp.Piecewise((A*(t/x0), (t+z/v <= x0)), (A, (t+z/v >= x0)))
    return f1


def double_exp(t, C, a, b):
    return C*(sp.exp(-a*t)-sp.exp(-b*t))

def double_exp_sp(C, a, b):
    x, z, t = sp.symbols('x z t')
    return C*(sp.exp(-a*t)-sp.exp(-b*t))

def double_exp_xy_sp(C, a, b):
    x, y, z, t, v = sp.symbols('x y z t v')
    return C*(sp.exp(-a*(t + x/v + y/v))-sp.exp(-b*(t + x/v + y/v)))

def double_exp_yz_sp(C, a, b):
    x, y, z, t, v = sp.symbols('x y z t v')
    return C*(sp.exp(-a*(t + z/v + y/v))-sp.exp(-b*(t + z/v + y/v)))

def double_exp_xz_sp(C, a, b):
    x, y, z, t, v = sp.symbols('x y z t v')
    return C*(sp.exp(-a*(t + x/v + z/v))-sp.exp(-b*(t + x/v + z/v)))
    

# def double_exp_x_sp(C, a, b):
#     x, z, t, v = sp.symbols('x z t v')
#     C_sp, a_sp, b_sp, v = sp.symbols('C_sp a_sp b_sp v')
#     f = C_sp*(sp.exp(-a_sp*(t+x/v))-sp.exp(-b_sp*(t+x/v)))
#     # f = C_sp*(sp.exp(-a_sp*(t+x/v))-sp.exp(-b_sp*(t+x/v)))
    
#     return f.subs(C_sp, C).subs(a_sp, a).subs(b_sp,b)

def null():
    n = sp.Symbol('n')
    return n*0.0