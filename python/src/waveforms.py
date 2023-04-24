import numpy as np
import sympy as sp

def gaussian(x, x0, s0):
    return np.exp( - (x-x0)**2 / (2*s0**2) )

def square_pulse(x, A, x0):
    return A*(x <= x0)*(x >= 0)

def triangle_pulse(x, A, x0):
    return A*(x/x0)*(x <= x0)*(x >= 0)

def trapezoidal_wave_sp(A, rise_time, fall_time, f0, D, vel = 0):
    x, z, t = sp.symbols('x z t')
    A_sp, rise_time_sp, fall_time_sp, f0_sp, D_sp, v = sp.symbols('A_sp rise_time_sp fall_time_sp f0_sp D_sp v')

    
    mod_time = t - sp.floor(f0_sp*t + D_sp) - x/v
    # mod_time = (t+x/v) - sp.floor(f0_sp*(t+x/v))
    # mod_time = sp.Mod(t, 1/f0_sp) + x/v

    cte_time = D_sp/f0_sp - 0.5 * (rise_time_sp + fall_time_sp)
    t1, t2, t3 = rise_time_sp, rise_time_sp + cte_time, rise_time_sp + cte_time + fall_time_sp

    f1 = sp.Piecewise((A_sp*(mod_time/rise_time_sp), mod_time <= t1), 
                        (A_sp, (mod_time > t1) & (mod_time <= t2)), 
                        (-A_sp*mod_time/fall_time_sp + A_sp*(rise_time_sp + cte_time + fall_time_sp)/fall_time_sp, (mod_time > t2) & (mod_time <= t3)))

    # return f1.subs(A_sp, A).subs(rise_time_sp, rise_time).subs(fall_time_sp, fall_time).subs(f0_sp, f0).subs(D_sp, D)
    return f1.subs(A_sp, A).subs(rise_time_sp, rise_time).subs(fall_time_sp, fall_time).subs(f0_sp, f0).subs(D_sp, D).subs(v, vel)
    
def trapezoidal_wave(x, A, rise_time, fall_time, f0, D):
    mod_time = x % (1/f0)
    cte_time = D/f0 - 0.5 * (rise_time + fall_time)
    
    t1, t2, t3 = rise_time, rise_time + cte_time, rise_time + cte_time + fall_time
    
    return A*(mod_time/rise_time)*(mod_time <= t1) + \
           A*(mod_time >= t1 )*(mod_time <= t2) + \
           (-A*mod_time/fall_time + A*(rise_time + cte_time + fall_time)/fall_time)*(mod_time >= t2)*(mod_time <= t3)

def ramp_pulse(x, A, x0):
    return A*(x/x0)*(x <= x0)*(x >= 0) + A*(x >= x0)
