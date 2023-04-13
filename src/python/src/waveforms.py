import numpy as np

def gaussian(x, x0, s0):
    return np.exp( - (x-x0)**2 / (2*s0**2) )

def square_pulse(x, A, x0):
    return A*(x <= x0)*(x >= 0)

def triangle_pulse(x, A, x0):
    return A*(x/x0)*(x <= x0)*(x >= 0)

def trapezoidal_wave(x, A, rise_time, fall_time, f0, D):
    mod_time = x % (1/f0)
    cte_time = D/f0 - 0.5 * (rise_time + fall_time)
    
    t1, t2, t3 = rise_time, rise_time + cte_time, rise_time + cte_time + fall_time
    
    return A*(mod_time/rise_time)*(mod_time <= t1) + \
           A*(mod_time >= t1 )*(mod_time <= t2) + \
           (-A*mod_time/fall_time + A*(rise_time + cte_time + fall_time)/fall_time)*(mod_time >= t2)*((mod_time <= t3))

def ramp_pulse(x, A, x0):
    return A*(x/x0)*(x <= x0)*(x >= 0) + A*(x >= x0)
