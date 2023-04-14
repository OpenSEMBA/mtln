import numpy as np
import matplotlib.pyplot as plt
import skrf

import utils

tau = np.array([1e-3,1e-4,1e-5,1e-6,1e-9])
k = np.array([10,100,0.1,10,100])
freq = np.logspace(1,12,400)
eps = utils.epsDebyeErr(freq, k, tau, 0)
plt.plot(freq, np.real(eps), label='real')
plt.plot(freq, np.imag(eps), label='imag')
plt.plot(freq, np.abs(eps), label='abs')
plt.xlabel('f [Hz]')
plt.ylabel(r'$\varepsilon (\omega)$')
plt.xscale('log')
# plt.yscale('log')
plt.legend()
plt.show()

f = open("1port_err.s1p", "w")
f.write("# Hz S RI R 50\n")
for e,fq in zip(eps,freq):
    f.write(str(fq)+" "+str(np.real(e))+" "+str(np.imag(e))+"\n")
f.close()


nw = skrf.Network('1port_err.s1p')
vf = skrf.VectorFitting(nw)
vf.vector_fit(n_poles_real=10, n_poles_cmplx=0, fit_constant = True, fit_proportional = False)
# vf.plot_convergence()
# plt.show()

freqs1 = np.logspace(1,12,400)
vf.plot_s_mag(0, 0, freqs1 ) # plot s11
plt.xscale('log')
# plt.yscale('log')
plt.show()
print('CONSTANT')
print(vf.constant_coeff)
print('PROP')
print(vf.proportional_coeff)
print('POLES')
print(-1/vf.poles)
print('RESIDUES')
print(-vf.residues/vf.poles)

