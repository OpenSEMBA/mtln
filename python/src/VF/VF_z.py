import numpy as np
import matplotlib.pyplot as plt
import skrf

import utils 

# tau = np.array([1e-3,1e-4,1e-5,1e-6,1e-9])
# k = np.array([10,100,0.1,10,100])
freq = np.logspace(5,7,500)

z1 = utils.Y_RLseriesDisp(freq, 10, 1e-8, 1.0)
z2 = utils.Y_RLseriesDisp(freq, 10, 1e-8, 1.0)
z3 = utils.Y_RLseriesDisp(freq, 10, 1e-8, 1.0)
z4 = utils.Y_RLseriesDisp(freq, 10, 1e-8, 1.0)

# z1 = utils.RLpar(freq, 100, 1e-6)
# z2 = utils.RLpar(freq, 100, 1e-6)
# z3 = utils.RLpar(freq, 100, 1e-6)
# z4 = utils.RLpar(freq, 100, 1e-6)

# z1 = utils.z(freq, 1e6, 0.5)
# z2 = utils.z(freq, 1e3, 2)
# z3 = utils.z(freq, 1e4, 10)
# z4 = utils.z(freq, 5e4, 5)


# plt.plot(freq, np.real(z1), 'b--',label='real')
# plt.plot(freq, np.imag(z1), 'g--', label='imag')
# plt.plot(freq, np.abs(z1), 'k--', label='abs')
# plt.plot(freq, np.real(z2), 'b*',label='real')
# plt.plot(freq, np.imag(z2), 'g*', label='imag')
# plt.plot(freq, np.abs(z2), 'k*', label='abs')
# plt.xlabel('f [Hz]')
# plt.ylabel(r'$Z$')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.show()


f = open("4portZ.s4p", "w")
f.write("# HZ S RI R 50\n")
for imp1,imp2,imp3, imp4, fq in zip(z1,z2,z3,z4,freq):
    f.write(str(fq)+" "+str(np.real(imp1))+" "+str(np.imag(imp1))+" 0.0 0.0 0.0 0.0 0.0 0.0 "+
                    "0.0 0.0 "+str(np.real(imp2))+" "+str(np.imag(imp2))+" 0.0 0.0 0.0 0.0 "+
                    "0.0 0.0 0.0 0.0 "+str(np.real(imp3))+" "+str(np.imag(imp3))+" 0.0 0.0 "+
                    "0.0 0.0 0.0 0.0 0.0 0.0 "+str(np.real(imp4))+" "+str(np.imag(imp4))+"\n")
f.close()


nw = skrf.Network('4portZ.s4p')
vf = skrf.VectorFitting(nw)
n_poles_real = 2

n_poles_cmplx = 0
fitCte = False
fitProp = False

vf.vector_fit(n_poles_real, n_poles_cmplx, fit_constant = fitCte, fit_proportional = fitProp)
# vf.plot_convergence()
# plt.show()

freqs1 = np.logspace(5,7,500)
fig, ax = plt.subplots(2, 2)
fig.set_size_inches(12, 8)

vf.plot_s_re(0, 0, freqs1, ax=ax[0][0] ) # plot s11
vf.plot_s_im(0, 0, freqs1, ax=ax[0][1] ) # plot s11

vf.plot_s_mag(2, 2, freqs1, ax=ax[1][0] ) # plot s11
vf.plot_s_mag(3, 3, freqs1, ax=ax[1][1] ) # plot s11
ax[0][0].set_xscale('log')
ax[0][1].set_xscale('log')
ax[1][0].set_xscale('log')
ax[1][1].set_xscale('log')
# ax[0][0].set_yscale('log')
# ax[0][1].set_yscale('log')
# ax[1][0].set_yscale('log')
# ax[1][1].set_yscale('log')
fig.tight_layout()
plt.show()
print('CONSTANT')
print(vf.constant_coeff)
print(np.shape(vf.constant_coeff))
print('PROP')
print(vf.proportional_coeff)
print(np.shape(vf.proportional_coeff))
print('POLES')
print(vf.poles)
print('RESIDUES')
print(vf.residues)
print(np.shape(vf.residues))

# utils.serializeJSON(n_poles_real, n_poles_cmplx, fitCte, fitProp, 
#                     vf.constant_coeff, vf.proportional_coeff, vf.poles, vf.residues)

