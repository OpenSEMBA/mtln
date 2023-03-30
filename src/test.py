import numpy as np
import matplotlib.pyplot as plt

# #MnZN
# ks, kd = 108 + 0j, 4892 + 0j
# ws, wd = 57e6 + 0j, 7.1e6 + 0j
# beta = 18.3e6 + 0j
#NiZn
ks, kd = 113 + 0j, 527 + 0j
ws, wd = 150e6 + 0j, 238e6 + 0j
beta = 803e6 + 0j

w = np.logspace(5,9,100)
s = 1j*w

######

chi_s_1 = ks*ws**2/(w**2+ws**2)
k1 = 0.5*ks*ws**2/(-s*(s+ws))
k2 = 0.5*ks*ws**2/(-s*(s-ws))
chi_s_2 = k1+k2

######

chi_ss_1 = ks*w*ws/(w**2+ws**2)
k1 = 1j*0.5*ks*ws/(s+ws)
k2 = 1j*0.5*ks*ws/(s-ws)
chi_ss_2 = k1+k2

######

chi_d_1 = kd*wd**2*(wd**2-w**2)/((wd**2-w**2)**2+beta**2*w**2)

w1 = np.sqrt(0.5*((2*wd**2-beta**2)+beta*np.sqrt(beta**2-4*wd**2)))
w2 = np.sqrt(0.5*((2*wd**2-beta**2)-beta*np.sqrt(beta**2-4*wd**2)))
Ap = 0.5*(kd*wd**2*(wd**2-w1**2))/(w1**3-w1*w2**2)
Bp = -Ap
Cp = 0.5*(kd*wd**2*(wd**2-w2**2))/(w2**3-w2*w1**2)
Dp = -Cp
chi_d_2 = 0+0j
chi_d_2 = Ap/(w-w1) + Bp/(w+w1) + Cp/(w-w2) + Dp/(w+w2)

######

chi_dd_1 = kd*wd**2*beta*w/((wd**2-w**2)**2+beta**2*w**2)
App = -0.5*(kd*wd**2*beta)/(w2**2-w1**2)
Bpp = App
Cpp = -App
Dpp = -App

chi_dd_2 = App/(w-w1) + Bpp/(w+w1) + Cpp/(w-w2) + Dpp/(w+w2)

######


plt.figure(figsize=(8, 5))
plt.subplot(221)
plt.plot(w, np.real(chi_s_1),'r', label = r'$\chi_s^{\prime}$')
plt.plot(w, np.real(chi_s_2),'b--', label = r'$\chi_{s;pr}^{\prime}$')
plt.plot(w, np.imag(chi_s_2),'g--', label = r'$\chi_{s;pr}^{\prime}$')
plt.xscale('log')
plt.legend()

plt.subplot(222)
plt.plot(w, np.real(chi_ss_1),'r', label = r'$\chi_s^{\prime\prime}$')
plt.plot(w, np.real(chi_ss_2),'b--', label = r'$\chi_{s;pr}^{\prime\prime}$')
plt.plot(w, np.imag(chi_ss_2),'g--', label = r'$\chi_{s;pr}^{\prime\prime}$')
plt.xscale('log')
plt.legend()

plt.subplot(223)
plt.plot(w, np.real(chi_d_1),'r', label = r'$\chi_d^{\prime}$')
plt.plot(w, np.real(chi_d_2),'b--', label = r'$\chi_{d;pr}^{\prime}$')
plt.plot(w, np.imag(chi_d_2),'g--', label = r'$\chi_{d;pr}^{\prime}$')
plt.xscale('log')
plt.legend()

plt.subplot(224)
plt.plot(w, np.real(chi_dd_1),'r', label = r'$\chi_d^{\prime\prime}$')
plt.plot(w, np.real(chi_dd_2),'b--', label = r'$\chi_{d;pr}^{\prime\prime}$')
plt.plot(w, np.imag(chi_dd_2),'g--', label = r'$\chi_{d;pr}^{\prime\prime}$')
plt.xscale('log')
# plt.yscale('log')
plt.legend()

plt.show()

mu = chi_s_1 + chi_d_1 + 1 - 1j*(chi_ss_1 + chi_dd_1)
ZL = 1j*w*mu
plt.subplot(211)
plt.semilogx(w, np.abs(mu), 'b', label = r'$|\mu|$')
plt.semilogx(w, np.real(mu), 'b.', label = r'$R(\mu)$')
plt.semilogx(w, -np.imag(mu), 'b--', label = r'$I(\mu)$')
plt.legend()
plt.subplot(212)
plt.loglog(w, np.abs(ZL), 'b', label = r'$|Z_L|$')
# plt.semilogx(w, np.real(ZL), 'b.', label = r'$R(Z_L)$')
# plt.semilogx(w, -np.imag(ZL), 'b--', label = r'$I(Z_L)$')
plt.legend()
plt.show()
