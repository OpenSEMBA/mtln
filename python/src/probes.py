import numpy as np
from numpy.fft import fft, fftfreq, fftshift

class Probe:
    def __init__(self, position, conductor, type):
        self.type = type
        self.position = position
        self.conductor = conductor
        self.t = np.array([])
        self.val = np.array([])


class PortProbe:
    def __init__(self, v0: Probe, v1: Probe, i0: Probe):
        self.v0 = v0
        self.v1 = v1
        self.i0 = i0

    def get_v_i_fft(self):
        dt = self.v1.t[1] - self.v0.t[0]
        f = fftshift(fftfreq(len(self.v0.t), dt))
        v0_fft = fftshift(fft(self.v0.val))
        v1_fft = fftshift(fft(self.v1.val))
        v_fft = (v0_fft + v1_fft) / 2.0

        valAux = np.append(self.i0.val[0], self.i0.val)
        i = (valAux[:-1] + valAux[1:]) / 2.0
        i_fft = fftshift(fft(i))
        return f, v_fft, i_fft

    def extract_z(self):
        dt = self.v1.t[1] - self.v0.t[0]
        f = fftshift(fftfreq(len(self.v0.t), dt))
        v0_fft = fftshift(fft(self.v0.val))
        v1_fft = fftshift(fft(self.v1.val))

        tAux = np.append(- dt/2.0, self.i0.t)
        self.i0.t = (tAux[:-1] + tAux[1:]) / 2.0
        valAux = np.append(self.i0.val[0], self.i0.val)
        self.i0.val = (valAux[:-1] + valAux[1:]) / 2.0
        i0_fft = fftshift(fft(self.i0.val))
        z11_fft = (v0_fft + v1_fft) / 2.0 / i0_fft

        return f, z11_fft

    @staticmethod
    def extract_s(port1, port2, z0 = [[50], [50]]):
        ''' Using: https://en.wikipedia.org/wiki/Scattering_parameters '''
        f, v1, i1 = port1.get_v_i_fft()
        _, v2, i2 = port2.get_v_i_fft()
        i2 = -i2

        a1 = (1.0/2.0) * (v1 + z0[0][0]*i1)/np.sqrt(z0[0][0]) # TODO Only valid for first conductor!!
        b1 = (1.0/2.0) * (v1 - z0[0][0]*i1)/np.sqrt(z0[0][0])

        a2 = (1.0/2.0) * (v2 + z0[1][0]*i2)/np.sqrt(z0[1][0])
        b2 = (1.0/2.0) * (v2 - z0[1][0]*i2)/np.sqrt(z0[1][0])

        s = np.zeros((len(f), 2, 2), dtype=complex)
        s[:,0,0] = b1/a1
        s[:,1,0] = b2/a1
        s[:,0,1] = b1/a2
        s[:,1,1] = b2/a2
        
        return f, s