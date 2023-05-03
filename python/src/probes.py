import numpy as np
from numpy.fft import fft, fftfreq, fftshift


class Probe:
    def __init__(self, position, type):
        self.type = type
        self.position = position
        self.t = np.array([])
        self.val = np.array([])

    def save(self, time, new_values):
        self.t = np.append(self.t, time)

        if self.val.shape == (0,):
            self.val = new_values
        else:
            self.val = np.vstack((self.val, new_values))

class Port:
    def __init__(self, v0: Probe, v1: Probe, i0: Probe, z0):
        self.v0 = v0
        self.v1 = v1
        self.i0 = i0
        self.z0 = np.diag(z0)

    def __get_v_i_fft(self):
        dt = self.v1.t[1] - self.v0.t[0]
        f = fftshift(fftfreq(len(self.v0.t), dt))
        v0_fft = fftshift(fft(self.v0.val, axis=0), axes=0)
        v1_fft = fftshift(fft(self.v1.val, axis=0), axes=0)
        v_fft = (v0_fft + v1_fft) / 2.0

        valAux = np.vstack((self.i0.val[0,:], self.i0.val))
        i = (valAux[:-1,:] + valAux[1:,:]) / 2.0
        i_fft = fftshift(fft(i, axis=0), axes=0)
        return f, v_fft, i_fft

    def __get_incident_and_reflected_power_wave(self, invertCurrent=False):
        f, v, i = self.__get_v_i_fft()
        if invertCurrent:
            i = -i
        a = (1.0/2.0) * (v + self.z0*i)/np.sqrt(self.z0)
        b = (1.0/2.0) * (v - self.z0*i)/np.sqrt(self.z0)
        return f, a, b

    @staticmethod
    def extract_s(port1, port2):
        ''' Using: https://en.wikipedia.org/wiki/Scattering_parameters '''
        f, a1, b1 = port1.__get_incident_and_reflected_power_wave()
        _, a2, b2 = port2.__get_incident_and_reflected_power_wave(
            invertCurrent=True)

        s = np.zeros((len(f), 2, 2), dtype=complex)
        s[:, 0, 0] = b1[:,0]/a1[:,0]
        s[:, 1, 0] = b2[:,0]/a1[:,0]
        s[:, 0, 1] = b1[:,0]/a2[:,0]
        s[:, 1, 1] = b2[:,0]/a2[:,0]

        return f, s
