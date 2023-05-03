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
    def __init__(self, v0: Probe, v1: Probe, i0: Probe, z0: float):
        self.v0 = v0
        self.v1 = v1
        self.i0 = i0
        self.z0 = z0

    def __get_v_i_fft(self):
        dt = self.v1.t[1] - self.v0.t[0]
        f = fftshift(fftfreq(len(self.v0.t), dt))
        v0_fft = fftshift(fft(self.v0.val))
        v1_fft = fftshift(fft(self.v1.val))
        v_fft = (v0_fft + v1_fft) / 2.0

        valAux = np.append(self.i0.val[0], self.i0.val)
        i = (valAux[:-1] + valAux[1:]) / 2.0
        i_fft = fftshift(fft(i))
        return f, v_fft, i_fft

    def __get_incident_and_reflected_power_wave(self, invertCurrent=False):
        f, v1, i1 = self.__get_v_i_fft()
        if invertCurrent:
            i1 = -i1
        a1 = (1.0/2.0) * (v1 + self.z0*i1)/np.sqrt(self.z0)
        b1 = (1.0/2.0) * (v1 - self.z0*i1)/np.sqrt(self.z0)
        return f, a1, b1

    @staticmethod
    def extract_s(port1, port2):
        ''' Using: https://en.wikipedia.org/wiki/Scattering_parameters '''
        f, a1, b1 = port1.__get_incident_and_reflected_power_wave()
        _, a2, b2 = port2.__get_incident_and_reflected_power_wave(invertCurrent=True)

        s = np.zeros((len(f), 2, 2), dtype=complex)
        s[:, 0, 0] = b1/a1
        s[:, 1, 0] = b2/a1
        s[:, 0, 1] = b1/a2
        s[:, 1, 1] = b2/a2

        return f, s
