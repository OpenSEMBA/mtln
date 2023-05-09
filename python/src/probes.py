import numpy as np
from numpy.fft import fft, fftfreq, fftshift

class Probe:
    def __init__(self, position, type, dt, x):
        self.type = type

        self.t = np.array([])
        self.val = np.array([])

        self.dt = dt
        self.current_frame = 0

        self.position = position
        self.index = np.argmin(np.abs(x - position))

    def resize_frames(self, num_frames, num_conductors):
        self.current_frame = 0
        self.t = np.zeros(num_frames)
        self.val = np.zeros((num_frames, num_conductors))

    def update(self, t, x, v, i):
        if self.type == "voltage":
            self.__save_frame(t, v[:, self.index])
        elif self.type == "current":
            t = t + self.dt/2.0
            if self.index == i.shape[1]:
                self.__save_frame(t, i[:, self.index-1])
            else:
                self.__save_frame(t, i[:, self.index])
        else:
            raise ValueError("undefined probe")

    def __save_frame(self, time, new_values):
        if self.current_frame < len(self.t):
            self.t[self.current_frame] = time
            self.val[self.current_frame] = new_values
        else:
            self.t = np.append(self.t, time)
            if self.val.shape == (0,):
                self.val = new_values
            else:
                self.val = np.vstack((self.val, new_values))

        self.current_frame += 1


class Port:
    def __init__(self, v0: Probe, v1: Probe, i0: Probe, z0):
        self.v0 = v0
        self.v1 = v1
        self.i0 = i0
        self.z0 = np.diag(z0)

    def __get_v_i_fft(self, n):
        dt = self.v1.t[1] - self.v0.t[0]
        f = fftshift(fftfreq(len(self.v0.t), dt))
        v0_fft = fftshift(fft(self.v0.val[:, n]))
        v1_fft = fftshift(fft(self.v1.val[:, n]))
        v_fft = (v0_fft + v1_fft) / 2.0

        valAux = np.vstack((self.i0.val[0, :], self.i0.val))
        i = (valAux[:-1, :] + valAux[1:, :]) / 2.0
        i_fft = fftshift(fft(i[:, n]))
        return f, v_fft, i_fft
    
    def getTerminal(self):
        if self.v0.position == 0.0:
            return "S"
        else:
            return "L"

    def get_incident_and_reflected_power_wave(self, conductor):
        z0 = self.z0[conductor]
        assert (z0 != 0.0)

        f, v, i = self.__get_v_i_fft(conductor)
        if self.getTerminal() == "L":
            i = -i

        a = (1.0/2.0) * (v + z0*i)/np.sqrt(z0)
        b = (1.0/2.0) * (v - z0*i)/np.sqrt(z0)

        return f, a, b