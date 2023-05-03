import numpy as np
import skrf as rf

from copy import deepcopy

import sympy as sp

from types import FunctionType
from types import LambdaType

from .probes import *


class Field:
    def __init__(self, incident_x, incident_z):
        # assert (type(incident_x) == LambdaType and type(incident_z) == LambdaType)

        self.e_x = incident_x
        self.e_z = incident_z


class MTL:
    """
    Lossless Multiconductor Transmission Line
    """

    def __init__(self, l, c, length=1.0, nx=100, Zs=0.0, Zl=0.0):
        self.x = np.linspace(0, length, nx+1)

        types = [float, np.float64]

        if type(l) in types and type(c) in types:
            self.l = np.array([[l]])
            self.c = np.array([[c]])
        elif type(l) == np.ndarray and type(c) == np.ndarray:
            assert (l.shape == c.shape)
            assert (l.shape[0] == l.shape[1])
            self.l = l
            self.c = c
        else:
            raise ValueError("Invalid input L and/or C")

        self.number_of_conductors = np.shape(self.l)[0]

        if (type(Zs) != np.ndarray and type(Zl) != np.ndarray):
            self.zs = Zs * np.eye(self.number_of_conductors)
            self.zl = Zl * np.eye(self.number_of_conductors)
        elif (type(Zs) == np.ndarray and type(Zl) == np.ndarray):
            assert (Zs.shape == Zl.shape)
            if (len(Zs.shape) == 1):
                assert (Zs.shape[0] == self.number_of_conductors)
                self.zs = Zs.reshape(
                    self.number_of_conductors, 1) * np.eye(self.number_of_conductors)
                self.zl = Zl.reshape(
                    self.number_of_conductors, 1) * np.eye(self.number_of_conductors)

            elif (len(Zs.shape) == 2):
                assert (Zs.shape[0] == 1 or Zs.shape[1] == 1)
                assert (
                    Zs.shape[0] == self.number_of_conductors or Zs.shape[1] == self.number_of_conductors)
                self.zs = Zs * np.eye(self.number_of_conductors)
                self.zl = Zl * np.eye(self.number_of_conductors)
        else:
            raise ValueError(
                "Invalid input Zs and/or Zc. Use two floats or two arrays")

        self.v = np.zeros([self.number_of_conductors, self.x.shape[0]])
        self.i = np.zeros([self.number_of_conductors, self.x.shape[0]-1])

        self.probes = []
        self.port_probes = []
        self.v_sources = np.empty(
            shape=(self.number_of_conductors, self.x.shape[0]), dtype=object)
        self.v_sources.fill(lambda n: 0)

        self.e_L = np.empty(shape=(self.number_of_conductors,
                            self.x.shape[0]-1), dtype=object)
        self.e_T = np.empty(
            shape=(self.number_of_conductors, self.x.shape[0]), dtype=object)
        self.e_L.fill(lambda n: 0)
        self.e_T.fill(lambda n: 0)

        self.time = 0.0

        self.dx = self.x[1] - self.x[0]

        self.dt = self.get_max_timestep()

        self.v_term = np.eye(self.number_of_conductors)
        self.i_term = np.eye(self.number_of_conductors)

        self.i_diff = self.dt / self.dx * np.linalg.inv(self.c)
        self.v_diff = self.dt / self.dx * np.linalg.inv(self.l)

        left_port_c = np.matmul(self.zs, self.c)
        right_port_c = np.matmul(self.zl, self.c)

        self.left_port_term_1 = np.matmul(
            np.linalg.inv(self.dx*left_port_c/self.dt +
                          np.eye(self.number_of_conductors)),
            self.dx*left_port_c/self.dt-np.eye(self.number_of_conductors))

        self.left_port_term_2 = np.linalg.inv(
            self.dx*left_port_c/self.dt+np.eye(self.number_of_conductors))

        self.right_port_term_1 = np.matmul(
            np.linalg.inv(self.dx*right_port_c/self.dt +
                          np.eye(self.number_of_conductors)),
            self.dx*right_port_c/self.dt-np.eye(self.number_of_conductors))

        self.right_port_term_2 = np.linalg.inv(
            self.dx*right_port_c/self.dt+np.eye(self.number_of_conductors))

    def get_phase_velocities(self):
        return 1/np.sqrt(np.diag(self.l.dot(self.c)))

    def get_max_timestep(self):
        return self.dx / np.max(self.get_phase_velocities())

    def set_time_step(self, NDT, final_time):
        self.dt = final_time/NDT

    def get_time_range(self, final_time):
        return np.arange(0, np.floor(final_time / self.dt))

    def set_voltage(self, conductor, voltage):
        self.v[conductor] = voltage(self.x)

    def update_probes(self):
        for probe in self.probes:
            index = np.argmin(np.abs(self.x - probe.position))
            if probe.type == "voltage":
                probe.t = np.append(probe.t, self.time)
                probe.val = np.append(
                    probe.val, self.v[probe.conductor, index])
            elif probe.type == "current":
                probe.t = np.append(probe.t, self.time + self.dt/2.0)
                if index == self.i.shape[1]:
                    probe.val = np.append(
                        probe.val, self.i[probe.conductor, index-1])
                else:
                    probe.val = np.append(
                        probe.val, self.i[probe.conductor, index])
            else:
                raise ValueError("undefined probe")

    def update_sources(self):
        self.v_sources_now = np.vectorize(FunctionType.__call__, otypes=[
                                          "float64"])(self.v_sources, self.time)
        self.v_sources_prev = np.vectorize(FunctionType.__call__, otypes=[
                                           "float64"])(self.v_sources, self.time-self.dt)

        self.e_L_now = np.vectorize(FunctionType.__call__, otypes=[
                                    "float64"])(self.e_L, self.time+self.dt/2)
        self.e_L_prev = np.vectorize(FunctionType.__call__, otypes=[
                                     "float64"])(self.e_L, self.time-self.dt/2)
        self.e_T_now = np.vectorize(FunctionType.__call__, otypes=[
                                    "float64"])(self.e_T, self.time)
        self.e_T_prev = np.vectorize(FunctionType.__call__, otypes=[
                                     "float64"])(self.e_T, self.time-self.dt)

    def step(self):

        self.update_sources()

        self.v[:, 0] = self.left_port_term_1.dot(self.v[:, 0]) + \
            self.left_port_term_2.dot(-2*np.matmul(self.zs, self.i[:, 0]) +
                                      (self.v_sources_now[:, 0] + self.v_sources_prev[:, 0]) -
                                      (self.dx/self.dt) * (self.zs.dot(self.c)).dot(self.e_T_now[:, 0] - self.e_T_prev[:, 0]))

        self.v[:, 1:-1] = self.v_term.dot(self.v[:, 1:-1]) - \
            self.i_diff.dot(self.i[:, 1:]-self.i[:, :-1]) - \
            (self.e_T_now[:, 1:-1] - self.e_T_prev[:, 1:-1])

        self.v[:, -1] = self.right_port_term_1.dot(self.v[:, -1]) + \
            self.right_port_term_2.dot(+2*np.matmul(self.zl, self.i[:, -1]) +
                                       (self.v_sources_now[:, -1] + self.v_sources_prev[:, -1]) -
                                       (self.dx/self.dt)*(self.zl.dot(self.c)).dot(self.e_T_now[:, -1] - self.e_T_prev[:, -1]))

        self.i[:, :] = self.i_term.dot(self.i[:, :]) -\
            self.v_diff.dot(self.v[:, 1:]-self.v[:, :-1] +
                            (self.e_T_now[:, 1:] - self.e_T_now[:, :-1]) -
                            (self.dx/2)*(self.e_L_now[:, :] + self.e_L_prev[:, :]))

        self.time += self.dt
        self.update_probes()

    def run_until(self, finalTime):
        for _ in self.get_time_range(finalTime):
            self.step()

    def add_voltage_source(self, position: float, conductor: int, magnitude):
        index = np.argmin(np.abs(self.x - position))
        self.v_sources[conductor, index] = magnitude
        probe = self.add_probe(position, conductor, 'voltage')
        return probe

    def add_external_field(self, e_x, e_z, ref_distance, distances: np.ndarray):

        field = Field(e_x, e_z)
        ex = sp.Function('ex')
        ez = sp.Function('ez')
        x, z, t, v = sp.symbols("x z t v")
        ex = field.e_x
        ez = field.e_z

        vmax = np.max(self.get_phase_velocities())

        for n in range(self.number_of_conductors):
            et = ex(x, z, t).integrate(x, (x, ref_distance, distances[n]))
            for nz in range(self.x.size - 1):
                pos = self.x[nz]
                self.e_L[n, nz] = sp.lambdify(t,
                                              ez(x, z, t).subs(x, distances[n]).subs(v, vmax).subs(z, pos - 0.5*self.dx) -
                                              ez(x, z, t).subs(x, ref_distance).subs(
                                                  v, vmax).subs(z, pos - 0.5*self.dx)
                                              )
                self.e_T[n, nz] = sp.lambdify(t, et.subs(z, pos))

            nz = self.x.size - 1
            pos = self.x[nz]
            self.e_T[n, nz] = sp.lambdify(t, et.subs(z, pos))

    def add_probe(self, position: float, conductor: int, type: str):
        if (position > self.x[-1]) or (position < 0.0):
            raise ValueError("Probe position is out of MTL length.")

        probe = Probe(position, conductor, type)
        self.probes.append(probe)
        return probe

    def add_port_probe(self, terminal, conductor):
        if terminal == 0:
            x0 = self.x[0]
            x1 = self.x[1]
        if terminal == 1:
            x0 = self.x[-2]
            x1 = self.x[-1]

        v0 = self.add_probe(position=x0, conductor=0, type='voltage')
        v1 = self.add_probe(position=x1, conductor=0, type='voltage')
        i0 = self.add_probe(position=(x0+x1)/2.0, conductor=0, type='current')

        port_probe = PortProbe(v0, v1, i0)
        self.port_probes.append(port_probe)

        return port_probe

    def create_clean_copy(self):
        r = deepcopy(self)
        r.v.fill(0.0)
        r.i.fill(0.0)
        r.time = 0.0
        return r

    def extract_network(self, fMin, fMax, finalTime):

        # line = self.create_clean_copy()

        # spread = 1/fMax/2.0
        # delay = 8*spread

        # def gauss(t):
        #     return np.exp(- (t-delay)**2 / (2*spread**2))
        # line.add_voltage_source(position=line.x[0], conductor=0, magnitude=gauss)
        # port_1 = line.add_port_probe(terminal=0, conductor=0)
        # port_2 = line.add_port_probe(terminal=1, conductor=0)

        # line.run_until(finalTime)

        # f, s = PortProbe.extract_s(port_1, port_2, [self.zs, self.zl])
        # fq = rf.Frequency.from_f(f[(f >= fMin) & (f < fMax)], unit='Hz')
        # s = s[(f >= fMin) & (f < fMax), :, :]
        # ntw = rf.Network(frequency=fq, s=s)

        line = self.create_clean_copy()

        spread = 1/fMax/2.0
        delay = 8*spread

        def gauss(t):
            return np.exp(- (t-delay)**2 / (2*spread**2))
        line.add_voltage_source(
            position=self.x[0], conductor=0, magnitude=gauss)
        p11 = line.add_port_probe(terminal=0, conductor=0)

        for _ in line.get_time_range(finalTime):
            line.step()

        f, z11_fft = p11.extract_z()
        fq = rf.Frequency.from_f(f[(f >= fMin) & (f < fMax)], unit='Hz')
        z11 = np.zeros((len(fq.f), 1, 1), dtype="complex64")
        z11[:, 0, 0] = z11_fft[(f >= fMin) & (f < fMax)]

        ntw = rf.Network.from_z(z11)
        ntw.frequency = fq

        return ntw


class MTL_losses(MTL):
    def __init__(self, l, c, g, r, length=1.0, nx=100, Zs=0.0, Zl=0.0):
        super().__init__(l, c, length=length, nx=nx, Zs=Zs, Zl=Zl)

        if type(g) == float and type(r) == float:
            self.g = np.array([[g]])
            self.r = np.array([[r]])
        elif type(g) == np.ndarray and type(r) == np.ndarray:
            assert (g.shape == r.shape)
            assert (g.shape[0] == r.shape[1])
            self.g = g
            self.r = r
        else:
            raise ValueError("Invalid input G and/or R")

        dx = self.dx

        self.v_term = np.linalg.inv(
            (dx/self.dt)*self.c + (dx/2)*self.g).dot((dx/self.dt)*self.c - (dx/2)*self.g)
        self.i_term = np.linalg.inv(
            (dx/self.dt)*self.l + (dx/2)*self.r).dot((dx/self.dt)*self.l - (dx/2)*self.r)

        self.i_diff = np.linalg.inv((dx/self.dt)*self.c + (dx/2)*self.g)
        self.v_diff = np.linalg.inv((dx/self.dt)*self.l + (dx/2)*self.r)

        self.left_port_term_1 = np.matmul(
            np.linalg.inv(dx*(np.matmul(self.zs, self.c))/self.dt + dx *
                          (np.matmul(self.zs, self.g))/2 + np.eye(self.number_of_conductors)),
            dx*(np.matmul(self.zs, self.c))/self.dt - dx*(np.matmul(self.zs, self.g))/2 - np.eye(self.number_of_conductors))

        self.left_port_term_2 = np.linalg.inv(dx*(np.matmul(self.zs, self.c))/self.dt + dx*(
            np.matmul(self.zs, self.g))/2 + np.eye(self.number_of_conductors))

        self.right_port_term_1 = np.matmul(
            np.linalg.inv(dx*(np.matmul(self.zl, self.c))/self.dt + dx *
                          (np.matmul(self.zl, self.g))/2 + np.eye(self.number_of_conductors)),
            dx*(np.matmul(self.zl, self.c))/self.dt - dx*(np.matmul(self.zl, self.g))/2 - np.eye(self.number_of_conductors))

        self.right_port_term_2 = np.linalg.inv(dx*(np.matmul(self.zl, self.c))/self.dt + dx*(
            np.matmul(self.zl, self.g))/2 + np.eye(self.number_of_conductors))
