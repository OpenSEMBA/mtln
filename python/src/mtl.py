import numpy as np

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
        self.x = np.linspace(0, length, nx + 1)

        types = [float, np.float64]

        if type(l) in types and type(c) in types:
            self.l = np.empty(shape=(self.x.shape[0] - 1, 1, 1))
            self.c = np.empty(shape=(self.x.shape[0], 1, 1))
            self.l[:] = l
            self.c[:] = c
        elif type(l) == np.ndarray and type(c) == np.ndarray:
            assert l.shape == c.shape
            assert l.shape[0] == l.shape[1]
            n = l.shape[0]
            self.l = np.empty(shape=(self.x.shape[0] - 1, n, n))
            self.c = np.empty(shape=(self.x.shape[0], n, n))
            self.l[:] = l
            self.c[:] = c
        else:
            raise ValueError("Invalid input L and/or C")

        self.number_of_conductors = np.shape(self.l)[1]

        if type(Zs) != np.ndarray and type(Zl) != np.ndarray:
            self.zs = Zs * np.eye(self.number_of_conductors)
            self.zl = Zl * np.eye(self.number_of_conductors)
        elif type(Zs) == np.ndarray and type(Zl) == np.ndarray:
            assert Zs.shape == Zl.shape
            if len(Zs.shape) == 1:
                assert Zs.shape[0] == self.number_of_conductors
                self.zs = Zs.reshape(self.number_of_conductors, 1) * np.eye(
                    self.number_of_conductors
                )
                self.zl = Zl.reshape(self.number_of_conductors, 1) * np.eye(
                    self.number_of_conductors
                )

            elif len(Zs.shape) == 2:
                assert Zs.shape[0] == 1 or Zs.shape[1] == 1
                assert (
                    Zs.shape[0] == self.number_of_conductors
                    or Zs.shape[1] == self.number_of_conductors
                )
                self.zs = Zs * np.eye(self.number_of_conductors)
                self.zl = Zl * np.eye(self.number_of_conductors)
        else:
            raise ValueError("Invalid input Zs and/or Zc. Use two floats or two arrays")

        self.v = np.zeros([self.number_of_conductors, self.x.shape[0]])
        self.i = np.zeros([self.number_of_conductors, self.x.shape[0] - 1])

        self.probes = []
        self.port_probes = []
        self.v_sources = np.empty(
            shape=(self.number_of_conductors, self.x.shape[0]), dtype=object
        )
        self.v_sources.fill(lambda n: 0)

        self.e_L = np.empty(
            shape=(self.number_of_conductors, self.x.shape[0] - 1), dtype=object
        )
        self.e_T = np.empty(
            shape=(self.number_of_conductors, self.x.shape[0]), dtype=object
        )
        self.e_L.fill(lambda n: 0)
        self.e_T.fill(lambda n: 0)

        self.time = 0.0

        self.dx = self.x[1] - self.x[0]

        self.dt = self.get_max_timestep()

        self.v_term = np.empty(
            shape=(
                self.x.shape[0],
                self.number_of_conductors,
                self.number_of_conductors,
            )
        )
        self.i_term = np.empty(
            shape=(
                self.x.shape[0] - 1,
                self.number_of_conductors,
                self.number_of_conductors,
            )
        )
        self.v_term[:] = np.eye(self.number_of_conductors)
        self.i_term[:] = np.eye(self.number_of_conductors)

        self.i_diff = self.dt / self.dx * np.linalg.inv(self.c)
        self.v_diff = self.dt / self.dx * np.linalg.inv(self.l)

        left_port_c = np.matmul(self.zs, self.c[0])
        right_port_c = np.matmul(self.zl, self.c[-1])

        self.left_port_term_1 = np.matmul(
            np.linalg.inv(
                self.dx * left_port_c / self.dt + np.eye(self.number_of_conductors)
            ),
            self.dx * left_port_c / self.dt - np.eye(self.number_of_conductors),
        )

        self.left_port_term_2 = np.linalg.inv(
            self.dx * left_port_c / self.dt + np.eye(self.number_of_conductors)
        )

        self.right_port_term_1 = np.matmul(
            np.linalg.inv(
                self.dx * right_port_c / self.dt + np.eye(self.number_of_conductors)
            ),
            self.dx * right_port_c / self.dt - np.eye(self.number_of_conductors),
        )

        self.right_port_term_2 = np.linalg.inv(
            self.dx * right_port_c / self.dt + np.eye(self.number_of_conductors)
        )

    def get_phase_velocities(self):
        return np.array([
            1 / np.sqrt(np.diag(self.l[k].dot(self.c[1 + k])))
            for k in range(self.x.shape[0] - 1)
        ])

    def get_max_timestep(self):
        return self.dx / np.max(self.get_phase_velocities())

    def set_time_step(self, NDT, final_time):
        self.dt = final_time / NDT

    def get_time_range(self, final_time):
        return np.arange(0, np.floor(final_time / self.dt))

    def set_voltage(self, conductor, voltage):
        self.v[conductor] = voltage(self.x)

    def update_probes(self):
        for p in self.probes:
            p.update(self.time, self.x, self.v, self.i)

    def update_sources(self):
        self.v_sources_now = np.vectorize(FunctionType.__call__, otypes=["float64"])(
            self.v_sources, self.time
        )
        self.v_sources_prev = np.vectorize(FunctionType.__call__, otypes=["float64"])(
            self.v_sources, self.time - self.dt
        )

        self.e_L_now = np.vectorize(FunctionType.__call__, otypes=["float64"])(
            self.e_L, self.time + self.dt / 2
        )
        self.e_L_prev = np.vectorize(FunctionType.__call__, otypes=["float64"])(
            self.e_L, self.time - self.dt / 2
        )
        self.e_T_now = np.vectorize(FunctionType.__call__, otypes=["float64"])(
            self.e_T, self.time
        )
        self.e_T_prev = np.vectorize(FunctionType.__call__, otypes=["float64"])(
            self.e_T, self.time - self.dt
        )

        

    def step(self):
        self.update_sources()

        self.v[:, 0] = self.left_port_term_1.dot(
            self.v[:, 0]
        ) + self.left_port_term_2.dot(
            -2 * np.matmul(self.zs, self.i[:, 0])
            + (self.v_sources_now[:, 0] + self.v_sources_prev[:, 0])
            - (self.dx / self.dt)
            * (self.zs.dot(self.c[0])).dot(self.e_T_now[:, 0] - self.e_T_prev[:, 0])
        )

        self.v[:, 1:-1] = np.einsum('...ij,...j->...i',self.v_term[1:-1,:,:],self.v.T[1:-1,:]).T-\
                          np.einsum('...ij,...j->...i',self.i_diff[1:-1,:,:],(self.i[:,1:]-self.i[:,:-1]).T).T-\
                          (self.e_T_now[:, 1:-1] - self.e_T_prev[:, 1:-1])


        self.v[:, -1] = self.right_port_term_1.dot(
            self.v[:, -1]
        ) + self.right_port_term_2.dot(
            +2 * np.matmul(self.zl, self.i[:, -1])
            + (self.v_sources_now[:, -1] + self.v_sources_prev[:, -1])
            - (self.dx / self.dt)
            * (self.zl.dot(self.c[-1])).dot(self.e_T_now[:, -1] - self.e_T_prev[:, -1])
        )


        self.i[:, :] = np.einsum('...ij,...j->...i',self.i_term[:,:,:],self.i.T[:,:]).T-\
                       np.einsum('...ij,...j->...i',self.v_diff[:,:,:],(self.v[:, 1:] - self.v[:, :-1]
                                                    + (self.e_T_now[:, 1:] - self.e_T_now[:, :-1])
                                                    - (self.dx / 2) * (self.e_L_now[:, :] + self.e_L_prev[:, :])).T).T


        self.time += self.dt
        self.update_probes()

    def run_until(self, finalTime):
        t = self.get_time_range(finalTime)

        for p in self.probes:
            p.resize_frames(len(t), self.number_of_conductors)

        for _ in t:
            self.step()

    def add_voltage_source(self, position: float, conductor: int, magnitude):
        index = np.argmin(np.abs(self.x - position))
        self.v_sources[conductor, index] = magnitude

    def add_external_field(self, e_x, e_z, ref_distance, distances: np.ndarray):
        field = Field(e_x, e_z)
        ex = sp.Function("ex")
        ez = sp.Function("ez")
        x, z, t, v = sp.symbols("x z t v")
        ex = field.e_x
        ez = field.e_z

        vmax = np.max(self.get_phase_velocities())

        for n in range(self.number_of_conductors):
            et = ex(x, z, t).integrate(x, (x, ref_distance, distances[n]))
            for nz in range(self.x.size - 1):
                pos = self.x[nz]
                self.e_L[n, nz] = sp.lambdify(
                    t,
                    ez(x, z, t)
                    .subs(x, distances[n])
                    .subs(v, vmax)
                    .subs(z, pos - 0.5 * self.dx)
                    - ez(x, z, t)
                    .subs(x, ref_distance)
                    .subs(v, vmax)
                    .subs(z, pos - 0.5 * self.dx),
                )
                self.e_T[n, nz] = sp.lambdify(t, et.subs(z, pos))

            nz = self.x.size - 1
            pos = self.x[nz]
            self.e_T[n, nz] = sp.lambdify(t, et.subs(z, pos))

    def add_probe(self, position: float, type: str):
        if (position > self.x[-1]) or (position < 0.0):
            raise ValueError("Probe position is out of MTL length.")

        probe = Probe(position, type, self.dt, self.x)
        self.probes.append(probe)
        return probe

    def add_port_probe(self, terminal):
        if terminal == 0:
            x0 = self.x[0]
            x1 = self.x[1]
            z0 = self.zs
        if terminal == 1:
            x0 = self.x[-2]
            x1 = self.x[-1]
            z0 = self.zl

        v0 = self.add_probe(position=x0,          type='voltage')
        v1 = self.add_probe(position=x1,          type='voltage')
        i0 = self.add_probe(position=(x0+x1)/2.0, type='current')

        port_probe = Port(v0, v1, i0, z0)
        self.port_probes.append(port_probe)

        return port_probe
    
    def add_port_probes(self):
        return self.add_port_probe(0), self.add_port_probe(1)

    def create_clean_copy(self):
        r = deepcopy(self)
        r.v.fill(0.0)
        r.i.fill(0.0)
        r.time = 0.0
        return r


class MTL_losses(MTL):
    def __init__(self, l, c, g, r, length=1.0, nx=100, Zs=0.0, Zl=0.0):
        super().__init__(l, c, length=length, nx=nx, Zs=Zs, Zl=Zl)

        types = [float, np.float64]

        if type(g) in types and type(r) in types:
            self.r = np.empty(shape=(self.x.shape[0] - 1, self.number_of_conductors, self.number_of_conductors))
            self.g = np.empty(shape=(self.x.shape[0], self.number_of_conductors, self.number_of_conductors))
            self.g[:] = g
            self.r[:] = r
        elif type(g) == np.ndarray and type(r) == np.ndarray:
            assert g.shape == r.shape
            assert g.shape[0] == r.shape[1]
            n = g.shape[0]
            self.r = np.empty(shape=(self.x.shape[0] - 1, n, n))
            self.g = np.empty(shape=(self.x.shape[0], n, n))
            self.r[:] = r
            self.g[:] = g
        else:
            raise ValueError("Invalid input G and/or R")


        self.phi = np.zeros(
            shape=(
                self.x.shape[0]-1,
                self.number_of_conductors
            ), dtype = np.ndarray
        )
        self.q1 = np.zeros(
            shape=(
                self.x.shape[0]-1,
                self.number_of_conductors,
                self.number_of_conductors
            ), dtype = np.ndarray
        )
        self.q2 = np.zeros(
            shape=(
                self.x.shape[0]-1,
                self.number_of_conductors,
                self.number_of_conductors
            ), dtype = np.ndarray
        )
        self.q3 = np.zeros(
            shape=(
                self.x.shape[0]-1,
                self.number_of_conductors,
                self.number_of_conductors
            ), dtype = np.ndarray
        )
        self.q3_phi_term = np.zeros(
            shape=(
                self.x.shape[0]-1,
                self.number_of_conductors
            )
        )
        self.d = np.zeros(
            shape=(
                self.x.shape[0]-1,
                self.number_of_conductors,
                self.number_of_conductors,
            )
        )
        self.e = np.zeros(
            shape=(
                self.x.shape[0]-1,
                self.number_of_conductors,
                self.number_of_conductors,
            )
        )
        self.q1sum = np.zeros(
            shape=(
                self.x.shape[0]-1,
                self.number_of_conductors,
                self.number_of_conductors,
            )
        )
        self.q2sum = np.zeros(
            shape=(
                self.x.shape[0]-1,
                self.number_of_conductors,
                self.number_of_conductors,
            )
        )

        dx = self.dx

        # for kz in range(self.x.shape[0] - 1):
        #     self.v_term[kz] = np.linalg.inv(
        #         (dx / self.dt) * self.c[kz] + (dx / 2) * self.g[kz]
        #     ).dot((dx / self.dt) * self.c[kz] - (dx / 2) * self.g[kz])
            # self.i_term[kz] = np.linalg.inv(
            #     (dx / self.dt) * self.l[kz] + (dx / 2) * self.r[kz]
            # ).dot((dx / self.dt) * self.l[kz] - (dx / 2) * self.r[kz])

        # kz = self.x.shape[0] - 1
        # self.v_term[kz] = np.linalg.inv(
        #     (dx / self.dt) * self.c[kz] + (dx / 2) * self.g[kz]
        # ).dot((dx / self.dt) * self.c[kz] - (dx / 2) * self.g[kz])


        # self.i_diff = np.linalg.inv((dx / self.dt) * self.c + (dx / 2) * self.g)
        # self.v_diff = np.linalg.inv((dx / self.dt) * self.l + (dx / 2) * self.r)
        self.__update_lr_terms()
        self.__update_cg_terms()

        self.left_port_term_1 = np.matmul(
            np.linalg.inv(
                dx * (np.matmul(self.zs, self.c[0])) / self.dt
                + dx * (np.matmul(self.zs, self.g[0])) / 2
                + np.eye(self.number_of_conductors)
            ),
            dx * (np.matmul(self.zs, self.c[0])) / self.dt
            - dx * (np.matmul(self.zs, self.g[0])) / 2
            - np.eye(self.number_of_conductors),
        )

        self.left_port_term_2 = np.linalg.inv(
            dx * (np.matmul(self.zs, self.c[0])) / self.dt
            + dx * (np.matmul(self.zs, self.g[0])) / 2
            + np.eye(self.number_of_conductors)
        )

        self.right_port_term_1 = np.matmul(
            np.linalg.inv(
                dx * (np.matmul(self.zl, self.c[-1])) / self.dt
                + dx * (np.matmul(self.zl, self.g[-1])) / 2
                + np.eye(self.number_of_conductors)
            ),
            dx * (np.matmul(self.zl, self.c[-1])) / self.dt
            - dx * (np.matmul(self.zl, self.g[-1])) / 2
            - np.eye(self.number_of_conductors),
        )

        self.right_port_term_2 = np.linalg.inv(
            dx * (np.matmul(self.zl, self.c[-1])) / self.dt
            + dx * (np.matmul(self.zl, self.g[-1])) / 2
            + np.eye(self.number_of_conductors)
        )

    def __update_lr_terms(self):
        for kz in range(self.x.shape[0] - 1):
            F1 = (
                (self.dx / self.dt) * self.l[kz]
                + (self.dx / 2) * self.d[kz]
                + (self.dx / self.dt) * self.e[kz]
                + (self.dx / 2) * self.r[kz]
                + self.dx * self.q1sum[kz]
            )
            F2 = (
                (self.dx / self.dt) * self.l[kz]
                - (self.dx / 2) * self.d[kz]
                + (self.dx / self.dt) * self.e[kz]
                - (self.dx / 2) * self.r[kz]
                - self.dx * self.q2sum[kz]
            )
            self.i_term[kz] = np.linalg.inv(F1).dot(F2)
            self.v_diff[kz] = np.linalg.inv(F1)
            
        # F1 = (
        #     (self.dx / self.dt) * self.l
        #     + (self.dx / 2) * self.d
        #     + (self.dx / self.dt) * self.e
        #     + (self.dx / 2) * self.r
        #     + self.dx * self.q1sum
        # )
        # F2 = (
        #     (self.dx / self.dt) * self.l
        #     - (self.dx / 2) * self.d
        #     + (self.dx / self.dt) * self.e
        #     - (self.dx / 2) * self.r
        #     - self.dx * self.q2sum
        # )
        
        # self.i_term = np.einsum('...ij,...ji->...ij' , np.linalg.inv(F1), F2)
        # self.v_diff = np.linalg.inv(F1)

        # 

        # for kz in range(self.x.shape[0] - 1):
        #     self.i_term[kz] = np.linalg.inv(
        #         (self.dx / self.dt) * self.l[kz] + (self.dx / 2) * self.r[kz]
        #     ).dot((self.dx / self.dt) * self.l[kz] - (self.dx / 2) * self.r[kz])

        # self.v_diff = np.linalg.inv((self.dx / self.dt) * self.l + (self.dx / 2) * self.r)

    def __update_cg_terms(self):
        # F1 = (self.dx / self.dt) * self.c + (self.dx / 2) * self.g
        # F2 = (self.dx / self.dt) * self.c - (self.dx / 2) * self.g
        # self.v_term = np.einsum('...ij,...ji->...ij' , np.linalg.inv(F1), F2)
        # self.i_diff = np.linalg.inv(F1)

        for kz in range(self.x.shape[0]):
            self.v_term[kz] = np.linalg.inv(
                (self.dx / self.dt) * self.c[kz] + (self.dx / 2) * self.g[kz]
            ).dot((self.dx / self.dt) * self.c[kz] - (self.dx / 2) * self.g[kz])

        self.i_diff = np.linalg.inv((self.dx / self.dt) * self.c + (self.dx / 2) * self.g)



    def add_resistance_at_point(self, position, conductor, resistance):
        self.add_resistance_in_region(position, position, conductor, resistance)
        
    def add_resistance_in_region(self, begin, end, conductor, resistance):
        assert(end >= begin)
        index1 = np.argmin(np.abs(self.x - begin))
        index2 = np.argmin(np.abs(self.x - end))
        if (index1 != index2):
            self.r[index1:index2+1][conductor][conductor] = resistance/(self.x[index2]-self.x[index1])
        else:
            self.r[index1:index2+1][conductor][conductor] = resistance/self.dx
        self.__update_lr_terms()

    def add_conductance_at_point(self, position, conductor, conductance):
        self.add_conductance_in_region(position, position, conductor, conductance)
        
    def add_conductance_in_region(self, begin, end, conductor, conductance):
        assert(end >= begin)
        index1 = np.argmin(np.abs(self.x - begin))
        index2 = np.argmin(np.abs(self.x - end))
        if (index1 != index2):
            self.g[index1:index2+1][conductor][conductor] = conductance/(self.x[index2]-self.x[index1])
        else:
            self.g[index1:index2+1][conductor][conductor] = conductance/self.dx
        self.__update_cg_terms()



    def v_sum(self,arr:np.ndarray): 
        return np.vectorize(np.sum)(arr)

    def add_dispersive_connector(
        self,
        position,
        conductor,
        d: float,
        e: float,
        poles: np.ndarray,
        residues: np.ndarray,
    ):
        # check complex poles are conjugate

        if (position > self.x[-1]) or (position < 0.0):
            raise ValueError("Connector position is out of MTL length.")

        index = np.argmin(np.abs(self.x - position))

        self.d[index, conductor, conductor] = d
        self.e[index, conductor, conductor] = e


        self.q1[index, conductor, conductor] = -(residues / poles) * (
            1 - (np.exp(poles * self.dt) - 1) / (poles * self.dt)
        )
        self.q2[index, conductor, conductor] = (residues / poles) * (
            1 / (poles * self.dt)
            + np.exp(poles * self.dt) * (1 - 1 / (poles * self.dt))
        )
        self.q3[index, conductor, conductor] = np.exp(poles * self.dt)

        self.q1sum = self.v_sum(self.q1)
        self.q2sum = self.v_sum(self.q2)

        # for kz in range(self.x.shape[0] - 1):
        #     F1 = (
        #         (self.dx / self.dt) * self.l[kz]
        #         + (self.dx / 2) * self.d[kz]
        #         + (self.dx / self.dt) * self.e[kz]
        #         + (self.dx / 2) * self.r[kz]
        #         + self.dx * self.q1sum[kz]
        #     )
        #     F2 = (
        #         (self.dx / self.dt) * self.l[kz]
        #         - (self.dx / 2) * self.d[kz]
        #         + (self.dx / self.dt) * self.e[kz]
        #         - (self.dx / 2) * self.r[kz]
        #         - self.dx * self.q2sum[kz]
        #     )
        #     self.i_term[kz] = np.linalg.inv(F1).dot(F2)
        #     self.v_diff[kz] = np.linalg.inv(F1)
        self.__update_lr_terms()

    def __update_q3_phi_term(self):
        for kz in range(0, self.i.shape[1]):
            self.q3_phi_term[kz] = self.v_sum(self.q3[kz].dot(self.phi[kz]))
        
    def __update_phi(self, i_prev, i_now):
        for kz in range(0, self.i.shape[1]):
            self.phi[kz, :] = (
                self.q1[kz, :, :].dot(i_now[:, kz])+\
                self.q2[kz, :, :].dot(i_prev[:, kz])+\
                self.q3[kz, :,:].dot(self.phi[kz, :])
            )


    def step(self):
        self.update_sources()

        self.v[:, 0] = self.left_port_term_1.dot(
            self.v[:, 0]
        ) + self.left_port_term_2.dot(
            -2 * np.matmul(self.zs, self.i[:, 0])
            + (self.v_sources_now[:, 0] + self.v_sources_prev[:, 0])
            - (self.dx / self.dt)
            * (self.zs.dot(self.c[0])).dot(self.e_T_now[:, 0] - self.e_T_prev[:, 0])
        )

        self.v[:, 1:-1] = np.einsum('...ij,...j->...i',self.v_term[1:-1,:,:],self.v.T[1:-1,:]).T-\
                          np.einsum('...ij,...j->...i',self.i_diff[1:-1,:,:],(self.i[:,1:]-self.i[:,:-1]).T).T-\
                          (self.e_T_now[:, 1:-1] - self.e_T_prev[:, 1:-1])


        self.v[:, -1] = self.right_port_term_1.dot(
            self.v[:, -1]
        ) + self.right_port_term_2.dot(
            +2 * np.matmul(self.zl, self.i[:, -1])
            + (self.v_sources_now[:, -1] + self.v_sources_prev[:, -1])
            - (self.dx / self.dt)
            * (self.zl.dot(self.c[-1])).dot(self.e_T_now[:, -1] - self.e_T_prev[:, -1])
        )

        self.__update_q3_phi_term()
        i_prev = self.i
        self.i[:, :] = np.einsum('...ij,...j->...i',self.i_term[:,:,:],self.i.T[:,:]).T-\
                       np.einsum('...ij,...j->...i',self.v_diff[:,:,:],(self.v[:, 1:] - self.v[:, :-1]
                                            + (self.e_T_now[:, 1:] - self.e_T_now[:, :-1])
                                            - (self.dx / 2) * (self.e_L_now[:, :] + self.e_L_prev[:, :])).T).T-\
                       np.einsum('...ij,...j->...i',self.v_diff[:,:,:],self.dx * self.q3_phi_term[:,:]).T

        

        i_now = self.i
        self.__update_phi(i_prev, i_now)

        self.time += self.dt
        self.update_probes()
