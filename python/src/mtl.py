import numpy as np
import numpy.typing as npt

from copy import deepcopy

import sympy as sp
import scipy.linalg as linalg


from types import FunctionType
from types import LambdaType

from multipledispatch import dispatch

from collections import OrderedDict
from typing import Dict
# from typing_extensions import TypeVarTuple

from .probes import *
from .dispersive import TransferImpedance, DispersiveConnector
from .dispersive import Dispersive
from .utils import add_t_functions as add, multiply
from .utils import point_in_line

class Field:
    def __init__(self, incident_x: sp.Function, incident_y: sp.Function, incident_z: sp.Function):
    # def __init__(self, incident_x: FunctionType, incident_y: FunctionType, incident_z: FunctionType):
        # assert (type(incident_x) == LambdaType and type(incident_z) == LambdaType)
        self.e_x = incident_x
        self.e_y = incident_y
        self.e_z = incident_z

    def compute_eL_at_segment(self, du: npt.NDArray[np.float64]):
    # def compute_eL_at_segment(self, du: np.array):
        du = du/np.linalg.norm(du)
        assert (np.isclose(np.linalg.norm(du), 1, rtol = 0.01))
        return self.e_x*du[0] + self.e_y*du[1] + self.e_z*du[2]

class PlaneWave(Field):
    def __init__(self, field :  FunctionType, polar, azimuth, polarization):
        
        self.field = field
        self.polar = polar 
        self.azimuth = azimuth
        self.polarization = polarization
        
        self.compute_directions(polar, azimuth, polarization)
        self.compute_directions(polar, azimuth, polarization)

        # self.compute_velocities(polar, azimuth, velocity)
        
    def compute_directions(self, p, z, d):
        self.ex =  np.sin(d) * np.sin(p)
        self.ey = -np.sin(d) * np.cos(p) * np.cos(z) - np.cos(d) * np.sin(z)
        self.ez = -np.sin(d) * np.cos(p) * np.sin(z) + np.cos(d) * np.cos(z)

    def compute_velocities(self, v):
        self.vx = -v/np.cos(self.polar)
        self.vy = -v/(np.sin(self.polar)*np.cos(self.azimuth))
        self.vz = -v/(np.sin(self.polar)*np.sin(self.azimuth))
        
    def rotate_to_segment(self, u):

        thetax = np.arctan(u[1]/u[2])
        thetay = np.arctan(-u[0]/(u[1]**2+u[2]**2)**0.5)

        def rx(theta):
            return np.array([[1,0,0],[0, np.cos(theta), -np.sin(theta)],[0, np.sin(theta), np.cos(theta)]])
        def ry(theta):
            return np.array([[np.cos(theta), 0, np.sin(theta)],[0, 1, 0],[-np.sin(theta),0,np.cos(theta)]])
        
        F = np.array([np.cos(self.polar), np.sin(self.polar)*np.cos(self.azimuth), np.sin(self.azimuth)*np.sin(self.polar)])
        F_rot = ry(thetay).dot(rx(thetax)).dot(F)
        polar_rot = np.arccos(F_rot[0]/np.sqrt(F_rot[0]**2+F_rot[1]**2+F_rot[2]**2))
        if (F_rot[1] == 0 and F_rot[2] == 0):
            azimuth_rot = 0
        else:
            azimuth_rot = np.sign(F_rot[2])*np.arccos(F_rot[1]/np.sqrt(F_rot[1]**2+F_rot[2]**2))
        
        
        return PlaneWave(self.field, polar_rot, azimuth_rot, self.polarization)
        


class MTL():
    """
    Multiconductor Transmission Line
    """
    def __init__(self, l, c, r = 0.0, g = 0.0, length=1.0, node_positions = np.array([]), ndiv=[100], Zs=0.0, Zl=0.0, name = ""):
                 
        if (node_positions.size == 0):
            node_positions = np.ndarray(shape=(2,3), dtype = float)
            node_positions[0] = np.array([0.0,0.0,0.0])
            node_positions[1] = np.array([0.0,0.0,length])
        if (type(ndiv) == int):
            ndiv = np.array([ndiv])

        self.name = name

        self.u = np.empty(shape=(np.sum(ndiv) + 1, 3))
        self.du = np.empty(shape=(np.sum(ndiv) , 3))

        self.init_LC(l, c)
        self.number_of_conductors = self.l.shape[1]
        self.init_RG(r, g)
        self.init_Zt(Zs, Zl)

        self.du_norm = np.zeros(shape=(np.sum(ndiv),
                                       self.number_of_conductors,
                                       self.number_of_conductors))


        assert (node_positions.shape[0] == len(ndiv) + 1)
        
        for i, divisions in enumerate(ndiv):
            du = (node_positions[i+1]-node_positions[i])/divisions
            for k in range(divisions):
                self.u[i*divisions + k]  = node_positions[i]+k*du
                self.du[i*divisions + k] = du
                self.du_norm[i*divisions + k] = np.linalg.norm(du)*np.eye(self.number_of_conductors)
        self.u[-1] = node_positions[-1]

        self.time = 0.0
        self.dt = self.get_max_timestep()

        self.v = np.zeros([self.number_of_conductors, self.u.shape[0]])
        self.i = np.zeros([self.number_of_conductors, self.u.shape[0] - 1])

        self.probes = []
        self.port_probes = []
        
        self.connectors = DispersiveConnector(self.number_of_conductors, self.i.shape[1], self.u, self.dt)

        self.v_sources = np.empty(
            shape=(self.number_of_conductors, self.u.shape[0]), dtype=object
        )
        self.v_sources.fill(lambda t : 0)


        self.v_pw_term = np.empty(
            shape=(self.number_of_conductors, self.u.shape[0]), dtype=object
        )
        self.i_pw_term = np.empty(
            shape=(self.number_of_conductors, self.u.shape[0] - 1), dtype=object
        )
        self.v_pw_term.fill(lambda t : 0)
        self.i_pw_term.fill(lambda t : 0)

        self.e_L = np.empty(
            shape=(self.number_of_conductors, self.u.shape[0] - 1), dtype=object
        )
        self.e_T = np.empty(
            shape=(self.number_of_conductors, self.u.shape[0]), dtype=object
        )
        self.e_L.fill(lambda t : 0)
        self.e_T.fill(lambda t : 0)

        self.v_term = np.empty(
            shape=(
                self.u.shape[0],
                self.number_of_conductors,
                self.number_of_conductors,
            )
        )
        self.i_term = np.empty(
            shape=(
                self.u.shape[0] - 1,
                self.number_of_conductors,
                self.number_of_conductors,
            )
        )
    
        self.compute_terminal_terms()

    def init_LC(self, l, c):
        types = [float, np.float64]
        if type(l) in types and type(c) in types:
            self.l = np.empty(shape=(self.u.shape[0] - 1, 1, 1))
            self.c = np.empty(shape=(self.u.shape[0], 1, 1))
            self.l[:] = l
            self.c[:] = c
        elif type(l) == np.ndarray and type(c) == np.ndarray:
            assert l.shape == c.shape
            assert l.shape[0] == l.shape[1]
            n = l.shape[0]
            self.l = np.empty(shape=(self.u.shape[0] - 1, n, n))
            self.c = np.empty(shape=(self.u.shape[0], n, n))
            self.l[:] = l
            self.c[:] = c
        else:
            raise ValueError("Invalid input L and/or C")

    def init_RG(self, r, g):
        types = [float, np.float64]
        if type(g) in types:
            if (g != 0.0):
                assert(self.number_of_conductors == 1)
            self.g = np.empty(shape=(self.u.shape[0], self.number_of_conductors, self.number_of_conductors))
            self.g[:] = g
        elif type(g) == np.ndarray:
            assert g.shape[0] == self.l.shape[1]
            assert g.shape[0] == g.shape[1]
            n = g.shape[0]
            self.g = np.empty(shape=(self.u.shape[0], self.number_of_conductors, self.number_of_conductors))
            self.g[:] = g
        else:
            raise ValueError("Invalid input G")

        if type(r) in types:
            if (r != 0.0):
                assert(self.number_of_conductors == 1)
            self.r = np.empty(shape=(self.u.shape[0]-1, self.number_of_conductors, self.number_of_conductors))
            self.r[:] = r
        elif type(r) == np.ndarray:
            assert r.shape[0] == self.l.shape[1]
            assert r.shape[0] == r.shape[1]
            self.r = np.empty(shape=(self.u.shape[0]-1, self.number_of_conductors, self.number_of_conductors))
            self.r[:] = r
        else:
            raise ValueError("Invalid input R")


    def init_Zt(self, Zs, Zl):
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
        
        
    def compute_terminal_terms(self):
        self.left_port_term_1 = np.matmul(
            np.linalg.inv(
                self.du_norm[0][0][0] * (np.matmul(self.zs, self.c[0])) / self.dt
                + self.du_norm[0][0][0] * (np.matmul(self.zs, self.g[0])) / 2
                + np.eye(self.number_of_conductors)
            ),
            self.du_norm[0][0][0] * (np.matmul(self.zs, self.c[0])) / self.dt
            - self.du_norm[0][0][0] * (np.matmul(self.zs, self.g[0])) / 2
            - np.eye(self.number_of_conductors),
        )

        self.left_port_term_2 = np.linalg.inv(
            self.du_norm[0][0][0] * (np.matmul(self.zs, self.c[0])) / self.dt
            + self.du_norm[0][0][0] * (np.matmul(self.zs, self.g[0])) / 2
            + np.eye(self.number_of_conductors)
        )

        self.right_port_term_1 = np.matmul(
            np.linalg.inv(
                self.du_norm[-1][0][0] * (np.matmul(self.zl, self.c[-1])) / self.dt
                + self.du_norm[-1][0][0] * (np.matmul(self.zl, self.g[-1])) / 2
                + np.eye(self.number_of_conductors)
            ),
            self.du_norm[-1][0][0] * (np.matmul(self.zl, self.c[-1])) / self.dt
            - self.du_norm[-1][0][0] * (np.matmul(self.zl, self.g[-1])) / 2
            - np.eye(self.number_of_conductors),
        )

        self.right_port_term_2 = np.linalg.inv(
            self.du_norm[-1][0][0] * (np.matmul(self.zl, self.c[-1])) / self.dt
            + self.du_norm[-1][0][0] * (np.matmul(self.zl, self.g[-1])) / 2
            + np.eye(self.number_of_conductors)
        )
        

    def get_phase_velocities(self):
        return np.array([
            1/np.sqrt(np.real(linalg.eigvals(self.l[k].dot(self.c[k+1]))))
            for k in range(self.u.shape[0] - 1)
        ])

    def get_max_timestep(self):
        return np.min(self.du_norm[np.nonzero(self.du_norm)]) / np.max(self.get_phase_velocities())

    def set_time_step(self, NDT, final_time):
        self.dt = final_time / NDT

    def get_time_range(self, final_time):
        return np.arange(0, np.floor(final_time / self.dt))

    def set_voltage(self, conductor, voltage):
        self.v[conductor] = voltage(self.z)

    def update_probes(self):
        for p in self.probes:
            p.update(self.time, self.v, self.i)

    @dispatch()
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

    @dispatch(float, float)
    def update_sources(self, time, dt):
        self.v_sources_now = np.vectorize(FunctionType.__call__, otypes=["float64"])(
            self.v_sources, time
        )
        self.v_sources_prev = np.vectorize(FunctionType.__call__, otypes=["float64"])(
            self.v_sources, time - dt
        )

        self.e_L_now = np.vectorize(FunctionType.__call__, otypes=["float64"])(
            self.e_L, time + dt / 2
        )
        self.e_L_prev = np.vectorize(FunctionType.__call__, otypes=["float64"])(
            self.e_L, time - dt / 2
        )
        self.e_T_now = np.vectorize(FunctionType.__call__, otypes=["float64"])(
            self.e_T, time
        )
        self.e_T_prev = np.vectorize(FunctionType.__call__, otypes=["float64"])(
            self.e_T, time - dt
        )

    def update_lr_terms(self):
        
        F1 = np.einsum('...ij,...jk->...ik' , self.du_norm, (self.l/self.dt + 0.5*self.connectors.d + self.connectors.e/self.dt + 0.5*self.r + self.connectors.q1sum))
        F2 = np.einsum('...ij,...jk->...ik' , self.du_norm, (self.l/self.dt - 0.5*self.connectors.d + self.connectors.e/self.dt - 0.5*self.r - self.connectors.q2sum))
        
        IF1 = np.linalg.inv(F1)
        self.i_term = np.einsum('...ij,...jk->...ik' , IF1, F2)
        self.v_diff = IF1

    def update_cg_terms(self):
        
        du_norm = np.zeros(shape=(self.du_norm.shape[0]+1, self.number_of_conductors, self.number_of_conductors))
        du_norm[0] = self.du_norm[0]
        for i in range(1, self.du_norm.shape[0]):
            du_norm[i]= 0.5*(self.du_norm[i]+self.du_norm[i-1])
        du_norm[-1] = self.du_norm[-1]
        
        F1 = np.einsum('...ij,...jk->...ik' , du_norm, self.c/self.dt + self.g/2)
        F2 = np.einsum('...ij,...jk->...ik' , du_norm, self.c/self.dt - self.g/2)
        IF1 = np.linalg.inv(F1)
        self.v_term = np.einsum('...ij,...jk->...ik' , IF1, F2)
        self.i_diff = IF1


    def set_resistance_at_point(self, position, conductor, resistance):
        self.set_resistance_in_region(position, position, conductor, resistance)
        
    def set_resistance_in_region(self, start, end, conductor, resistance):
        assert(end >= start)
        if (type(start) == float):
            start = np.array([0.0,0.0,start])
        if (type(end) == float):
            end = np.array([0.0,0.0,end])

        index1 = np.argmin(np.apply_along_axis(np.linalg.norm, 1, np.abs(self.u - start)))
        index2 = np.argmin(np.apply_along_axis(np.linalg.norm, 1, np.abs(self.u - end)))
        
        if (index1 == self.r.shape[0]):
            index1 -= 1
        if (index2 == self.r.shape[0]):
            index2 -= 1
        
        if (index1 != index2):
            self.r[index1:index2+1][conductor][conductor] = resistance/np.linalg.norm(self.u[index2]-self.u[index1])
        else:
            self.r[index1:index2+1][conductor][conductor] = resistance/np.linalg.norm(self.du[index1])
            
        self.update_lr_terms()

    def add_conductance_at_point(self, position, conductor, conductance):
        self.add_conductance_in_region(position, position, conductor, conductance)
        
    def add_conductance_in_region(self, begin, end, conductor, conductance):
        assert(end >= begin)

        if (type(begin) == float):
            begin = np.array([0.0,0.0,begin])
        if (type(end) == float):
            end = np.array([0.0,0.0,end])

        index1 = np.argmin(np.abs(self.u - begin))
        index2 = np.argmin(np.abs(self.u - end))
        if (index1 != index2):
            self.g[index1:index2+1][conductor][conductor] = conductance/np.linalg.norm(self.u[index2]-self.u[index1])
        else:
            self.g[index1:index2+1][conductor][conductor] = conductance/np.linalg.norm(self.du[index1])
            
        self.update_cg_terms()


    def advance_voltage(self):
        self.v[:, 1:-1] = np.einsum('...ij,...j->...i',self.v_term[1:-1,:,:],self.v.T[1:-1,:]).T-\
                          np.einsum('...ij,...j->...i',self.i_diff[1:-1,:,:],(self.i[:,1:]-self.i[:,:-1]).T).T-\
                          (self.e_T_now[:, 1:-1] - self.e_T_prev[:, 1:-1])
      
        
    def advance_current(self):
        self.connectors.update_q3_phi_term()
        i_prev = self.i
        self.i[:, :] = np.einsum('...ij,...j->...i',self.i_term[:,:,:],self.i.T[:,:]).T-\
                       np.einsum('...ij,...j->...i',self.v_diff[:,:,:],(self.v[:, 1:] - self.v[:, :-1]
                                            + (self.e_T_now[:, 1:] - self.e_T_now[:, :-1])
                                            - (np.einsum('...ij,...j->...i', self.du_norm / 2, (self.e_L_now[:, :] + self.e_L_prev[:, :]).T).T)).T).T-\
                       np.einsum('...ij,...j->...i',self.v_diff[:,:,:],np.einsum('...ij,...j->...i',self.du_norm,self.connectors.q3_phi_term[:,:])).T

        i_now = self.i
        self.connectors.update_phi(i_prev, i_now)

    def add_voltage_source(self, position: np.ndarray, conductor: int, magnitude):
        if (type(position) == float):
            position = np.array([0.0,0.0,position])

        distances = np.apply_along_axis(np.linalg.norm, 1, np.abs(self.u - position))
        index = np.argmin(distances)
    
        # index = np.argmin(np.abs(self.u - position))
        self.v_sources[conductor, index] = magnitude

    # def add_planewave(self, planewave: PlaneWave, distances_x : np. ndarray, distances_y : np.ndarray):
        
    #     local_pw = planewave.rotate_to_segment(self.u)
    #     local_pw.compute_velocities(np.max(self.get_phase_velocities()))
    #     local_pw.aT = local_pw.ex*distances_x + local_pw.ey*distances_y
    #     local_pw.aL = local_pw.ez*(distances_x/local_pw.vx+distances_y/local_pw.vy)

    #     for k in range(self.z.size-1):
    #         self.v_pw_term[:,k] = lambda t : local_pw.field(t - self.z[k] / local_pw.vz)
    #         self.i_pw_term[:,k] = lambda t : local_pw.field(t - (self.z[k] + 0.5*self.dz)/local_pw.vz)
    #     pos = self.z[self.z.size - 1]
    #     self.v_pw_term[:,self.z.size - 1] = lambda t : local_pw.field(t - pos / local_pw.vz)

    
    def add_planewave(self, pw: PlaneWave, distances : np. ndarray):
        pw.compute_velocities(np.max(self.get_phase_velocities()))
        x, y, z, t = sp.symbols('x y z t')
        e_0 = sp.Function('e_0')
        # e_0 = pw.field(t + x/pw.vx + y/pw.vy + z/pw.vz) 
        e_0 = pw.field(t - x/pw.vx- y/pw.vy- z/pw.vz) 
        pw.e_x = e_0 * pw.ex
        pw.e_y = e_0 * pw.ey
        pw.e_z = e_0 * pw.ez
        self.add_external_field(pw, distances)
    
    def add_external_field(self, field : Field, distances: npt.NDArray[np.float64], field_localization: list[npt.NDArray[np.float64]] = []):
    # def add_external_field(self, field : Field, distances: np.array):

        assert(distances.shape == (self.number_of_conductors, self.u.shape[0], 3))
        x, y, z, t, v = sp.symbols('x y z t v')
         
        indices = range(self.u.shape[0] - 1)
        if len(field_localization) != 0:
            start_index = np.argmin(np.apply_along_axis(np.linalg.norm, 1, np.abs(self.u - field_localization[0])))
            end_index =   np.argmin(np.apply_along_axis(np.linalg.norm, 1, np.abs(self.u - field_localization[1])))
            indices = range(start_index, end_index)
         
         
        vu = self.get_phase_velocities()
        for n in range(self.number_of_conductors):
            for nz in indices:
              
                eL = field.compute_eL_at_segment(self.du[nz])
                x0, y0, z0 = self.u[nz,:]
                xi, yi, zi = distances[n, nz, :]
                l = np.abs(np.array([xi,yi,zi]))
                l /=np.linalg.norm(l)
                self.e_L[n, nz] = sp.lambdify(t, eL.subs(x, x0 + xi - 0.5*self.du[nz,0]).subs(y, y0 + yi- 0.5*self.du[nz,1]).subs(z, z0 + zi- 0.5*self.du[nz,2]).subs(v,np.max(vu[nz])) -\
                                                 eL.subs(x,x0- 0.5*self.du[nz,0]).subs(y, y0- 0.5*self.du[nz,1]).subs(z, z0- 0.5*self.du[nz,2]).subs(v,np.max(vu[nz])))

                self.e_T[n,nz] = sp.lambdify(t,
                        l[0]*sp.Integral(field.e_x.subs(y, y0).subs(z, z0).subs(v, np.max(vu[nz])), (x, x0, x0 + xi))+\
                        l[1]*sp.Integral(field.e_y.subs(x, x0).subs(z, z0).subs(v, np.max(vu[nz])), (y, y0, y0 + yi))+\
                        l[2]*sp.Integral(field.e_z.subs(x, x0).subs(y, y0).subs(v, np.max(vu[nz])), (z, z0, z0 + zi)))
            xi, yi, zi = distances[n, -1, :]
            l = np.abs(np.array([xi,yi,zi]))
            l /=np.linalg.norm(l)
            self.e_T[n,-1] = sp.lambdify(t,
                    l[0]*sp.Integral(field.e_x.subs(y, y0).subs(z, z0).subs(v, np.max(vu[-1])), (x, x0, x0 + xi))+\
                    l[1]*sp.Integral(field.e_y.subs(x, x0).subs(z, z0).subs(v, np.max(vu[-1])), (y, y0, y0 + yi))+\
                    l[2]*sp.Integral(field.e_z.subs(x, x0).subs(y, y0).subs(v, np.max(vu[-1])), (z, z0, z0 + zi)))


    def add_probe(self, position: np.ndarray, probe_type: str):
        if (type(position) == float):
            position = np.array([0.0,0.0,position])
        
        if (np.any(position > self.u[-1])) or np.any((position < self.u[0])):
            raise ValueError("Probe position is out of MTL length.")

        probe = Probe(position, probe_type, self.dt, self.u)
        self.probes.append(probe)
        return probe

    def add_port_probe(self, terminal):
        if terminal == 0:
            x0 = self.z[0]
            x1 = self.z[1]
            z0 = self.zs
        if terminal == 1:
            x0 = self.z[-2]
            x1 = self.z[-1]
            z0 = self.zl

        v0 = self.add_probe(position=x0,          probe_type='voltage')
        v1 = self.add_probe(position=x1,          probe_type='voltage')
        i0 = self.add_probe(position=(x0+x1)/2.0, probe_type='current')

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

    def add_dispersive_connector(
        self,        
        position : np.ndarray,
        conductor,
        d: float,
        e: float,
        poles: np.ndarray,
        residues: np.ndarray,
    ):
        self.connectors.add_dispersive_connector(position, conductor, d, e, poles, residues)


    def advance_left_terminal(self):
        self.v[:, 0] = self.left_port_term_1.dot(
            self.v[:, 0]
        ) + self.left_port_term_2.dot(
            -2 * np.matmul(self.zs, self.i[:, 0])
            + (self.v_sources_now[:, 0] + self.v_sources_prev[:, 0])
            - (self.du_norm[0,0,0] / self.dt)
            * (self.zs.dot(self.c[0])).dot(self.e_T_now[:, 0] - self.e_T_prev[:, 0])
        )


    def advance_right_terminal(self):
        self.v[:, -1] = self.right_port_term_1.dot(
            self.v[:, -1]
        ) + self.right_port_term_2.dot(
            +2 * np.matmul(self.zl, self.i[:, -1])
            + (self.v_sources_now[:, -1] + self.v_sources_prev[:, -1])
            - (self.du_norm[-1,0,0] / self.dt)
            * (self.zl.dot(self.c[-1])).dot(self.e_T_now[:, -1] - self.e_T_prev[:, -1])
        )


    def step(self):
        self.update_sources()

        self.advance_left_terminal()
        self.advance_voltage()
        self.advance_right_terminal()
        self.advance_current()
        
        self.time += self.dt
        self.update_probes()


    def run_until(self, finalTime):
        t = self.get_time_range(finalTime)
        self.update_lr_terms()
        self.update_cg_terms()

        for p in self.probes:
            p.resize_frames(len(t), self.number_of_conductors)

        for _ in t:
            self.step()


class MTLD:
    """
    Lossless Multiconductor Transmission Line Network with subdomains
    """
    def __init__(self, levels : Dict[int, np.ndarray], name = ""):

        assert(len(levels[0]) == 1)
        
        self.name = name
        self.ndiv = levels[0][0].u.shape[0]

        self.u = levels[0][0].u
        self.du = levels[0][0].du

        self.dt = 1e10
        self.time = 0.0
        self.levels = levels
        #with levels as input, the v and I arrays can be built
        #check that nz is the same for all
        self.number_of_conductors = 0
        self.probes : list[Probe] = []
        
        self.conductors_in_level = np.array([], dtype=int)
        

        
        q1 = (self.ndiv-1)*[np.ndarray(shape=(0, 0))]
        q2 = (self.ndiv-1)*[np.ndarray(shape=(0, 0))]
        q3 = (self.ndiv-1)*[np.ndarray(shape=(0, 0))]
        d  = (self.ndiv-1)*[np.ndarray(shape=(0, 0))]
        e  = (self.ndiv-1)*[np.ndarray(shape=(0, 0))]
        
        L = (self.ndiv-1)*[np.ndarray(shape=(0, 0))]
        C = (self.ndiv)*[np.ndarray(shape=(0, 0))]

        R = (self.ndiv-1)*[np.ndarray(shape=(0, 0))]
        G = (self.ndiv)*[np.ndarray(shape=(0, 0))]

        for level, mtls in levels.items():
            conductors = 0
            for line in mtls:
                assert(type(line) == MTL)
                assert(np.array_equal(self.u,line.u))
                # assert(self.nz == line.x.shape[0])
            
                conductors += line.l.shape[1]
                self.number_of_conductors += line.l.shape[1]
                # self.number_of_conductors += conductors

                if (line.dt < self.dt):
                    self.dt = line.dt
                #pul non-hom l c matrices
                for k in range(self.ndiv-1):
                    L[k] = linalg.block_diag(L[k], line.l[k])
                    C[k] = linalg.block_diag(C[k], line.c[k])
                    R[k] = linalg.block_diag(R[k], line.r[k])
                    G[k] = linalg.block_diag(G[k], line.g[k])
                    
                    q1[k]  = linalg.block_diag(q1[k], line.connectors.q1[k])
                    q2[k]  = linalg.block_diag(q2[k], line.connectors.q2[k])
                    q3[k]  = linalg.block_diag(q3[k], line.connectors.q3[k])
                    d[k]   = linalg.block_diag(d[k], line.connectors.d[k])
                    e[k]   = linalg.block_diag(e[k], line.connectors.e[k])
                    
                C[self.ndiv-1] = linalg.block_diag(C[self.ndiv-1], line.c[self.ndiv-1])
                G[self.ndiv-1] = linalg.block_diag(G[self.ndiv-1], line.g[self.ndiv-1])
            
            self.conductors_in_level = np.append(self.conductors_in_level, conductors)
            # self.levels[level] = {"mtl" :mtls, "conductors":conductors}
            
        
        self.du_norm = levels[0][0].du_norm[:,0,0].reshape(self.ndiv-1,1,1) * np.eye(self.number_of_conductors)

            
        self.l : npt.NDArray[np.float64]= np.ndarray(shape=(self.ndiv-1, self.number_of_conductors, self.number_of_conductors))
        self.c : npt.NDArray[np.float64]= np.ndarray(shape=(self.ndiv, self.number_of_conductors, self.number_of_conductors))
        self.l[0:self.ndiv-1] = L[0:self.ndiv-1]
        self.c[0:self.ndiv] = C[0:self.ndiv]

        self.r : npt.NDArray[np.float64]= np.ndarray(shape=(self.ndiv-1, self.number_of_conductors, self.number_of_conductors))
        self.g : npt.NDArray[np.float64]= np.ndarray(shape=(self.ndiv, self.number_of_conductors, self.number_of_conductors))
        self.r[0:self.ndiv-1] = R[0:self.ndiv-1]
        self.g[0:self.ndiv] = G[0:self.ndiv]

        self.transfer_impedance = TransferImpedance(self.number_of_conductors, self.ndiv - 1, self.u, self.dt)
        self.transfer_impedance.q1[0:self.ndiv-1] = q1[0:self.ndiv-1] 
        self.transfer_impedance.q2[0:self.ndiv-1] = q2[0:self.ndiv-1] 
        self.transfer_impedance.q3[0:self.ndiv-1] = q3[0:self.ndiv-1] 
        self.transfer_impedance.d[0:self.ndiv-1]  = d[0:self.ndiv-1] 
        self.transfer_impedance.e[0:self.ndiv-1]  = e[0:self.ndiv-1] 
        # self.dx = self.x[1] - self.x[0]

        self.v = np.zeros([self.number_of_conductors, self.ndiv])
        self.i = np.zeros([self.number_of_conductors, self.ndiv-1])

        self.e_L = np.empty(
            shape=(self.number_of_conductors, self.ndiv - 1), dtype=object
        )
        self.e_T = np.empty(
            shape=(self.number_of_conductors, self.ndiv), dtype=object
        )
        self.e_L.fill(lambda t : 0)
        self.e_T.fill(lambda t : 0)

        self.v_sources = np.empty(
            shape=(self.number_of_conductors, self.ndiv - 1), dtype=object
        )
        self.v_sources.fill(lambda t : 0)


    def add_localized_longitudinal_field(self, start: np.ndarray, end: np.ndarray, conductor: int, magnitude):
        if (type(start) == float):
            start = np.array([0.0,0.0,start])
        if (type(end) == float):
            end = np.array([0.0,0.0,end])

        for position in [start,end]:
            if (np.any(position > self.u[-1])) or np.any((position < self.u[0])):
                raise ValueError("Probe position is out of MTL length.")

        start_index = np.argmin(np.apply_along_axis(np.linalg.norm, 1, np.abs(self.u - start)))
        end_index   = np.argmin(np.apply_along_axis(np.linalg.norm, 1, np.abs(self.u - end)))
        
        source_factor = {}
        for index in range(start_index, end_index):
            source_factor[index]  = 1.0
        
        if (np.linalg.norm(np.abs(start - self.u[start_index])) > 1e-10) and start_index != 0:
            
            du = start-self.u[start_index]
            cross_prev = np.linalg.norm(np.cross(self.du[start_index-1], du))
            cross_curr = np.linalg.norm(np.cross(self.du[start_index], du))
            
            if (cross_prev < 1e-10 and cross_curr < 1e-10):
                if (du.dot(self.du[start_index]) < 0):
                    frac = np.linalg.norm(np.abs(start - self.u[start_index]))/np.linalg.norm(self.du[start_index-1])
                    source_factor[start_index-1] = frac
                elif (du.dot(self.du[start_index]) > 0):
                    frac = 1 - np.linalg.norm(np.abs(start - self.u[start_index]))/np.linalg.norm(self.du[start_index])
                    source_factor[start_index] = frac
            
            elif (cross_prev < 1e-10):
                frac = np.linalg.norm(np.abs(start - self.u[start_index]))/np.linalg.norm(self.du[start_index-1])
                source_factor[start_index-1] = frac
            elif (cross_curr < 1e-10):
                frac = 1 - np.linalg.norm(np.abs(start - self.u[start_index]))/np.linalg.norm(self.du[start_index])
                source_factor[start_index] = frac
                
        if (np.linalg.norm(np.abs(end - self.u[end_index])) > 1e-10) and end_index != self.u.shape[0]:
            
            du = end-self.u[end_index]
            cross_prev = np.linalg.norm(np.cross(self.du[end_index-1], du))
            cross_curr = np.linalg.norm(np.cross(self.du[end_index], du))
            
            if (cross_prev < 1e-10 and cross_curr < 1e-10):
                if (du.dot(self.du[end_index]) < 0):
                    frac = 1-np.linalg.norm(np.abs(end - self.u[end_index]))/np.linalg.norm(self.du[end_index-1])
                    source_factor[end_index-1] = frac
                    source_factor[end_index] = 0.0
                elif (du.dot(self.du[end_index]) > 0):
                    frac = np.linalg.norm(np.abs(end - self.u[end_index]))/np.linalg.norm(self.du[end_index])
                    source_factor[end_index] = frac
            
            elif (cross_prev < 1e-10):
                frac = 1 - np.linalg.norm(np.abs(end - self.u[end_index]))/np.linalg.norm(self.du[end_index-1])
                source_factor[end_index-1] = frac
                source_factor[end_index] = 0.0
            elif (cross_curr < 1e-10):
                frac = np.linalg.norm(np.abs(end - self.u[end_index]))/np.linalg.norm(self.du[end_index])
                source_factor[end_index] = frac
               
        for index, factor in source_factor.items():
            self.e_L[conductor, index] = add(self.e_L[conductor, index],multiply(magnitude, factor))

    def add_probe(self, position, probe_type = str):
        if (type(position) == float):
            position = np.array([0.0,0.0,position])
        
        start = self.u[0].copy()
        
        probe_in_segment = False
        for du in self.du:
            end = start + du
            if point_in_line(position, start, end):
                probe_in_segment = True
            start += du
        
        # if (np.any(position > self.u[-1])) or np.any((position < self.u[0])):
        if not probe_in_segment:
            raise ValueError("Probe position is out of MTL length.")

        probe = Probe(position, probe_type, self.dt, self.u)
        self.probes.append(probe)
        return probe


    def update_lr_terms(self):
        
        F1 = np.einsum('...ij,...jk->...ik' , 
                       self.du_norm, 
                       (self.l/self.dt + 0.5*self.transfer_impedance.d + self.transfer_impedance.e/self.dt + 0.5*self.r + self.transfer_impedance.q1sum))
        F2 = np.einsum('...ij,...jk->...ik' , 
                       self.du_norm, 
                       (self.l/self.dt - 0.5*self.transfer_impedance.d + self.transfer_impedance.e/self.dt - 0.5*self.r - self.transfer_impedance.q2sum))
        
        IF1 = np.linalg.inv(F1)
        self.i_term = np.einsum('...ij,...jk->...ik' , IF1, F2)
        self.v_diff = IF1

    def update_cg_terms(self):
        
        du_norm = np.zeros(shape=(self.du_norm.shape[0]+1, self.number_of_conductors, self.number_of_conductors))
        du_norm[0] = self.du_norm[0]
        for i in range(1, self.du_norm.shape[0]):
            du_norm[i]= 0.5*(self.du_norm[i]+self.du_norm[i-1])
        du_norm[-1] = self.du_norm[-1]
        
        F1 = np.einsum('...ij,...jk->...ik' , du_norm, self.c/self.dt + self.g/2)
        F2 = np.einsum('...ij,...jk->...ik' , du_norm, self.c/self.dt - self.g/2)
        try:
            IF1 = np.linalg.inv(F1)
        except:
            raise Exception('')
        self.v_term = np.einsum('...ij,...jk->...ik' , IF1, F2)
        self.i_diff = IF1




    def get_phase_velocities(self):
        return np.array([
            1/np.sqrt(np.real(linalg.eigvals(self.l[k].dot(self.c[k+1]))))
            for k in range(self.u.shape[0] - 1)
        ])


    # def add_external_field(self, field : Field, distances: np.array):
    def add_external_field(self, field : Field, distances: npt.NDArray[np.float64], field_localization: list[npt.NDArray[np.float64]] = []):

        assert(distances.shape == (self.levels[0][0].number_of_conductors, self.u.shape[0], 3))
        x, y, z, t, v = sp.symbols('x y z t v')

        indices = range(self.u.shape[0] - 1)
        if len(field_localization) != 0:
            start_index = np.argmin(np.apply_along_axis(np.linalg.norm, 1, np.abs(self.u - field_localization[0])))
            end_index =   np.argmin(np.apply_along_axis(np.linalg.norm, 1, np.abs(self.u - field_localization[1])))
            indices = range(start_index, end_index+1)
        
        vu = self.get_phase_velocities()
        for n in range(self.levels[0][0].number_of_conductors):
            for nz in indices:
              
                eL = field.compute_eL_at_segment(self.du[nz])
                x0, y0, z0 = self.u[nz,:]
                xi, yi, zi = distances[n, nz, :]
                l = np.abs(np.array([xi,yi,zi]))
                l /=np.linalg.norm(l)
                self.e_L[n, nz] = sp.lambdify(t, eL.subs(x, x0 + xi - 0.5*self.du[nz,0]).subs(y, y0 + yi- 0.5*self.du[nz,1]).subs(z, z0 + zi- 0.5*self.du[nz,2]).subs(v,np.max(vu[nz])) -\
                                                 eL.subs(x,x0- 0.5*self.du[nz,0]).subs(y, y0- 0.5*self.du[nz,1]).subs(z, z0- 0.5*self.du[nz,2]).subs(v,np.max(vu[nz])))

                self.e_T[n,nz] = sp.lambdify(t,
                        l[0]*sp.Integral(field.e_x.subs(y, y0).subs(z, z0).subs(v, np.max(vu[nz])), (x, x0, x0 + xi))+\
                        l[1]*sp.Integral(field.e_y.subs(x, x0).subs(z, z0).subs(v, np.max(vu[nz])), (y, y0, y0 + yi))+\
                        l[2]*sp.Integral(field.e_z.subs(x, x0).subs(y, y0).subs(v, np.max(vu[nz])), (z, z0, z0 + zi)))
            xi, yi, zi = distances[n, -1, :]
            l = np.abs(np.array([xi,yi,zi]))
            l /=np.linalg.norm(l)
            self.e_T[n,-1] = sp.lambdify(t,
                    l[0]*sp.Integral(field.e_x.subs(y, y0).subs(z, z0).subs(v, np.max(vu[-1])), (x, x0, x0 + xi))+\
                    l[1]*sp.Integral(field.e_y.subs(x, x0).subs(z, z0).subs(v, np.max(vu[-1])), (y, y0, y0 + yi))+\
                    l[2]*sp.Integral(field.e_z.subs(x, x0).subs(y, y0).subs(v, np.max(vu[-1])), (z, z0, z0 + zi)))

    @dispatch()
    def update_sources(self):
        # self.v_sources_now = np.vectorize(FunctionType.__call__, otypes=["float64"])(
        #     self.v_sources, self.time
        # )
        # self.v_sources_prev = np.vectorize(FunctionType.__call__, otypes=["float64"])(
        #     self.v_sources, self.time - self.dt
        # )
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

    @dispatch(float,float)
    def update_sources(self, time, dt):
        # self.v_sources_now = np.vectorize(FunctionType.__call__, otypes=["float64"])(
        #     self.v_sources, self.time
        # )
        # self.v_sources_prev = np.vectorize(FunctionType.__call__, otypes=["float64"])(
        #     self.v_sources, self.time - self.dt
        # )
       
        self.e_L_now = np.vectorize(FunctionType.__call__, otypes=["float64"])(
            self.e_L, time + dt / 2
        )
        self.e_L_prev = np.vectorize(FunctionType.__call__, otypes=["float64"])(
            self.e_L, time - dt / 2
        )
        self.e_T_now = np.vectorize(FunctionType.__call__, otypes=["float64"])(
            self.e_T, time
        )
        self.e_T_prev = np.vectorize(FunctionType.__call__, otypes=["float64"])(
            self.e_T, time - dt
        )

    def advance_voltage(self):
        self.v[:, 1:-1] = np.einsum('...ij,...j->...i',self.v_term[1:-1,:,:],self.v.T[1:-1,:]).T-\
                          np.einsum('...ij,...j->...i',self.i_diff[1:-1,:,:],(self.i[:,1:]-self.i[:,:-1]).T).T-\
                          (self.e_T_now[:, 1:-1] - self.e_T_prev[:, 1:-1])
      
        
    def advance_current(self):
        self.transfer_impedance.update_q3_phi_term()
        i_prev = self.i
        self.i[:, :] = np.einsum('...ij,...j->...i',self.i_term[:,:,:],self.i.T[:,:]).T-\
                       np.einsum('...ij,...j->...i',self.v_diff[:,:,:],(self.v[:, 1:] - self.v[:, :-1]
                                            + (self.e_T_now[:, 1:] - self.e_T_now[:, :-1])
                                            - (np.einsum('...ij,...j->...i', self.du_norm / 2, (self.e_L_now[:, :] + self.e_L_prev[:, :]).T).T)).T).T-\
                       np.einsum('...ij,...j->...i',self.v_diff[:,:,:],np.einsum('...ij,...j->...i',self.du_norm,self.transfer_impedance.q3_phi_term[:,:])).T

        

        i_now = self.i
        self.transfer_impedance.update_phi(i_prev, i_now)


    def add_transfer_impedance(self, 
        out_level, out_level_conductors,
        in_level, in_level_conductors,
        transfer_impedance: dict
    ):
        self.transfer_impedance.add_transfer_impedance(
            self.conductors_in_level,
            out_level, out_level_conductors, 
            in_level, in_level_conductors,
            transfer_impedance)

    def set_connector_transfer_impedance(self,
        side,
        out_level, out_level_conductors,
        in_level, in_level_conductors,
        transfer_impedance: dict
    ):
        
        if side == 'initial':
            index = 0
        elif side == 'end':
            index = -1
        else:
            raise Exception("side must be either 'initial' or 'end'")
        
        self.transfer_impedance.set_transfer_impedance_at_index(
            self.conductors_in_level,
            out_level, out_level_conductors, 
            in_level, in_level_conductors,
            index, transfer_impedance)

    def set_resistance_in_region(self, start, end, conductor, resistance):
        assert(end >= start)
        if (type(start) == float):
            start = np.array([0.0,0.0,start])
        if (type(end) == float):
            end = np.array([0.0,0.0,end])

        index1 = np.argmin(np.apply_along_axis(np.linalg.norm, 1, np.abs(self.u - start)))
        index2 = np.argmin(np.apply_along_axis(np.linalg.norm, 1, np.abs(self.u - end)))
        
        if (index1 != index2):
            self.r[index1:index2+1][conductor][conductor] = resistance/np.linalg.norm(self.u[index2]-self.u[index1])
        else:
            self.r[index1:index2+1][conductor][conductor] = resistance/np.linalg.norm(self.du[index1])
            
        self.update_lr_terms()

    def add_conductance_in_region(self, start, end, conductor, conductance):
        assert(end >= start)

        if (type(start) == float):
            start = np.array([0.0,0.0,start])
        if (type(end) == float):
            end = np.array([0.0,0.0,end])

        index1 = np.argmin(np.apply_along_axis(np.linalg.norm, 1, np.abs(self.u - start)))
        index2 = np.argmin(np.apply_along_axis(np.linalg.norm, 1, np.abs(self.u - end)))
        
        if (index1 != index2):
            self.g[index1:index2+1][conductor][conductor] = conductance/np.linalg.norm(self.u[index2]-self.u[index1])
        else:
            self.g[index1:index2+1][conductor][conductor] = conductance/np.linalg.norm(self.du[index1])
            
        self.update_cg_terms()

