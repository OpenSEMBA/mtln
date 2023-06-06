import numpy as np
import skrf as rf

from copy import deepcopy
from numpy.fft import fft, fftfreq, fftshift
import sympy as sp

import src.mtl as mtl
import scipy.linalg as linalg

from types import FunctionType
from types import LambdaType

class Network:
    """
    Networks can be joining tubes (junctions) or ending tubes (terminations)
    """
    def __init__(self, nw_number, nodes: list[int]):
        self.number_of_nodes = len(nodes)
        self.nw_number = nw_number
        self.nodes = nodes
        self.connections = {}
        self.bundle_connections = {}
        self.nw_v = np.zeros([0])
        self.nw_i = np.zeros([0])
        self.P1 = np.zeros([self.number_of_nodes, self.number_of_nodes])
        self.Ps = np.zeros([self.number_of_nodes, self.number_of_nodes])
        self.Pshort = np.zeros([self.number_of_nodes, self.number_of_nodes])
        self.v_sources = np.empty(shape=(self.number_of_nodes), dtype=object)
        self.v_sources.fill(lambda n: 0)

        self.c = np.ndarray(shape=(0,0))
        
    def add_nodes_in_bundle(self, bundle_number: int, bundle: mtl, connections: dict, side: str):
        assert (side == "S" or side == "L")
        
        if (side == "S"):
            self.c = linalg.block_diag(self.c, bundle.c[0])
        elif (side == "L"):
            self.c = linalg.block_diag(self.c, bundle.c[-1])
            
        for connection in connections:
            self._add_node(connection["node"], bundle_number, connection["conductor"], side)
        
    def _add_node(self,nw_node: int, bundle_number: int, conductor: int, side: str):
        assert (nw_node in self.nodes)
        assert (nw_node >= 0)
        assert (nw_node not in self.connections.keys())

        index = self.nw_v.shape[0]
        self.nw_v = np.append(self.nw_v,0.0)
        self.nw_i = np.append(self.nw_i,0.0)

        self.connections[nw_node] = {"bundle_number" : bundle_number, "conductor" : conductor, "side" : side, "index": index}
        self.bundle_connections[bundle_number, side] = {"conductor" : conductor, "node_number": nw_node}

      
    def connect_to_ground(self, node: int, R = 0, Vt = 0, side = ""):
        assert(node in self.connections.keys())
        index = self.connections[node]["index"]
           
        if (R != 0):
            self.P1[index, index] = -1/R
        if (Vt != 0):
            self.v_sources[index]  = Vt
            self.Ps[index, index] = 1/R
            
    def connect_nodes(self, node1: int, node2: int, R, Vt = 0):
        assert(node1 in self.connections.keys() and node2 in self.connections.keys())
        assert(R != 0)
        index1 = self.connections[node1]["index"]
        index2 = self.connections[node2]["index"]
        if (R != 0):
            #signos!?
            self.P1[index1, index1] = -1/R 
            self.P1[index1, index2] = 1/R
            self.P1[index2, index1] = 1/R
            self.P1[index2, index2] = -1/R
            if (Vt != 0):
                self.Ps[index1, index1] = 1/R 
                self.Ps[index1, index2] = 1/R
                self.Ps[index2, index1] = 1/R
                self.Ps[index2, index2] = 1/R
                
                self.v_sources[index1]  = Vt
                self.v_sources[index2]  = Vt
      
    def short_nodes(self, node1: int, node2: int):
        index1 = self.connections[node1]["index"]
        index2 = self.connections[node2]["index"]
        self.P1[index1, index1] =  -1e10
        self.P1[index1, index2] =  1e10
        self.P1[index2, index1] =  1e10
        self.P1[index2, index2] =  -1e10
      
    def step(self, time, dt):
        sources_now = np.vectorize(FunctionType.__call__, otypes=["float64"])(self.v_sources, time)
        sources_prev = np.vectorize(FunctionType.__call__, otypes=["float64"])(self.v_sources, time - dt)
        self.nw_v = self.terminal_term_1.dot(self.nw_v) + self.terminal_term_2.dot(self.Ps.dot(sources_now + sources_prev) - 2*self.nw_i[:]) #+ self.Pshort.dot(self.nw_v)
       

class MTLN:
    """
    Lossless Multiconductor Transmission Line Network 
    """
    def __init__(self):
        self.bundles = {}
        self.networks = {}
        self.dt = 1e10
        self.time = 0.0

    def add_bundle(self, bundle_number: int, bundle: mtl):
        assert(bundle_number not in self.bundles.keys())
        self.bundles[bundle_number] = bundle
        self.dx = bundle.dx        
        if (bundle.dt < self.dt):
            self.dt = bundle.dt
        
    def add_network(self, nw: Network):
        self.networks[nw.nw_number] = nw
        
    def get_time_range(self, final_time):
        return np.arange(0, np.floor(final_time / self.dt))

    def compute_v_terms(self):
        for nw in self.networks.values():
            inv = np.linalg.inv(self.dx * nw.c / self.dt - nw.P1)
            nw.terminal_term_1 = inv.dot(self.dx * nw.c / self.dt + nw.P1)
            nw.terminal_term_2 = inv

    def update_probes(self):
        for bundle in self.bundles.values():
            for p in bundle.probes:
                p.update(self.time, bundle.x, bundle.v, bundle.i)

    def run_until(self, finalTime):
        
        self.compute_v_terms()
        
        t = self.get_time_range(finalTime)
        for bundle in self.bundles.values():
            for p in bundle.probes:
                p.resize_frames(len(t), bundle.number_of_conductors)

        for _ in t:
            self.step()

    def update_networks_current(self):
        for nw  in self.networks.values():
            for node in nw.connections.values():
                if (node["side"] == "S"):
                    nw.nw_i[node["index"]] =  self.bundles[node["bundle_number"]].i[node["conductor"],0]
                if (node["side"] == "L"):
                    nw.nw_i[node["index"]] =  self.bundles[node["bundle_number"]].i[node["conductor"],-1]

    def update_networks_voltage(self):
        for nw  in self.networks.values():
            nw.step(self.time, self.dt)
            for node in nw.connections.values():
                if (node["side"] == "S"):
                    self.bundles[node["bundle_number"]].v[node["conductor"],0] = nw.nw_v[node["index"]]
                if (node["side"] == "L"):
                    self.bundles[node["bundle_number"]].v[node["conductor"],-1] = -nw.nw_v[node["index"]]

    
    def update_bundles_voltage(self):
        for bundle in self.bundles.values():
            bundle.update_sources()
            bundle.v[:, 1:-1] = np.einsum('...ij,...j->...i',bundle.v_term[1:-1,:,:], bundle.v.T[1:-1,:]).T-\
                                np.einsum('...ij,...j->...i',bundle.i_diff[1:-1,:,:],(bundle.i[:,1:]-bundle.i[:,:-1]).T).T-\
                                (bundle.e_T_now[:, 1:-1] - bundle.e_T_prev[:, 1:-1])


    def update_bundles_current(self):
        for bundle in self.bundles.values():
            bundle.i[:, :] = np.einsum('...ij,...j->...i',bundle.i_term[:,:,:],bundle.i.T[:,:]).T-\
                             np.einsum('...ij,...j->...i',bundle.v_diff[:,:,:],(bundle.v[:, 1:] - bundle.v[:, :-1]+\
                            (bundle.e_T_now[:, 1:] - bundle.e_T_now[:, :-1])-\
                            (bundle.dx / 2) * (bundle.e_L_now[:, :] + bundle.e_L_prev[:, :])
                            ).T).T
                             

    def step(self):
        self.update_bundles_voltage()
        self.update_networks_voltage()
        self.update_bundles_current()
        self.update_networks_current()

        self.time += self.dt
        self.update_probes()