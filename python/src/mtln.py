import numpy as np
import skrf as rf

from copy import deepcopy
from numpy.fft import fft, fftfreq, fftshift
import sympy as sp

import src.mlt as mtl
import scipy.linalg as linalg

from types import FunctionType
from types import LambdaType


# class Node:
#     def __init__(self, network: Network):
#         self.connections = {}

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
        self.v = np.zeros([self.number_of_nodes])
        self.P1 = np.zeros([self.nw_number, self.nw_number])
        self.Ps = np.zeros([self.nw_number, self.nw_number])
        self.v_sources = np.empty(shape=(self.number_of_nodes), dtype=object)
        self.c = np.zeros([0])
        
    def add_nodes_in_bundle(self, bundle_number, bundle_c, connections: dict, side: str):
        
        self.c = linalg.block_diag(self.c, bundle_c)
        
        for connection in connections:
            self._add_node(connection["node"], bundle_number, connection["conductor"], side)
        
    def _add_node(self,nw_node: int, bundle_number: int, conductor: int, side: str):
        assert (nw_node in self.nodes)
        assert (nw_node >= 0)
        assert (nw_node not in self.connections.keys())
        # assert (side == "L" or side == "R")
        self.connections[nw_node] = {"bundle_number" : bundle_number, "conductor" : conductor, "side" : side}
        self.bundle_connections[bundle_number, side] = {"conductor" : conductor, "node_number": nw_node}
        self.v[nw_node] = float()
      
      
    def connect_to_ground(self, node1: int, R = 0, Vt = 0):
        assert(node1 in self.connections.keys())
        if (R != 0):
            self.P1[node1, node1] = -1/R 
        if (Vt != 0):
            self.v_sources[node1]  = Vt
            self.Ps[node1, node1] = 1/R 
            
    def connect_nodes(self, node1: int, node2: int, R = 0, Vt = 0):
        assert(node1 in self.connections.keys() and node2 in self.connections.keys())
        if (R != 0):
            #signos!?
            self.P1[node1, node1] = -1/R 
            self.P1[node1, node2] = 1/R
            self.P1[node2, node1] = -1/R
            self.P1[node2, node2] = 1/R
            pass
        if (Vt != 0):
            self.Ps[node1, node1] = 1/R 
            self.Ps[node1, node2] = 1/R
            self.Ps[node2, node1] = 1/R
            self.Ps[node2, node2] = 1/R
            
            self.v_sources[node1]  = Vt
            self.v_sources[node2]  = Vt

      
    def step(self):
        #update the voltage of each node. Depends on the state of the network
        # self.v[] = self.
        pass

        

class MTLN(mtl):
    """
    Lossless Multiconductor Transmission Line Network 
    """
    def __init__(self):
        self.bundles = {}
        self.networks = {}
        self.dt = 1e10
        
    def add_bundle(self, bundle_number: int, bundle: mtl):
        self.bundles[bundle_number] = bundle
        
        if (bundle.dt < self.dt):
            self.dt = bundle.dt
        
    def add_network(self, nw: Network):
        self.networks[nw.nw_number] = nw
        
    def get_time_range(self, final_time):
        return np.arange(0, np.floor(final_time / self.dt))

    def compute_v_terms(self):
        for nw, nw_number in self.networks:
            self.networks[nw_number].terminal_term_1 = self.dx * nw.c / self.dt - nw.P1
            self.networks[nw_number].terminal_term_2 = self.dx * nw.c / self.dt + nw.P1


    def run_until(self, finalTime):
        
        self.compute_v_terms()
        
        t = self.get_time_range(finalTime)

        for p in self.probes:
            p.resize_frames(len(t), self.number_of_conductors)

        for _ in t:
            self.step()

    #the step function has to be redefined, to be compatible with the
    #network structure and update
    def step(self):
        #update voltage of each bundle
        for bndl_number, bundle in self.bundles.items():
            self.bundles[bndl_number].v[:, 1:-1] = np.einsum('...ij,...j->...i',bundle.v_term[1:-1,:,:], bundle.v.T[1:-1,:]).T-\
                                np.einsum('...ij,...j->...i',bundle.i_diff[1:-1,:,:],(bundle.i[:,1:]-bundle.i[:,:-1]).T).T-\
                                (bundle.e_T_now[:, 1:-1] - bundle.e_T_prev[:, 1:-1])

        # are this and the expression above equivalent?
        # self.bundles[:].v[:, 1:-1] = np.einsum('...ij,...j->...i',self.bundles.v_term[1:-1,:,:], self.bundles.v.T[1:-1,:]).T-\
        #                              np.einsum('...ij,...j->...i',self.bundles.i_diff[1:-1,:,:],(self.bundles.i[:,1:]-self.bundles.i[:,:-1]).T).T-\
        #                              (self.bundles.e_T_now[:, 1:-1] - self.bundles.e_T_prev[:, 1:-1])

        for nw_number, nw  in self.networks.items():
            # update voltage of each node of each network
            self.networks[nw_number].step()
            # these voltages correspond to v[0] and v[-1] of the bundles. Use the connections dictionary to match them
            for (node_number, node) in nw.connections.items():
                if (node["side"] == "L"):
                    # nw.v[node_number] = 0#compute according to its state
                    self.bundle[node["bundle_number"]].v[node["conductor"],0] = nw.v[node_number]
                if (node["side"] == "R"):
                    # nw.v[node_number] = 0#compute according to its state
                    self.bundle[node["bundle_number"]].v[node["conductor"],-1] = nw.v[node_number]
        

        for bndl_number, bundle in self.bundles.items():
            self.bundles[bndl_number].i[:, :] = np.einsum('...ij,...j->...i',bundle.i_term[:,:,:],bundle.i.T[:,:]).T-\
                                                np.einsum('...ij,...j->...i',bundle.v_diff[:,:,:],(bundle.v[:, 1:] - bundle.v[:, :-1]+\
                                                (bundle.e_T_now[:, 1:] - bundle.e_T_now[:, :-1])-\
                                                (bundle.dx / 2) * (bundle.e_L_now[:, :] + bundle.e_L_prev[:, :])).T).T

        # are this and the expression above equivalent?
        # self.bundles[:].i[:, :] = np.einsum('...ij,...j->...i',self.bundles[:].i_term[:,:,:],self.bundles[:].i.T[:,:]).T-\
        #                           np.einsum('...ij,...j->...i',self.bundles[:].v_diff[:,:,:],(self.bundles[:].v[:, 1:] - self.bundles[:].v[:, :-1]+\
        #                           (self.bundles[:].e_T_now[:, 1:] - self.bundles[:].e_T_now[:, :-1])-\
        #                           (self.bundles[:].dx / 2) * (self.bundles[:].e_L_now[:, :] + self.bundles[:].e_L_prev[:, :])).T).T

        #update current of each bundle