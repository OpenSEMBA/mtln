import numpy as np
import scipy.linalg as linalg

from types import FunctionType
from typing import Dict

import src.mtl as mtl

class Network:
    """
    Networks can be joining tubes (junctions) or ending tubes (terminations)
    """
    def __init__(self, nw_number, nodes: list[int], bundles: list[int]):
        self.number_of_nodes = len(nodes)
        self.nw_number = nw_number
        self.nodes = nodes
        self.bundles = bundles
        self.connections = {}
        self.nw_v = np.zeros([0])
        self.nw_i = np.zeros([0])
        self.P1 = np.zeros([self.number_of_nodes, self.number_of_nodes])
        self.Ps = np.zeros([self.number_of_nodes, self.number_of_nodes])
        self.v_sources = np.empty(shape=(self.number_of_nodes), dtype=object)
        self.v_sources.fill(lambda n: 0)

        self.dx = np.ndarray(shape=(0,0))
        # self.dx = np.zeros([0])
        self.c = np.ndarray(shape=(0,0))
        
        self.e_T = np.empty(shape=(0), dtype=object)
        self.e_T.fill(lambda n: 0)
        
    def add_nodes_in_bundle(self, bundle_number: int, bundle: mtl, connections: dict, side: str):
        assert (side == "S" or side == "L")
        assert (bundle_number in self.bundles)
        
        if (side == "S"):
            side_idx = 0
        elif (side == "L"):
            side_idx = -1

        self.c = linalg.block_diag(self.c, bundle.c[side_idx])
           
        for connection in connections:
            self._add_node(connection, bundle_number, side_idx, bundle)
        
    def _add_node(self, connection: dict, bundle_number: int, side_idx: int, bundle):
        assert (connection["node"] in self.nodes)
        assert (connection["node"] >= 0)
        assert (connection["node"] not in self.connections.keys())

        index = self.nw_v.shape[0]
        self.nw_v = np.append(self.nw_v,0.0)
        self.nw_i = np.append(self.nw_i,0.0)
        self.dx = linalg.block_diag(self.dx, bundle.dx)
        self.e_T = np.append(self.e_T, bundle.e_T[connection["conductor"]][side_idx])

        self.connections[connection["node"]] = {"bundle_number" : bundle_number, "conductor" : connection["conductor"], "side" : side_idx, "index": index}
        if "mtl" in connection.keys():
            self.connections[connection["node"]]["mtl"] = connection["mtl"]
            
    def connect_to_ground(self, node: int, R = 0, Vt = 0):
        assert(node in self.connections.keys())
        index = self.connections[node]["index"]
        if (R != 0):
            self.P1[index, index] = -1/R
        if (Vt != 0):
            self.v_sources[index]  = Vt
            self.Ps[index, index] = 1/R

    def short_to_ground(self, node: int):
        assert(node in self.connections.keys())
        index = self.connections[node]["index"]
        self.P1[index, index] = -1e10
            
    def connect_nodes(self, node1: int, node2: int, R, Vt = 0):
        assert(node1 in self.connections.keys() and node2 in self.connections.keys())
        assert(R != 0)
        index1 = self.connections[node1]["index"]
        index2 = self.connections[node2]["index"]
        assert(self.P1[index1, index1] == 0)
        if (R != 0):
            # self.P1[index1, index1] = -1/R 
            # self.P1[index1, index2] = -1/R
            # self.P1[index2, index1] = -1/R
            # self.P1[index2, index2] = -1/R
            self.P1[index1, index1] = -1/R 
            self.P1[index1, index2] = 1/R
            self.P1[index2, index1] = 1/R
            self.P1[index2, index2] = -1/R
            if (Vt != 0):
                # self.Ps[index1, index1] =  1/R 
                # self.Ps[index2, index2] = 1/R
                self.Ps[index1, index1] =  1/R 
                self.Ps[index2, index2] = -1/R
                
                self.v_sources[index1]  = Vt
                self.v_sources[index2]  = Vt
      
    def short_nodes(self, node1: int, node2: int):
        index1 = self.connections[node1]["index"]
        index2 = self.connections[node2]["index"]
        # self.P1[index1, index1] =   -1e10
        # self.P1[index1, index2] =   -1e10
        # self.P1[index2, index1] =   -1e10
        # self.P1[index2, index2] =   -1e10
        self.P1[index1, index1] =  -1e10
        self.P1[index1, index2] =   1e10
        self.P1[index2, index1] =   1e10
        self.P1[index2, index2] =  -1e10
      
    #External field missing
    def update_sources(self, time, dt):
        self.v_sources_now = np.vectorize(FunctionType.__call__, otypes=["float64"])(self.v_sources, time)
        self.v_sources_prev = np.vectorize(FunctionType.__call__, otypes=["float64"])(self.v_sources, time - dt)
        self.e_T_now = np.vectorize(FunctionType.__call__, otypes=["float64"])(self.e_T,  time)
        self.e_T_prev = np.vectorize(FunctionType.__call__, otypes=["float64"])(self.e_T, time - dt)

    def advance_voltage(self):
        self.nw_v = self.terminal_term_1.dot(self.nw_v) +\
                    self.terminal_term_2.dot(self.Ps.dot(self.v_sources_now + self.v_sources_prev) - 2*self.nw_i)+\
                    self.terminal_term_3.dot(self.e_T_now - self.e_T_prev)
       
    def update_voltages(self, bundles):
        for node in self.connections.values():
            bundles[node["bundle_number"]].v[node["conductor"],node["side"]] = self.nw_v[node["index"]]

    def i_factor(self,node):
        if (node["side"] == 0):
            return 1.0
        elif (node["side"] == -1):
            return -1.0

    def update_currents(self, bundles):
        for node in self.connections.values():
            self.nw_i[node["index"]] =  self.i_factor(node)*bundles[node["bundle_number"]].i[node["conductor"],node["side"]]

    def compute_v_terms(self, dt):
        inv = np.linalg.inv(self.dx.dot(self.c) / dt - self.P1)
        self.terminal_term_1 = inv.dot(self.dx.dot(self.c) / dt + self.P1)
        self.terminal_term_2 = inv
        self.terminal_term_3 = -inv.dot(self.dx).dot(self.c)/dt
       
class LNetwork:
    """
    Networks can be joining tubes (junctions) or ending tubes (terminations)
    """
    def __init__(self, levels : Dict[int, Network], nw_number : int):
        self.nw_number = nw_number
        self.levels = levels

    def update_sources(self, time, dt):
        for nw in self.levels.values():
            nw.update_sources(time, dt)

    def advance_voltage(self):
        for nw in self.levels.values():
            nw.advance_voltage()

    def update_voltages(self, bundles):
        for nw in self.levels.values():
            nw.update_voltages(bundles)

    def update_currents(self, bundles):
        for nw in self.levels.values():
            nw.update_currents(bundles)

    def compute_v_terms(self, dt):
        for nw in self.levels.values():
            nw.compute_v_terms(dt)

