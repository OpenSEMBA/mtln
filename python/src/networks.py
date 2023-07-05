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

        self.c = np.ndarray(shape=(0,0))
        
    def add_nodes_in_bundle(self, bundle_number: int, bundle: mtl, connections: dict, side: str):
        assert (side == "S" or side == "L")
        assert (bundle_number in self.bundles)
        if (side == "S"):
            self.c = linalg.block_diag(self.c, bundle.c[0])
        elif (side == "L"):
            self.c = linalg.block_diag(self.c, bundle.c[-1])
            
        for connection in connections:
            self._add_node(connection, bundle_number, side)
        
    def _add_node(self, connection: dict, bundle_number: int, side: str):
        assert (connection["node"] in self.nodes)
        assert (connection["node"] >= 0)
        assert (connection["node"] not in self.connections.keys())

        index = self.nw_v.shape[0]
        self.nw_v = np.append(self.nw_v,0.0)
        self.nw_i = np.append(self.nw_i,0.0)

        self.connections[connection["node"]] = {"bundle_number" : bundle_number, "conductor" : connection["conductor"], "side" : side, "index": index}
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
            
    def connect_nodes(self, node1: int, node2: int, R, Vt = 0):
        assert(node1 in self.connections.keys() and node2 in self.connections.keys())
        assert(R != 0)
        index1 = self.connections[node1]["index"]
        index2 = self.connections[node2]["index"]
        assert(self.P1[index1, index1] == 0)
        if (R != 0):
            self.P1[index1, index1] = -1/R 
            self.P1[index1, index2] = 1/R
            self.P1[index2, index1] = 1/R
            self.P1[index2, index2] = -1/R
            if (Vt != 0):
                self.Ps[index1, index1] =  1/R 
                self.Ps[index2, index2] = -1/R
                
                self.v_sources[index1]  = Vt
                self.v_sources[index2]  = Vt
      
    def short_nodes(self, node1: int, node2: int):
        index1 = self.connections[node1]["index"]
        index2 = self.connections[node2]["index"]
        self.P1[index1, index1] =  -1e10
        self.P1[index1, index2] =   1e10
        self.P1[index2, index1] =   1e10
        self.P1[index2, index2] =  -1e10
      
    #External field missing
    def update_sources(self, time, dt):
        self.v_sources_now = np.vectorize(FunctionType.__call__, otypes=["float64"])(self.v_sources, time)
        self.v_sources_prev = np.vectorize(FunctionType.__call__, otypes=["float64"])(self.v_sources, time - dt)
      
    def advance_voltage(self):
        self.nw_v = self.terminal_term_1.dot(self.nw_v) + self.terminal_term_2.dot(self.Ps.dot(self.v_sources_now + self.v_sources_prev) - 2*self.nw_i[:])
       
    def update_voltages(self, bundles):
        for node in self.connections.values():
            if (node["side"] == "S"):
                bundles[node["bundle_number"]].v[node["conductor"],0] = self.nw_v[node["index"]]
            if (node["side"] == "L"):
                bundles[node["bundle_number"]].v[node["conductor"],-1] = self.nw_v[node["index"]]

    def update_currents(self, bundles):
        for node in self.connections.values():
            if (node["side"] == "S"):
                self.nw_i[node["index"]] =  bundles[node["bundle_number"]].i[node["conductor"],0]
            if (node["side"] == "L"):
                self.nw_i[node["index"]] =  -bundles[node["bundle_number"]].i[node["conductor"],-1]

    def compute_v_terms(self, dx, dt):
        inv = np.linalg.inv(dx * self.c / dt - self.P1)
        self.terminal_term_1 = inv.dot(dx * self.c / dt + self.P1)
        self.terminal_term_2 = inv

       
class LNetwork:
    """
    Networks can be joining tubes (junctions) or ending tubes (terminations)
    """
    # def __init__(self, nw_number, nodes: list[int], bundles: list[int]):
    def __init__(self, levels : Dict[int, Network], nw_number : int):
        # self.number_of_nodes = len(nodes)
        self.nw_number = nw_number
        # self.nodes = nodes
        # self.bundles = bundles
        # self.connections = {}
        # self.nw_v = np.zeros([0])
        # self.nw_i = np.zeros([0])
        # self.P1 = np.zeros([self.number_of_nodes, self.number_of_nodes])
        # self.Ps = np.zeros([self.number_of_nodes, self.number_of_nodes])
        # self.v_sources = np.empty(shape=(self.number_of_nodes), dtype=object)
        # self.v_sources.fill(lambda n: 0)
        # self.c = np.ndarray(shape=(0,0))

        self.levels = levels

    # def add_level_network(self, nw, level):
    #     self.levels[level] = nw
        
    def update_sources(self, time, dt):
        for nw in self.levels.values():
            nw.update_sources(time, dt)


    def advance_voltage(self):
        for nw in self.levels.values():
            nw.advance_voltage()

    def update_voltages(self, bundles):
        for level, nw in self.levels.items():
            for node in nw.connections.values():
                if (node["side"] == "S"):
                    bundles[node["bundle_number"]].levels[level][node["mtl"]].v[node["conductor"],0] = nw.nw_v[node["index"]]
                    # bundles[node["bundle_number"]][level].v[node["conductor"],0] = nw.nw_v[node["index"]]
                if (node["side"] == "L"):
                    bundles[node["bundle_number"]].levels[level][node["mtl"]].v[node["conductor"],-1] = nw.nw_v[node["index"]]
                    # bundles[node["bundle_number"]][level].v[node["conductor"],-1] = nw.nw_v[node["index"]]

    def update_currents(self, bundles):
        for level, nw in self.levels.items():
            for node in nw.connections.values():
                if (node["side"] == "S"):
                    nw.nw_i[node["index"]] =  bundles[node["bundle_number"]].levels[level][node["mtl"]].i[node["conductor"],0]
                    # nw.nw_i[node["index"]] =  bundles[node["bundle_number"]][level].i[node["conductor"],0]
                if (node["side"] == "L"):
                    nw.nw_i[node["index"]] =  -bundles[node["bundle_number"]].levels[level][node["mtl"]].i[node["conductor"],-1]
                    # nw.nw_i[node["index"]] =  -bundles[node["bundle_number"]][level].i[node["conductor"],-1]

    def compute_v_terms(self, dx, dt):
        for nw in self.levels.values():
            inv = np.linalg.inv(dx * nw.c / dt - nw.P1)
            nw.terminal_term_1 = inv.dot(dx * nw.c / dt + nw.P1)
            nw.terminal_term_2 = inv


    # def add_nodes_in_bundle(self, bundle_number: int, bundle: mtl, connections: dict, side: str):
    #     assert (side == "S" or side == "L")
    #     assert (bundle_number in self.bundles)
    #     if (side == "S"):
    #         self.c = linalg.block_diag(self.c, bundle.c[0])
    #     elif (side == "L"):
    #         self.c = linalg.block_diag(self.c, bundle.c[-1])
            
    #     for connection in connections:
    #         self._add_node(connection["node"], bundle_number, connection["conductor"], side)
        
    # def _add_node(self,nw_node: int, bundle_number: int, conductor: int, side: str):
    #     assert (nw_node in self.nodes)
    #     assert (nw_node >= 0)
    #     assert (nw_node not in self.connections.keys())

    #     index = self.nw_v.shape[0]
    #     self.nw_v = np.append(self.nw_v,0.0)
    #     self.nw_i = np.append(self.nw_i,0.0)

    #     self.connections[nw_node] = {"bundle_number" : bundle_number, "conductor" : conductor, "side" : side, "index": index}
      
    # def connect_to_ground(self, node: int, R = 0, Vt = 0):
    #     assert(node in self.connections.keys())
    #     index = self.connections[node]["index"]
    #     if (R != 0):
    #         self.P1[index, index] = -1/R
    #     if (Vt != 0):
    #         self.v_sources[index]  = Vt
    #         self.Ps[index, index] = 1/R
            
    # def connect_nodes(self, node1: int, node2: int, R, Vt = 0):
    #     assert(node1 in self.connections.keys() and node2 in self.connections.keys())
    #     assert(R != 0)
    #     index1 = self.connections[node1]["index"]
    #     index2 = self.connections[node2]["index"]
    #     assert(self.P1[index1, index1] == 0)
    #     if (R != 0):
    #         self.P1[index1, index1] = -1/R 
    #         self.P1[index1, index2] = 1/R
    #         self.P1[index2, index1] = 1/R
    #         self.P1[index2, index2] = -1/R
    #         if (Vt != 0):
    #             self.Ps[index1, index1] =  1/R 
    #             self.Ps[index2, index2] = -1/R
                
    #             self.v_sources[index1]  = Vt
    #             self.v_sources[index2]  = Vt
      
    # def short_nodes(self, node1: int, node2: int):
    #     index1 = self.connections[node1]["index"]
    #     index2 = self.connections[node2]["index"]
    #     self.P1[index1, index1] =  -1e10
    #     self.P1[index1, index2] =   1e10
    #     self.P1[index2, index1] =   1e10
    #     self.P1[index2, index2] =  -1e10
      
