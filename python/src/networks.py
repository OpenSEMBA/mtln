import numpy as np
import numpy.typing as npt

import scipy.linalg as linalg

from types import FunctionType
from typing import Dict

import src.mtl as mtl
from src.utils import add_t_functions as add

class Network:
    """
    Networks can be joining tubes (junctions) or ending tubes (terminations)
    """
    def __init__(self, description):

        self.number_of_nodes = 0
        self.number_of_state_vars = 0
        self.connections : Dict[int, Dict[str, int, int, int, str]] = {}
        self.c : npt.NDArray[np.float64] = np.ndarray(shape=(0,0))
        self.nw_v = np.zeros([0])
        self.nw_i = np.zeros([0])
        self.dx : npt.NDArray[np.float64] = np.ndarray(shape=(0,0))
        self.e_T = np.empty(shape=(0), dtype=object)
        self.e_T.fill(lambda n: 0)

        for b in description.items():
            self.number_of_nodes += len(b[1]["connections"])
            self._add_nodes_in_line(b)

        self.X = np.zeros([self.number_of_state_vars])
        self.H = np.zeros([self.number_of_nodes + self.number_of_state_vars])

        self.nw_v_prev = self.nw_v

        self.P1 = np.zeros([self.number_of_nodes, self.number_of_nodes])
        self.Ps = np.zeros([self.number_of_nodes, self.number_of_nodes])

        self.M = np.zeros([self.number_of_state_vars, self.number_of_state_vars])
        self.N1 = np.zeros([self.number_of_state_vars, self.number_of_nodes])
        self.Ns = np.zeros([self.number_of_state_vars, self.number_of_nodes])
        self.O = np.zeros([self.number_of_nodes, self.number_of_state_vars])

        self.Q1 = np.zeros([self.number_of_nodes, self.number_of_nodes])
        self.Qs = np.zeros([self.number_of_nodes, self.number_of_nodes])

        self.v_sources = np.empty(shape=(self.number_of_nodes), dtype=object)
        self.v_sources.fill(lambda n: 0)

        
    def _get_number_of_state_vars_in_connector(self, connection):
        n_state = 0
        connector = connection[2]
        if (connector["connectorType"] == "Conn_sRLC"):
            if (connector["inductance"] != 0.0):
                n_state += 1
            if (connector["capacitance"] < 1e22):
                n_state += 1
        
        if (connector["connectorType"] == "Conn_C"):
            n_state += 1
        elif (connector["connectorType"] == "Conn_LCpRs"):
            n_state += 2
        
        if (connector["connectorType"] == "MultiwireConnector"):
            if (connector["connections"][connection[1]] == "Conn_C"):
                n_state += 1
            elif (connector["connections"][connection[1]] == "Conn_LCpRs"):
                n_state += 2

        return n_state

    def _get_number_of_state_vars(self, connections):
        n_state = 0
        for conductor, connection in enumerate(connections):
            connector = connection[2]
        
            if (connector["connectorType"] == "Conn_sRLC"):
                if (connector["inductance"] != 0.0):
                    n_state += 1
                if (connector["capacitance"] < 1e22):
                    n_state += 1
            
            if (connector["connectorType"] == "Conn_C"):
                n_state += 1
            
            if (connector["connectorType"] == "MultiwireConnector"):
                if (connector["connections"][conductor] == "Conn_C"):
                    n_state += 1
                elif (connector["connections"][conductor] == "Conn_LCpRs"):
                    n_state += 2
        return n_state
        
    def _add_nodes_in_line(self, nw_conn):
        assert (nw_conn[1]["side"] == "S" or nw_conn[1]["side"] == "L")
        
        if (nw_conn[1]["side"] == "S"):
            side_idx = 0
        elif (nw_conn[1]["side"] == "L"):
            side_idx = -1

        self.c = linalg.block_diag(self.c, nw_conn[0].c[side_idx])
        
        def order_by_conductor(conn):
            return conn[1]
        
        for connection in sorted(nw_conn[1]["connections"], key=order_by_conductor):
            self._add_node(connection, side_idx, nw_conn[0], nw_conn[1]["bundle"].name)
        
    def _add_node(self, connection: list, side_idx: int, line, bundle_name: str):
        assert (connection[0] >= 0)
        assert (connection[0] not in self.connections.keys())

        node_state_vars = self._get_number_of_state_vars_in_connector(connection)
        state_index = []
        if (node_state_vars):
            state_index = list( range(self.number_of_state_vars, self.number_of_state_vars+node_state_vars) )
        self.number_of_state_vars += node_state_vars

        index = self.nw_v.shape[0]
        self.nw_v = np.append(self.nw_v,0.0)
        self.nw_i = np.append(self.nw_i,0.0)
        self.dx = linalg.block_diag(self.dx, line.du_norm[side_idx][0,0])
        self.e_T = np.append(self.e_T, line.e_T[connection[1]][side_idx])

        self.connections[connection[0]] = {
            "line_name" : line.name, 
            "line_index" : connection[1], 
            "side" : side_idx, 
            "index": index, 
            "bundle_name" : bundle_name,
            "state_index" : state_index
        }
            
    def connect_to_ground(self, node: int, R = 0, Vt = 0):
        assert(node in self.connections.keys())
        index = self.connections[node]["index"]
        if (R != 0):
            self.P1[index, index] = -1/R
        if (Vt != 0):
            self.v_sources[index]  = Vt
            self.Ps[index, index] = 1/R

    def connect_to_ground_R(self, node: int, R = 0, Vt = 0):
        assert(node in self.connections.keys())
        index = self.connections[node]["index"]
        if (R != 0):
            self.P1[index, index] = -1/R
        if (Vt != 0):
            self.v_sources[index]  = Vt
            self.Ps[index, index] = 1/R

    def connect_to_ground_LCpRs(self, node: int, R = 0, L = 0, C = 0, Vt = 0):
        assert(node in self.connections.keys())
        index = self.connections[node]["index"]
        x_index = self.connections[node]["state_index"]
        assert (R >= 0 and L >= 0 and C<1e22)
        self.M[x_index[0]:x_index[-1]+1,x_index[0]:x_index[-1]+1] = np.array([[1/(R*C),1/C],[-1/L, 0.0]])
        self.N1[x_index[0]:x_index[-1]+1, index] = np.array([[-1/(R*C) ,0]])
        self.O[index, x_index[0]:x_index[-1]+1] = np.array([[1/R ,0]])
        self.P1[index, index] = -1/R

        if (Vt != 0):
            self.v_sources[index]  = Vt
            self.Ps[index, index] = 1/R
            self.Ns[x_index[0]:x_index[-1]+1, index] = np.array([[1/(R*C) ,0]])

    def connect_to_ground_C(self, node: int, C = 0, Vt = 0):
        assert(node in self.connections.keys())
        index = self.connections[node]["index"]
        if (C != 0):
            self.Q1[index, index] = -C
        if (Vt != 0):
            self.v_sources[index]  = Vt
            self.Qs[index, index] = C

    def short_to_ground(self, node: int):
        assert(node in self.connections.keys())
        index = self.connections[node]["index"]
        self.P1[index, index] = -1e10
            
    def connect_nodes(self, node1: int, node2: int, R, Vt = 0):
        assert(node1 in self.connections.keys() and node2 in self.connections.keys())
        assert(R != 0)
        index1 = self.connections[node1]["index"]
        index2 = self.connections[node2]["index"]
        # assert(self.P1[index1, index1] == 0)
        if (R != 0):
            self.P1[index1, index1] += -1/R 
            self.P1[index1, index2] += 1/R
            self.P1[index2, index1] += 1/R
            self.P1[index2, index2] += -1/R
            if (Vt != 0):
                self.Ps[index1, index1] +=  1/R 
                self.Ps[index2, index2] += -1/R
                
                self.v_sources[index1]  = add(self.v_sources[index1],Vt)
                self.v_sources[index2]  = add(self.v_sources[index2],Vt)
      
    def short_nodes(self, node1: int, node2: int):
        index1 = self.connections[node1]["index"]
        index2 = self.connections[node2]["index"]
        self.P1[index1, index1] +=  -1e10
        self.P1[index1, index2] +=   1e10
        self.P1[index2, index1] +=   1e10
        self.P1[index2, index2] +=  -1e10
      
    def update_sources(self, time, dt):
        self.v_sources_now = np.vectorize(FunctionType.__call__, otypes=["float64"])(self.v_sources, time)
        self.v_sources_prev = np.vectorize(FunctionType.__call__, otypes=["float64"])(self.v_sources, time - dt)
        self.e_T_now = np.vectorize(FunctionType.__call__, otypes=["float64"])(self.e_T,  time)
        self.e_T_prev = np.vectorize(FunctionType.__call__, otypes=["float64"])(self.e_T, time - dt)

        self.v_sources_prev2 = np.vectorize(FunctionType.__call__, otypes=["float64"])(self.v_sources, time - 2*dt)

        # self.v_sources_now = self.v_sources_now.reshape(self.v_sources_now.shape[0],1)
        # self.v_sources_prev =self.v_sources_prev.reshape(self.v_sources_prev.shape[0],1) 
        # self.v_sources_prev2 =self.v_sources_prev2.reshape(self.v_sources_prev2.shape[0],1) 
        # self.e_T_now = self.e_T_now.reshape(self.e_T_now.shape[0],1)
        # self.e_T_prev = self.e_T_prev.reshape(self.e_T_prev.shape[0],1)

    def advance_voltage(self, dt):
        try:
            self.Is = self.O.dot(self.X.reshape(self.number_of_state_vars,1)) + self.P1.dot(self.nw_v.reshape(self.number_of_nodes,1)) + self.Ps.dot(self.v_sources_prev[:,np.newaxis]) +\
                    self.Q1.dot(self.nw_v.reshape(self.number_of_nodes,1) - self.nw_v_prev.reshape(self.number_of_nodes,1))/dt +\
                    self.Qs.dot(self.v_sources_prev[:,np.newaxis] - self.v_sources_prev2[:,np.newaxis])/dt
        except:
            raise Exception("block fail")        
            
        try:
            self.B3 = np.block([
                [self.Is - self.Qs.dot(self.v_sources_prev)[:,np.newaxis]/dt  - 2*(self.nw_i.reshape(self.number_of_nodes,1)) - self.dx.dot(self.c).dot(self.e_T_now - self.e_T_prev)[:,np.newaxis]/dt],
                [ np.zeros([self.number_of_state_vars,1])]
            ])
        except:
            raise Exception("block fail")        

        self.nw_v_prev = self.nw_v
        # assert( (self.B1.dot(self.H[:,np.newaxis]) + self.B2.dot(self.v_sources_now[:,np.newaxis]) + self.B3).shape[1] == self.H.shape[0])
        self.H = np.linalg.solve(self.A, self.B1.dot(self.H.reshape(self.number_of_nodes + self.number_of_state_vars,1)) + self.B2.dot(self.v_sources_now[:,np.newaxis]) + self.B3)
        self.nw_v = self.H[0:self.number_of_nodes]
        self.nw_v_prev.shape = self.nw_v.shape
        self.X = self.H[self.number_of_nodes:]

    # def advance_voltage(self):
    #     self.nw_v = self.terminal_term_1.dot(self.nw_v) +\
    #                 self.terminal_term_2.dot(self.Ps.dot(self.v_sources_now + self.v_sources_prev) - 2*self.nw_i)+\
    #                 self.terminal_term_3.dot(self.e_T_now - self.e_T_prev)
       
    def update_voltages(self, lines):
        for node in self.connections.values():
            lines[node["bundle_name"]].v[node["line_index"],node["side"]] = self.nw_v[node["index"]]

    def i_factor(self,node):
        if (node["side"] == 0):
            return 1.0
        elif (node["side"] == -1):
            return -1.0

    def update_currents(self, lines):
        for node in self.connections.values():
            self.nw_i[node["index"]] =  self.i_factor(node)*lines[node["bundle_name"]].i[node["line_index"],node["side"]]

    def compute_v_terms(self, dt):
        self.A = np.block([
            [self.dx.dot(self.c) / dt - self.P1 - self.Q1 / dt     ,  -self.O                            ],
            [-dt * self.N1                                         ,  np.eye(self.M.shape[0]) - dt*self.M]
        ])
        self.B1 = np.block([
            [self.dx.dot(self.c) / dt - self.Q1 / dt   ,   np.zeros(self.O.shape)],
            [np.zeros(self.N1.shape)                   ,   np.eye(self.M.shape[0])]
        ])
        self.B2 = np.block([
            [self.Ps + self.Qs / dt],
            [dt * self.Ns          ]
        ])

        # inv = np.linalg.inv(self.dx.dot(self.c) / dt - self.P1)
        # self.terminal_term_1 = inv.dot(self.dx.dot(self.c) / dt + self.P1)
        # self.terminal_term_2 = inv
        # self.terminal_term_3 = -inv.dot(self.dx).dot(self.c)/dt
       
class NetworkD:
    """
    Networks can be joining tubes (junctions) or ending tubes (terminations)
    """
    def __init__(self, levels : Dict[int, Network]):
        self.levels = levels

    def update_index_numbers(self, bundles):
        for level, nw in self.levels.items():
            for node in nw.connections.values():
                bundle = bundles[node["bundle_name"]]
                conductors_above = sum(bundle.conductors_in_level[0:level])
                node["line_index"] += conductors_above

    def update_sources(self, time, dt):
        for nw in self.levels.values():
            nw.update_sources(time, dt)

    def advance_voltage(self, dt):
        for nw in self.levels.values():
            nw.advance_voltage(dt)

    def update_voltages(self, bundles):
        for nw in self.levels.values():
            nw.update_voltages(bundles)

    def update_currents(self, bundles):
        for nw in self.levels.values():
            nw.update_currents(bundles)

    def compute_v_terms(self, dt):
        for nw in self.levels.values():
            nw.compute_v_terms(dt)

