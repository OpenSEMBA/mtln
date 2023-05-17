import numpy as np
import skrf as rf

from copy import deepcopy
from numpy.fft import fft, fftfreq, fftshift
import sympy as sp

import src.mlt as mtl

from types import FunctionType
from types import LambdaType


# class Node:
#     def __init__(self, network: Network):
#         self.connections = {}

class Network:
    """
    Networks can be joining tubes (junctions) or ending tubes (terminations)
    """
    def __init__(self, nodes: list[int]):
        self.number_of_nodes = len(nodes)
        self.connections = {}
    
    def add_connection(self,nw_node, tube:mtl.mtl, conductor: int):
        self.connections[nw_node] = [mtl,conductor]
        


        

class MTLN:
    """
    Lossless Multiconductor Transmission Line Network 
    """
    def __init__(self, l, c, g, r, length=1.0, nx=100, Zs=0.0, Zl=0.0):
        super().__init__(l, c, length=length, nx=nx, Zs=Zs, Zl=Zl)


    #the step function has to be redefined, to be compatible with the
    #network structure and update