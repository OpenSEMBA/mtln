import numpy as np
import skrf as rf
import bigtree as tree

from scipy import linalg

from types import FunctionType
from types import LambdaType

from collections import OrderedDict

import src.mtl as mtl


class MTLND:
    """
    Lossless Multiconductor Transmission Line Network with subdomains
    """

    def __init__(self, l0_mtl, level_number):
        assert(level_number == 0)
        self.Zt = np.ndarray(shape=(0,0))
        self.levels = {0:l0_mtl}
        self.conductors_in_level = {}
        self.nz = l0_mtl[0].x.shape[0]
        
        for line in l0_mtl:
            assert(type(line) == mtl)
            self.conductors_in_level[level_number] += line.l.shape[1]
            assert(self.nz == line.x.shape[0])
        
    def add_level(self, level_number:int):
        assert (level_number == self.current_level)
        self.current_level += 1

    def add_mtl(self, line, level_number):
        assert (self.nz == line.x.shape[0])
        
        if (level_number in self.levels.keys()):
            self.levels[level_number]["mtl"].append(line)
        else:
            self.levels[level_number] = {"mtl" : [line]}
            
        self.conductors_in_level[level_number] += line.l.shape[1]

    def number_of_conductors(self):
        nc = 0
        for n in self.conductors_in_level.values():
            nc += n
        return nc
    
    def build_pul_matrices(self):
        nc = self.number_of_conductors()
        self.L = np.ndarray(shape=(self.nz-1, nc, nc))
        self.C = np.ndarray(shape=(self.nz, nc, nc))
        i = 0
        out_in_levels = OrderedDict((self.levels.items())).items()
        for k, v in out_in_levels:
            for line in v:
                n_cond = line.l.shape[1]
                for k in range(self.nz):
                    self.L[k,i:i+n_cond,i:i+n_cond] = line.mtl.l[k,:,:]
                    self.C[k,i:i+n_cond,i:i+n_cond] = line.mtl.c[k,:,:]
                self.C[self.nz,i:i+n_cond,i:i+n_cond] = line.mtl.c[self.nz,:,:]

                i += n_cond

            
    def get_conductors_in_level(self, level):
        n = 0
        for line in self.levels[level]:
            n += line.l.shape[1]

    def add_transfer_impedance(self, out_level, in_level, zt):
        assert(np.abs(out_level - in_level) == 1)
        assert(zt.shape == [self.conductors_in_level[out_level], self.conductors_in_level[in_level]])

