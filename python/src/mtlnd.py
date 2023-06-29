import numpy as np
import skrf as rf
import bigtree as tree

from scipy import linalg

from types import FunctionType
from types import LambdaType

from collections import OrderedDict

import src.mtl as mtl
import src.mtln as mtln

class MTLND:
    """
    Lossless Multiconductor Transmission Line Network with subdomains
    """

    def __init__(self, line, level_number):
        assert(level_number == 0)
        assert(type(line) == mtl.MTL)

        self.levels = {0:{"mtl" :[line], "conductors":line.l.shape[1]}}
        self.conductors_in_level = {}
        self.nz = line.x.shape[0]
        # self.conductors_in_level[level_number] = line.l.shape[1]
        self.transfer_impedance = {}
        
        self.networks = {}

    def add_network(self, nw: mtln.Network, level: int):
        if (nw.nw_number in self.networks.keys()):
            assert (level not in self.networks[nw.nw_number].keys())
            self.networks[nw.nw_number][level] = nw
        else:
            self.networks[nw.nw_number] = {level : nw}
        
            
    def add_mtl(self, line, level_number):
        assert (self.nz == line.x.shape[0])
        
        if (level_number in self.levels.keys()):
            self.levels[level_number]["mtl"].append(line)
            self.levels[level_number]["conductors"] += line.l.shape[1]
        else:
            self.levels[level_number] = {"mtl" :[line], "conductors":line.l.shape[1]}

    def number_of_conductors(self):
        nc = 0
        for _, level in self.levels.items():
            nc += level["conductors"]
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

            
    def get_conductors_in_levels(self):
        return [value["conductors"] for value in self.levels.values()]

    def build_transfer_impedance_matrix(self):
        nc = self.number_of_conductors()
        self.Zt = np.zeros([nc,nc])

    #bi, uni, in-out        
    def add_transfer_impedance(self, out_level, in_level, zt):
        assert(np.abs(out_level - in_level) == 1)
        n_out = self.levels[out_level]["conductors"]
        n_in = self.levels[in_level]["conductors"]
        assert(zt.shape == (n_out, n_in))

        n_before_out = sum(self.get_conductors_in_levels()[0:out_level])
        n_before_in =  sum(self.get_conductors_in_levels()[0:in_level])
        o1 = n_before_out
        o2 = n_before_out + n_out
        i1 = n_before_in
        i2 = n_before_in + n_in

        self.Zt[o1:o2,i1:i2] = zt
        self.Zt[i1:i2,o1:o2] = np.transpose(zt)


    #unmodified
    def run_until(self, finalTime):
        
        self.compute_v_terms()
        
        t = self.get_time_range(finalTime)
        for bundle in self.bundles.values():
            for p in bundle.probes:
                p.resize_frames(len(t), bundle.number_of_conductors)

        for _ in t:
            self.step()

    """
    Step has to evolve:
    * each bundle using the coupled equation (with possibility of ext.field)
    * each network, which can evolve level by level: loop over levels, 
    update them, keep track of which node belogns to which conductor in which level
    * assign voltages/currents between terminals and bundles as in mltn
    * anything else
    """
    def step():
        pass    
    