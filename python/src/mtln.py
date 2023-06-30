import numpy as np
import skrf as rf

from copy import deepcopy
from numpy.fft import fft, fftfreq, fftshift
import sympy as sp

import src.mtl as mtl
import scipy.linalg as linalg

from types import FunctionType
from types import LambdaType

class MTLN:
    """
    Multiconductor Transmission Line Network 
    Works as a manager. Can hold MTLs, MTLs with losses, MTL with levels.
    Networks can be single or tiered
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
        
    def add_network(self, nw):
        self.networks[nw.nw_number] = nw
        
    def get_time_range(self, final_time):
        return np.arange(0, np.floor(final_time / self.dt))

    def compute_nw_v_terms(self):
        for nw in self.networks.values():
            nw.compute_v_terms(self.dx, self.dt)

    def update_probes(self):
        for bundle in self.bundles.values():
            for p in bundle.probes:
                p.update(self.time, bundle.x, bundle.v, bundle.i)

    def run_until(self, finalTime):
        
        self.compute_nw_v_terms()
        
        t = self.get_time_range(finalTime)
        for bundle in self.bundles.values():
            for p in bundle.probes:
                p.resize_frames(len(t), bundle.number_of_conductors)

        for _ in t:
            self.step()

    def update_networks_current(self):
        for nw in self.networks.values():
            nw.update_currents(self.bundles)

    def advance_networks_voltage(self):
        for nw in self.networks.values():
            nw.advance_voltage(self.time, self.dt)
            nw.update_voltages(self.bundles)
    
    def advance_bundles_voltage(self):
        for bundle in self.bundles.values():
            bundle.update_sources()
            bundle.advance_voltage()

    def advance_bundles_current(self):
        for bundle in self.bundles.values():
            bundle.advance_current()

    def step(self):
        self.advance_bundles_voltage()
        self.advance_networks_voltage()
        self.advance_bundles_current()
        self.update_networks_current()

        self.time += self.dt
        self.update_probes()