import numpy as np
import src.mtl as mtl

class MTLN:
    """
    Multiconductor Transmission Line Network 
    Works as a manager. Can hold MTLs, MTLs with losses, MTL with levels.
    Networks can be single or tiered
    """
    def __init__(self, bundles = [], networks = []):
        self.bundles = {}
        self.networks = []
        self.dt = 1e10
        self.time = 0.0
        self.du = {}
        
        for b in bundles:
            self.add_bundle(b)
        for nw in networks:
            self.add_network(nw)
            

    def add_bundle(self, bundle):
        assert(bundle.name not in self.bundles.keys())
        # if (self.dx == 0):
        #     self.dx = bundle.dx        
        self.du[bundle.name] = bundle.du
        self.bundles[bundle.name] = bundle
        if (bundle.dt < self.dt):
            self.dt = bundle.dt
        
    def add_network(self, nw):
        self.networks.append(nw)
        nw.update_index_numbers(self.bundles)

    def add_networks(self, nw: list):
        for n in nw:
            self.networks.append(n)
        
    def get_time_range(self, final_time):
        return np.arange(0, np.floor(final_time / self.dt))

    def compute_nw_v_terms(self):
        for nw in self.networks:
            nw.compute_v_terms(self.dt)

    def update_probes(self):
        for bundle in self.bundles.values():
            for p in bundle.probes:
                p.update(self.time, bundle.v, bundle.i)

    def add_planewave(self, pw, distances):
        for bundle in self.bundles.values():
            bundle.add_planewave(pw, distances)

    def add_external_field(self, field: mtl.Field, distances):
        for bundle in self.bundles.values():
            bundle.add_external_field(field, distances)


    def run_until(self, finalTime):
        
        self.compute_nw_v_terms()
        
        t = self.get_time_range(finalTime)
        for bundle in self.bundles.values():
            bundle.update_lr_terms()
            bundle.update_cg_terms()
            for p in bundle.probes:
                p.resize_frames(len(t), bundle.number_of_conductors)

        for _ in t:
            self.step()

    def update_networks_current(self):
        for nw in self.networks:
            nw.update_currents(self.bundles)

    def advance_networks_voltage(self):
        for nw in self.networks:
            nw.update_sources(self.time, self.dt)
            nw.advance_voltage()
            nw.update_voltages(self.bundles)
    
    def advance_bundles_voltage(self):
        for bundle in self.bundles.values():
            bundle.update_sources(self.time, self.dt)
            bundle.advance_voltage()

    def advance_bundles_current(self):
        for bundle in self.bundles.values():
            bundle.advance_current()

    def advance_time(self):
        self.time += self.dt


    def step(self):
        self.advance_bundles_voltage()
        self.advance_networks_voltage()
        self.advance_bundles_current()
        self.update_networks_current()

        self.advance_time()
        self.update_probes()