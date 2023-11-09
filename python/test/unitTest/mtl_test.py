import numpy as np
import src.mtl as mtl

def test_get_phase_velocities():
    v = mtl.MTL(l=0.25e-6, c=100e-12, ndiv=100).get_phase_velocities()
    assert v.shape == (100,1)
    assert np.isclose(2e8, v[:][0])

    v = mtl.MTL(l=0.25e-6, c=100e-12, ndiv=50).get_phase_velocities()
    assert v.shape == (50,1)
    assert np.isclose(2e8, v[:][0])

def test_symmetry_in_voltage_excitation():
    """ 
    Test results are identical when exciting from S or from L.
    """
    def magnitude(t): return wf.square_pulse(t, 100, 6e-6)
    finalTime = 18e-6

    line_vs = mtl.MTL(l=0.25e-6, c=100e-12, length=400.0, Zs=150, Zl=150)
    line_vs.add_voltage_source(position=0.0, conductor=0, magnitude=magnitude)
    vs_probe = line_vs.add_probe(position=0.0, probe_type='voltage')
    line_vs.run_until(finalTime)

    line_vl = mtl.MTL(l=0.25e-6, c=100e-12, length=400.0, Zs=150, Zl=150)
    line_vl.add_voltage_source(position=400.0, conductor=0, magnitude=magnitude)
    vl_probe = line_vl.add_probe(position=400.0, probe_type='voltage')
    line_vl.run_until(finalTime)

    assert np.all(vl_probe.val == vs_probe.val)
    
    # plt.figure()
    # plt.plot(vs_probe.t, vs_probe.val)
    # plt.plot(vl_probe.t, vl_probe.val)
    # plt.show()
