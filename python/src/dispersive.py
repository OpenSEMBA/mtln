import numpy as np


class Dispersive():
    def __init__(self, number_of_conductors, nx, u, dt):
        self.nx = nx
        self.u = u
        self.dt = dt
        
        self.phi = np.zeros(
            shape=(
                nx,
                number_of_conductors
            ), dtype = np.ndarray
        )
        
        self.q1 = np.zeros(
            shape=(
                nx,
                number_of_conductors,
                number_of_conductors
            ), dtype = np.ndarray
        )
        
        self.q2 = np.zeros(
            shape=(
                nx,
                number_of_conductors,
                number_of_conductors
            ), dtype = np.ndarray
        )
        self.q3 = np.zeros(
            shape=(
                nx,
                number_of_conductors,
                number_of_conductors
            ), dtype = np.ndarray
        )
        self.q3_phi_term = np.zeros(
            shape=(
                nx,
                number_of_conductors
            )
        )
        self.d = np.zeros(
            shape=(
                nx,
                number_of_conductors,
                number_of_conductors,
            )
        )
        self.e = np.zeros(
            shape=(
                nx,
                number_of_conductors,
                number_of_conductors,
            )
        )
        self.q1sum = np.zeros(
            shape=(
                nx,
                number_of_conductors,
                number_of_conductors,
            )
        )
        self.q2sum = np.zeros(
            shape=(
                nx,
                number_of_conductors,
                number_of_conductors,
            )
        )
    
    def v_sum(self,arr:np.ndarray):
        return np.vectorize(np.sum, otypes=[float])(arr)

    def update_q3_phi_term(self):
        for kz in range(0, self.nx):
            self.q3_phi_term[kz] = self.v_sum(self.q3[kz].dot(self.phi[kz]))
        
    def update_phi(self, i_prev, i_now):
        for kz in range(0, self.nx):
            self.phi[kz, :] = (
                self.q1[kz, :, :].dot(i_now[:, kz])+\
                self.q2[kz, :, :].dot(i_prev[:, kz])+\
                self.q3[kz, :,:].dot(self.phi[kz, :])
            )

        

class DispersiveConnector(Dispersive):
    def __init__(self, number_of_conductors, nx, u, dt):
        super().__init__(number_of_conductors, nx, u, dt)

    def add_dispersive_connector(
        self,
        position : np.ndarray,
        conductor,
        model :dict 
    ):
        if (np.any(position > self.u[-1])) or np.any((position < self.u[0])):
            raise ValueError("Connector position is out of MTL length.")

        index = np.argmin(np.apply_along_axis(np.linalg.norm, 1, np.abs(self.u - position)))
        du = np.linalg.norm(self.u[index+1]-self.u[index])
        
        d = model["cte"]/du
        e = model["prop"]/du
        
        assert (len(model["poles"]["real"]) == len(model["poles"]["imag"]))
        assert (len(model["residues"]["real"]) == len(model["residues"]["imag"]))
        assert (len(model["poles"]["real"]) == len(model["residues"]["real"]))
        assert (len(model["poles"]["imag"]) == len(model["residues"]["imag"]))
        
        n_poles = len(model["poles"]["real"])
        poles = np.ndarray(shape=n_poles,dtype=complex)
        residues = np.ndarray(shape=n_poles,dtype=complex)
        for i in range(n_poles):
            poles[i] = model["poles"]["real"][i] + 1j*model["poles"]["imag"][i]
            residues[i] = (model["residues"]["real"][i] + 1j*model["residues"]["imag"][i])/du


        assert self.d[index, conductor, conductor]  == 0.0, f"Dispersive connector already in conductor {conductor} at position {index}"
        assert self.e[index, conductor, conductor]  == 0.0, f"Dispersive connector already in conductor {conductor} at position {index}"
        assert self.q1[index, conductor, conductor] == 0.0, f"Dispersive connector already in conductor {conductor} at position {index}"
        assert self.q2[index, conductor, conductor] == 0.0, f"Dispersive connector already in conductor {conductor} at position {index}"
        assert self.q3[index, conductor, conductor] == 0.0, f"Dispersive connector already in conductor {conductor} at position {index}"


        self.d[index, conductor, conductor] += d
        self.e[index, conductor, conductor] += e


        self.q1[index, conductor, conductor] -= (residues / poles) * (
            1 - (np.exp(poles * self.dt) - 1) / (poles * self.dt)
        )
        self.q2[index, conductor, conductor] += (residues / poles) * (
            1 / (poles * self.dt)
            + np.exp(poles * self.dt) * (1 - 1 / (poles * self.dt))
        )
        self.q3[index, conductor, conductor] += np.exp(poles * self.dt)

        self.q1sum = self.v_sum(self.q1)
        self.q2sum = self.v_sum(self.q2)


class TransferImpedance(Dispersive):
    def __init__(self, number_of_conductors, nx, u, dt):
        super().__init__(number_of_conductors, nx, u, dt)

    def add_transfer_impedance(
        self, 
        levels,
        out_level, out_level_conductors,
        in_level, in_level_conductors,
        transfer_impedance
    ):
        assert(np.abs(out_level - in_level) == 1)
        
        factor = 1
        if "factor" in transfer_impedance.keys():
            factor = transfer_impedance["factor"]

        d = transfer_impedance["cte"]/factor
        e = transfer_impedance["prop"]/factor

        assert (len(transfer_impedance["poles"]["real"]) == len(transfer_impedance["poles"]["imag"]))
        assert (len(transfer_impedance["residues"]["real"]) == len(transfer_impedance["residues"]["imag"]))
        assert (len(transfer_impedance["poles"]["real"]) == len(transfer_impedance["residues"]["real"]))
        assert (len(transfer_impedance["poles"]["imag"]) == len(transfer_impedance["residues"]["imag"]))
        
        n_poles = len(transfer_impedance["poles"]["real"])
        poles = np.ndarray(shape=n_poles,dtype=complex)
        residues = np.ndarray(shape=n_poles,dtype=complex)
        for i in range(n_poles):
            poles[i] = transfer_impedance["poles"]["real"][i] + 1j*transfer_impedance["poles"]["imag"][i]
            residues[i] = (transfer_impedance["residues"]["real"][i] + 1j*transfer_impedance["residues"]["imag"][i])/factor

        
        if d == 0 and e == 0 and len(poles) == 0 and len(residues) == 0:
            return

        n_before_out = sum(levels[0:out_level])
        n_before_in = sum(levels[0:in_level])
        
        range_out = n_before_out + np.array(out_level_conductors)
        range_in  = n_before_in  + np.array(in_level_conductors)

        direction = transfer_impedance["direction"]

        for i in range_out:
            for j in range_in:

                if (direction == ("in" or "both")):
                    self.d[:, j, i] -= d
                    self.e[:, j, i] -= e
                    if (residues.size != 0 and poles.size != 0):
                        self.q1[:, j, i] += (residues / poles) * (
                            1 - (np.exp(poles * self.dt) - 1) / (poles * self.dt)
                        )
                        self.q2[:, j, i] -= (residues / poles) * (
                            1 / (poles * self.dt)
                            + np.exp(poles * self.dt) * (1 - 1 / (poles * self.dt))
                        )
                        self.q3[:, j, i] -= np.exp(poles * self.dt)
                        
                if (direction == ("out" or "both")):
                    self.d[:, i, j] -= d
                    self.e[:, i, j] -= e
                    if (residues.size != 0 and poles.size != 0):
                        self.q1[:, i, j] += (residues / poles) * (
                            1 - (np.exp(poles * self.dt) - 1) / (poles * self.dt)
                        )
                        self.q2[:, i, j] -= (residues / poles) * (
                            1 / (poles * self.dt)
                            + np.exp(poles * self.dt) * (1 - 1 / (poles * self.dt))
                        )
                        self.q3[:, i, j] -= np.exp(poles * self.dt)

        self.q1sum = self.v_sum(self.q1)
        self.q2sum = self.v_sum(self.q2)

    def set_transfer_impedance_at_index(
        self, 
        levels,
        out_level, out_level_conductors,
        in_level, in_level_conductors,
        index: int,
        transfer_impedance
    ):
        assert(np.abs(out_level - in_level) == 1)
        
        factor = 1
        if "factor" in transfer_impedance.keys():
            factor = transfer_impedance["factor"]

        d = transfer_impedance["cte"]/factor
        e = transfer_impedance["prop"]/factor

        assert (len(transfer_impedance["poles"]["real"]) == len(transfer_impedance["poles"]["imag"]))
        assert (len(transfer_impedance["residues"]["real"]) == len(transfer_impedance["residues"]["imag"]))
        assert (len(transfer_impedance["poles"]["real"]) == len(transfer_impedance["residues"]["real"]))
        assert (len(transfer_impedance["poles"]["imag"]) == len(transfer_impedance["residues"]["imag"]))
        
        n_poles = len(transfer_impedance["poles"]["real"])
        poles = np.ndarray(shape=n_poles,dtype=complex)
        residues = np.ndarray(shape=n_poles,dtype=complex)
        for i in range(n_poles):
            poles[i] = transfer_impedance["poles"]["real"][i] + 1j*transfer_impedance["poles"]["imag"][i]
            residues[i] = (transfer_impedance["residues"]["real"][i] + 1j*transfer_impedance["residues"]["imag"][i])/factor
       
        if d == 0 and e == 0 and len(poles) == 0 and len(residues) == 0:
            return

        n_before_out = sum(levels[0:out_level])
        n_before_in = sum(levels[0:in_level])
        
        range_out = n_before_out + np.array(out_level_conductors)
        range_in  = n_before_in  + np.array(in_level_conductors)

        direction = transfer_impedance["direction"]

        for i in range_out:
            for j in range_in:

                if (direction == ("in" or "both")):
                    self.d[index, j, i] = -d
                    self.e[index, j, i] = -e
                    if (residues.size != 0 and poles.size != 0):
                        self.q1[index, j, i] = (residues / poles) * (
                            1 - (np.exp(poles * self.dt) - 1) / (poles * self.dt)
                        )
                        self.q2[index, j, i] = -(residues / poles) * (
                            1 / (poles * self.dt)
                            + np.exp(poles * self.dt) * (1 - 1 / (poles * self.dt))
                        )
                        self.q3[index, j, i] = -np.exp(poles * self.dt)
                        
                if (direction == ("out" or "both")):
                    self.d[index, i, j] = -d
                    self.e[index, i, j] = -e
                    if (residues.size != 0 and poles.size != 0):
                        self.q1[index, i, j] = (residues / poles) * (
                            1 - (np.exp(poles * self.dt) - 1) / (poles * self.dt)
                        )
                        self.q2[index, i, j] = -(residues / poles) * (
                            1 / (poles * self.dt)
                            + np.exp(poles * self.dt) * (1 - 1 / (poles * self.dt))
                        )
                        self.q3[index, i, j] = -np.exp(poles * self.dt)

        self.q1sum = self.v_sum(self.q1)
        self.q2sum = self.v_sum(self.q2)
