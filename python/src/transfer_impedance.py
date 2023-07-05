import numpy as np

class TransferImpedance():
    def __init__(self, number_of_conductors, nx):
        #nx is the shape of I
        self.nx = nx
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

    def add_transfer_impedance(
        self, 
        levels,
        out_level, out_conductor,
        in_level, i_conductor,
        d: float,
        e: float,
        poles: np.ndarray,
        residues: np.ndarray
    ):
        
        assert(np.abs(out_level - in_level) == 1)
        n_out = levels[out_level]["conductors"]
        n_in = levels[in_level]["conductors"]

        n_before_out = sum([value["conductors"] for value in levels.values()][0:out_level])
        n_before_in = sum([value["conductors"] for value in levels.values()][0:in_level])
        
        o1, o2 = n_before_out, n_before_out + n_out
        i1, i2 = n_before_in, n_before_in + n_in

        self.d[:, o1:o2, i1:i2] = d
        self.e[:, o1:o2, i1:i2] = e

        self.q1[:, o1:o2, i1:i2] = -(residues / poles) * (
            1 - (np.exp(poles * self.dt) - 1) / (poles * self.dt)
        )
        self.q2[:, o1:o2, i1:i2] = (residues / poles) * (
            1 / (poles * self.dt)
            + np.exp(poles * self.dt) * (1 - 1 / (poles * self.dt))
        )
        self.q3[:, o1:o2, i1:i2] = np.exp(poles * self.dt)

        self.q1sum = self.v_sum(self.q1)
        self.q2sum = self.v_sum(self.q2)
    
    def v_sum(self,arr:np.ndarray): 
        return np.vectorize(np.sum)(arr)

    #nz =  self.i.shape[1]
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
