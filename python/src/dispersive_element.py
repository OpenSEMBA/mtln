import numpy as np

class TransferImpedance():
    def __init__(self, number_of_conductors, nx):
    
        self.phi = np.zeros(
            shape=(
                nx-1,
                number_of_conductors
            ), dtype = np.ndarray
        )
        
        self.q1 = np.zeros(
            shape=(
                nx-1,
                number_of_conductors,
                number_of_conductors
            ), dtype = np.ndarray
        )
        
        self.q2 = np.zeros(
            shape=(
                nx-1,
                number_of_conductors,
                number_of_conductors
            ), dtype = np.ndarray
        )
        self.q3 = np.zeros(
            shape=(
                nx-1,
                number_of_conductors,
                number_of_conductors
            ), dtype = np.ndarray
        )
        self.q3_phi_term = np.zeros(
            shape=(
                nx-1,
                number_of_conductors
            )
        )
        self.d = np.zeros(
            shape=(
                nx-1,
                number_of_conductors,
                number_of_conductors,
            )
        )
        self.e = np.zeros(
            shape=(
                nx-1,
                number_of_conductors,
                number_of_conductors,
            )
        )
        self.q1sum = np.zeros(
            shape=(
                nx-1,
                number_of_conductors,
                number_of_conductors,
            )
        )
        self.q2sum = np.zeros(
            shape=(
                nx-1,
                number_of_conductors,
                number_of_conductors,
            )
        )

    def add_transfer_impedance(
        self, 
        out_level, out_conductor,
        in_level, i_conductor,
        d: float,
        e: float,
        poles: np.ndarray,
        residues: np.ndarray
    ):
        
    
    