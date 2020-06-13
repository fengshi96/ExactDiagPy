import numpy as np
import scipy.sparse as sp


class Dofs:

    def __init__(self, spin="SpinHalf"):
        if spin == "SpinHalf":

            self.hilbsize = 2
            self.type = "SpinHalf"

            I = np.array([0, 1], dtype=int)
            J = np.array([1, 0], dtype=int)
            V = np.array([0.5, 0.5], dtype=complex)
            self.Sx = sp.coo_matrix((V, (I, J)))

            I = np.array([0, 1], dtype=int)
            J = np.array([1, 0], dtype=int)
            V = np.array([-0.5j, 0.5j], dtype=complex)
            self.Sy = sp.coo_matrix((V, (I, J)))

            I = np.array([0, 1], dtype=int)
            J = np.array([0, 1], dtype=int)
            V = np.array([0.5, -0.5], dtype=complex)
            self.Sz = sp.coo_matrix((V, (I, J)))

            I = np.array([0], dtype=int)
            J = np.array([1], dtype=int)
            V = np.array([1], dtype=complex)
            self.Sp = sp.coo_matrix((V, (I, J)))

            I = np.array([1], dtype=int)
            J = np.array([0], dtype=int)
            V = np.array([1], dtype=complex)
            self.Sm = sp.coo_matrix((V, (I, J)))

            self.I = sp.eye(2)

        elif spin == "SpinOne":

            self.hilbsize = 3
            self.type = "SpinOne"

            I = np.array([0, 1, 1, 2], dtype=int)
            J = np.array([1, 0, 2, 1], dtype=int)
            V = np.array([1, 1, 1, 1], dtype=complex) * np.sqrt(0.5)
            self.Sx = sp.coo_matrix((V, (I, J)))

            I = np.array([0, 1, 1, 2], dtype=int)
            J = np.array([1, 0, 2, 1], dtype=int)
            V = np.array([-1j, 1j, -1j, 1j], dtype=complex) * np.sqrt(0.5)
            self.Sy = sp.coo_matrix((V, (I, J)))

            I = np.array([0, 2], dtype=int)
            J = np.array([0, 2], dtype=int)
            V = np.array([1, -1], dtype=complex)
            self.Sz = sp.coo_matrix((V, (I, J)))

            I = np.array([0, 1], dtype=int)
            J = np.array([1, 2], dtype=int)
            V = np.array([1, 1], dtype=complex) * np.sqrt(2)
            self.Sp = sp.coo_matrix((V, (I, J)))

            I = np.array([1, 2], dtype=int)
            J = np.array([0, 1], dtype=int)
            V = np.array([1, 1], dtype=complex) * np.sqrt(2)
            self.Sm = sp.coo_matrix((V, (I, J)))

            self.I = sp.eye(3)

        else:
            raise TypeError("Dof type not yet supported..")

# Spins = Dofs("SpinOne")
# print(Spins.Sx, "\n")
# print(Spins.Sy, "\n")
# print(Spins.Sz, "\n")
# print(Spins.Sp, "\n")
# print(Spins.Sm, "\n")
#
# print(Spins.Sz * Spins.Sz)

#
# print(Spins.Sx.shape[0])
