import numpy as np
import scipy.sparse as sp


class Dofs:

    def __init__(self, dof="SpinHalf", boson_dim=0):
        if dof == "SpinHalf":

            self.hilbsize = 2
            self.type = "SpinHalf"

            I = np.array([0, 1], dtype=int)
            J = np.array([1, 0], dtype=int)
            V = np.array([0.5, 0.5], dtype=complex)
            self.Sx = sp.coo_matrix((V, (I, J)), shape=(2, 2))

            I = np.array([0, 1], dtype=int)
            J = np.array([1, 0], dtype=int)
            V = np.array([-0.5j, 0.5j], dtype=complex)
            self.Sy = sp.coo_matrix((V, (I, J)), shape=(2, 2))

            I = np.array([0, 1], dtype=int)
            J = np.array([0, 1], dtype=int)
            V = np.array([0.5, -0.5], dtype=complex)
            self.Sz = sp.coo_matrix((V, (I, J)), shape=(2, 2))

            I = np.array([0], dtype=int)
            J = np.array([1], dtype=int)
            V = np.array([1], dtype=complex)
            self.Sp = sp.coo_matrix((V, (I, J)), shape=(2, 2))

            I = np.array([1], dtype=int)
            J = np.array([0], dtype=int)
            V = np.array([1], dtype=complex)
            self.Sm = sp.coo_matrix((V, (I, J)), shape=(2, 2))

            self.I = sp.eye(2)

        elif dof == "SpinOne":

            self.hilbsize = 3
            self.type = "SpinOne"

            I = np.array([0, 1, 1, 2], dtype=int)
            J = np.array([1, 0, 2, 1], dtype=int)
            V = np.array([1, 1, 1, 1], dtype=complex) * np.sqrt(0.5)
            self.Sx = sp.coo_matrix((V, (I, J)), shape=(3, 3))

            I = np.array([0, 1, 1, 2], dtype=int)
            J = np.array([1, 0, 2, 1], dtype=int)
            V = np.array([-1j, 1j, -1j, 1j], dtype=complex) * np.sqrt(0.5)
            self.Sy = sp.coo_matrix((V, (I, J)), shape=(3, 3))

            I = np.array([0, 2], dtype=int)
            J = np.array([0, 2], dtype=int)
            V = np.array([1, -1], dtype=complex)
            self.Sz = sp.coo_matrix((V, (I, J)), shape=(3, 3))

            I = np.array([0, 1], dtype=int)
            J = np.array([1, 2], dtype=int)
            V = np.array([1, 1], dtype=complex) * np.sqrt(2)
            self.Sp = sp.coo_matrix((V, (I, J)), shape=(3, 3))

            I = np.array([1, 2], dtype=int)
            J = np.array([0, 1], dtype=int)
            V = np.array([1, 1], dtype=complex) * np.sqrt(2)
            self.Sm = sp.coo_matrix((V, (I, J)), shape=(3, 3))

            self.I = sp.eye(3)

        elif dof == "SpinThreeHalf":

            self.hilbsize = 4
            self.type = "SpinThreeHalf"

            I = np.array([0, 1, 1, 2, 2, 3], dtype=int)
            J = np.array([1, 0, 2, 1, 3, 2], dtype=int)
            V = np.array([np.sqrt(3), np.sqrt(3), 2, 2, np.sqrt(3), np.sqrt(3)], dtype=complex) * 0.5
            self.Sx = sp.coo_matrix((V, (I, J)), shape=(4, 4))

            I = np.array([0, 1, 1, 2, 2, 3], dtype=int)
            J = np.array([1, 0, 2, 1, 3, 2], dtype=int)
            V = np.array([np.sqrt(3), -np.sqrt(3), 2, -2, np.sqrt(3), -np.sqrt(3)], dtype=complex) * -0.5j
            self.Sy = sp.coo_matrix((V, (I, J)), shape=(4, 4))

            I = np.array([0, 1, 2, 3], dtype=int)
            J = np.array([0, 1, 2, 3], dtype=int)
            V = np.array([1.5, 0.5, -0.5, -1.5], dtype=complex)
            self.Sz = sp.coo_matrix((V, (I, J)), shape=(4, 4))

            I = np.array([0, 1, 2], dtype=int)
            J = np.array([1, 2, 3], dtype=int)
            V = np.array([np.sqrt(3), 2, np.sqrt(3)], dtype=complex)
            self.Sp = sp.coo_matrix((V, (I, J)), shape=(4, 4))

            I = np.array([1, 2, 3], dtype=int)
            J = np.array([0, 1, 2], dtype=int)
            V = np.array([np.sqrt(3), 2, np.sqrt(3)], dtype=complex)
            self.Sm = sp.coo_matrix((V, (I, J)), shape=(4, 4))

            self.I = sp.eye(4)

        elif dof == "Boson":

            self.hilbsize = boson_dim
            self.type = str(boson_dim) + "_Boson"

            # define a and a^\dagger
            I = np.array(range(0, boson_dim - 1), dtype=int)
            J = np.array(range(1, boson_dim), dtype=int)
            V = np.sqrt(np.array(range(1, boson_dim)), dtype=complex)
            self.annih = sp.coo_matrix((V, (I, J)), shape=(boson_dim, boson_dim))
            self.creat = np.transpose(self.annih)
            self.occup = self.creat.dot(self.annih)
            self.I = sp.eye(boson_dim)

        elif dof == "Fermion":

            self.hilbsize = 2
            self.type = "Fermion"

            # define f and f^\dagger
            I = np.array(range(0, 1), dtype=int)
            J = np.array(range(1, 2), dtype=int)
            V = np.sqrt(np.array(range(1, 2)), dtype=float)
            self.annih = sp.coo_matrix((V, (I, J)), shape=(2, 2))
            self.creat = np.transpose(self.annih)
            self.occup = self.creat.dot(self.annih)
            self.I = sp.eye(2)



        else:
            raise TypeError("Dof type not yet supported..")

# fermion = Dofs("Fermion")
# print(fermion.annih, "\n")
# print(fermion.creat, "\n")
# print(fermion.occup, "\n")

#
# print(Spins.Sz * Spins.Sz)

#
# print(Spins.Sx.shape[0])
