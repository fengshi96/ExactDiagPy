import numpy as np
import scipy.sparse as sp
from src.Hamiltonian import Hamiltonian, PairConstructor, TwoSpinOps
from src.Dofs import Dofs
from src.Helper import matprint


class Fermions(Hamiltonian):
    def __init__(self, Lat, Para):
        super().__init__(Lat, Para)

        self.t = Para.parameters["t"]
        self.U = Para.parameters["U"]
        self.mu = Para.parameters["mu"]


        if Para.parameters["Model"] == "Fermi_Hubbard":
            self.Ham = self.BuildFermiHubbard()
        else:
            raise ValueError("Model not yet supported")

    def BuildFermiHubbard(self):
        for bond in range(0, self.Lat.Number1neigh):
            for i in range(0, self.Nsite):
                j = self.Lat.nn_[i, bond]
                # print(j)
                if i < j and j >= 0:
                    self.tGraph_[i, j] = self.t
                    self.tGraph_[j, i] = self.t

        print("\ntGraph_:")
        matprint(self.tGraph_)
        self.tPair_, self.tcoef_ = PairConstructor(self.tGraph_, self.Nsite)

        # ---------------------Build Hamiltonian as Sparse Matrix-------------------

        print("[Hamiltonian.py] Building Hamiltonian as Sparse Matrix...")

        # hopping term
        Ham = TwoSpinOps(self.tPair_, self.tcoef_, self.creat, self.annih, self.Nsite)
        Ham += np.transpose(Ham.conj())
        # chemical potential
        for i in range(0, self.Nsite):
            ida = sp.eye(2 ** i)
            idb = sp.eye(2 ** (self.Nsite - i - 1))
            Ham += sp.kron(ida, sp.kron(self.occup, idb)) * self.mu
        # on-site interaction
        for i in range(0, self.Nsite):
            ida = sp.eye(2 ** i)
            idb = sp.eye(2 ** (self.Nsite - i - 1))
            Ham += sp.kron(ida, sp.kron(self.occup.dot(self.occup - self.Id), idb)) * self.U
        return Ham
