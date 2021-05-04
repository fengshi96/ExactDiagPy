import numpy as np
import scipy.sparse as sp
from src.Dofs import Dofs
from src.Hamiltonian import Hamiltonian, PairConstructor, TwoSpinOps
from src.Helper import matprint


class Bosons(Hamiltonian):
    def __init__(self, Lat, Para):
        super().__init__(Lat, Para)

        self.t = Para.t
        self.U = Para.U
        self.mu = Para.mu
        self.maxOccupation = Para.maxOccupation
        self.tGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
        self.tPair_ = np.zeros(())
        self.tcoef_ = np.zeros(())
        if Para.Model == "Bose_Hubbard":
            self.Ham = self.BuildBoseHubbard()
        else:
            raise ValueError("Model not yet supported")

    def BuildBoseHubbard(self):
        for bond in range(0, self.Lat.Number1neigh):
            for i in range(0, self.Nsite):
                j = self.Lat.nn_[i, bond]
                # print(j)
                if i < j and j >= 0:
                    # Kxx_ij * S_i^x S_j^x
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
            ida = sp.eye(self.maxOccupation ** i)
            idb = sp.eye(self.maxOccupation ** (self.Nsite - i - 1))
            Ham += sp.kron(ida, sp.kron(self.occup, idb)) * self.mu
        # on-site interaction
        for i in range(0, self.Nsite):
            ida = sp.eye(self.maxOccupation ** i)
            idb = sp.eye(self.maxOccupation ** (self.Nsite - i - 1))
            Ham += sp.kron(ida, sp.kron(self.occup.dot(self.occup - self.Id), idb)) * self.U
        return Ham
