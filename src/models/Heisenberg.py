import numpy as np
import scipy.sparse as sp
from src.Dofs import Dofs
from src.Hamiltonian import Hamiltonian, PairConstructor, TwoSpinOps
from src.Helper import matprint


class Heisenberg(Hamiltonian):
    def __init__(self, Lat, Para):
        super().__init__(Lat, Para)

        self.Hx = Para.Hx
        self.Hy = Para.Hy
        self.Hz = Para.Hz

        self.KxxPair_ = np.zeros(())  # pairwise non-zero coupling \\
        self.KyyPair_ = np.zeros(())  # 1st and 2nd cols are site indices
        self.KzzPair_ = np.zeros(())

        self.Kxxcoef_ = np.zeros(())  # pairwise non-zero coupling strength
        self.Kyycoef_ = np.zeros(())
        self.Kzzcoef_ = np.zeros(())

        self.Kxx = Para.Kxx  # coupling strength of x-bond
        self.Kyy = Para.Kyy
        self.Kzz = Para.Kzz

        self.KxxGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
        self.KyyGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
        self.KzzGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
        self.Ham = self.BuildHeisenberg()

    def BuildHeisenberg(self):
        lat = self.Lat

        for bond in range(0, lat.Number1neigh):
            for i in range(0, self.Nsite):
                j = lat.nn_[i, bond]
                # print(j)
                if i < j and j >= 0:
                    # Kxx_ij * S_i^x S_j^x
                    self.KxxGraph_[i, j] = self.Kxx
                    self.KxxGraph_[j, i] = self.Kxx

                    # Kyy_ij * S_i^y S_j^y
                    self.KyyGraph_[i, j] = self.Kyy
                    self.KyyGraph_[j, i] = self.Kyy

                    # Kzz_ij * S_i^z S_j^z
                    self.KzzGraph_[i, j] = self.Kzz
                    self.KzzGraph_[j, i] = self.Kzz

        print("\nKxxGraph_:")
        matprint(self.KxxGraph_)
        print("\nKyyGraph_:")
        matprint(self.KyyGraph_)
        print("\nKzzGraph_:")
        matprint(self.KzzGraph_)

        # matprint(self.KxxGraph_-self.KzzGraph_); matprint(self.KxxGraph_-self.KyyGraph_)

        self.KxxPair_, self.Kxxcoef_ = PairConstructor(self.KxxGraph_, self.Nsite)
        self.KyyPair_, self.Kyycoef_ = PairConstructor(self.KyyGraph_, self.Nsite)
        self.KzzPair_, self.Kzzcoef_ = PairConstructor(self.KzzGraph_, self.Nsite)
        # print(self.KzzPair_)
        # print(self.Kzzcoef_)
        # ---------------------Build Hamiltonian as Sparse Matrix-------------------

        print("[Hamiltonian.py] Building Hamiltonian as Sparse Matrix...")

        Hamx = TwoSpinOps(self.KxxPair_, self.Kxxcoef_, self.sx, self.sx, self.Nsite)
        Hamy = TwoSpinOps(self.KyyPair_, self.Kyycoef_, self.sy, self.sy, self.Nsite)
        Hamz = TwoSpinOps(self.KzzPair_, self.Kzzcoef_, self.sz, self.sz, self.Nsite)

        Ham = Hamx + Hamy + Hamz

        # --------------------------- Add external field -------------------------

        for i in range(0, self.Nsite):
            ida = sp.eye(2 ** i)
            idb = sp.eye(2 ** (self.Nsite - i - 1))
            Ham += sp.kron(ida, sp.kron(self.sx, idb)) * self.Hx
            Ham += sp.kron(ida, sp.kron(self.sy, idb)) * self.Hy
            Ham += sp.kron(ida, sp.kron(self.sz, idb)) * self.Hz

        return Ham



