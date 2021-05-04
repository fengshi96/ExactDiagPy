import numpy as np
import scipy.sparse as sp
from src.Dofs import Dofs
from src.Hamiltonian import Hamiltonian, PairConstructor, TwoSpinOps
from src.Helper import matprint


class AKLT(Hamiltonian):
    def __init__(self, Lat, Para):
        super().__init__(Lat, Para)

        self.Nsite = Lat.LLX

        self.Hx = Para.Hx
        self.Kxx1 = Para.Kxx
        self.Kxx2 = Para.Kxx / 3.0

        self.Hy = Para.Hy
        self.Kyy1 = Para.Kyy
        self.Kyy2 = Para.Kyy / 3.0

        self.Hz = Para.Hz
        self.Kzz1 = Para.Kzz
        self.Kzz2 = Para.Kzz / 3.0

        self.Kxx1Pair_ = np.zeros(())  # pairwise non-zero coupling \\
        self.Kyy1Pair_ = np.zeros(())  # 1st and 2nd cols are site indices
        self.Kzz1Pair_ = np.zeros(())

        self.Kxx2Pair_ = np.zeros(())
        self.Kyy2Pair_ = np.zeros(())
        self.Kzz2Pair_ = np.zeros(())

        self.Kxx1coef_ = np.zeros(())  # pairwise non-zero coupling strength
        self.Kyy1coef_ = np.zeros(())
        self.Kzz1coef_ = np.zeros(())

        self.Kxx2coef_ = np.zeros(())
        self.Kyy2coef_ = np.zeros(())
        self.Kzz2coef_ = np.zeros(())

        self.Ham = self.BuildAKLT()

    def BuildAKLT(self):

        for bond in range(0, self.Lat.Number1neigh):
            for i in range(0, self.Nsite):
                j = self.Lat.nn_[i, bond]
                # print(j)
                if i < j and j >= 0:
                    # Kxx1,2_ij
                    self.Kxx1Graph_[i, j] = self.Kxx1
                    self.Kxx1Graph_[j, i] = self.Kxx1
                    self.Kxx2Graph_[i, j] = self.Kxx2
                    self.Kxx2Graph_[j, i] = self.Kxx2

                    # Kyy1,2_ij
                    self.Kyy1Graph_[i, j] = self.Kyy1
                    self.Kyy1Graph_[j, i] = self.Kyy1
                    self.Kyy2Graph_[i, j] = self.Kyy2
                    self.Kyy2Graph_[j, i] = self.Kyy2

                    # Kzz1,2_ij
                    self.Kzz1Graph_[i, j] = self.Kzz1
                    self.Kzz1Graph_[j, i] = self.Kzz1
                    self.Kzz2Graph_[i, j] = self.Kzz2
                    self.Kzz2Graph_[j, i] = self.Kzz2

        print("\nKxx1Graph_:")
        matprint(self.Kxx1Graph_)
        print("\nKyy1Graph_:")
        matprint(self.Kyy1Graph_)
        print("\nKzz1Graph_:")
        matprint(self.Kzz1Graph_)
        print("\nKxx2Graph_:")
        matprint(self.Kxx2Graph_)
        print("\nKyy2Graph_:")
        matprint(self.Kyy2Graph_)
        print("\nKzz2Graph_:")
        matprint(self.Kzz2Graph_)

        self.Kxx1Pair_, self.Kxx1coef_ = PairConstructor(self.Kxx1Graph_, self.Nsite)
        self.Kyy1Pair_, self.Kyy1coef_ = PairConstructor(self.Kyy1Graph_, self.Nsite)
        self.Kzz1Pair_, self.Kzz1coef_ = PairConstructor(self.Kzz1Graph_, self.Nsite)

        self.Kxx2Pair_, self.Kxx2coef_ = PairConstructor(self.Kxx2Graph_, self.Nsite)
        self.Kyy2Pair_, self.Kyy2coef_ = PairConstructor(self.Kyy2Graph_, self.Nsite)
        self.Kzz2Pair_, self.Kzz2coef_ = PairConstructor(self.Kzz2Graph_, self.Nsite)

        # ---------------------Build Hamiltonian as Sparse Matrix-------------------

        print("[Hamiltonian.py] Building Hamiltonian as Sparse Matrix...")

        Hamx1 = TwoSpinOps(self.Kxx1Pair_, self.Kxx1coef_, self.sx, self.sx, self.Nsite)
        Hamy1 = TwoSpinOps(self.Kyy1Pair_, self.Kyy1coef_, self.sy, self.sy, self.Nsite)
        Hamz1 = TwoSpinOps(self.Kzz1Pair_, self.Kzz1coef_, self.sz, self.sz, self.Nsite)

        Hamxxxx2 = TwoSpinOps(self.Kxx2Pair_, self.Kxx2coef_, self.sx * self.sx, self.sx * self.sx, self.Nsite)
        Hamxxyy2 = TwoSpinOps(self.Kyy2Pair_, self.Kyy2coef_, self.sy * self.sy, self.sy * self.sy, self.Nsite)
        Hamxxzz2 = TwoSpinOps(self.Kzz2Pair_, self.Kzz2coef_, self.sz * self.sz, self.sz * self.sz, self.Nsite)

        Hamxy2 = TwoSpinOps(self.Kxx2Pair_, self.Kxx2coef_, self.sx * self.sy, self.sx * self.sy, self.Nsite)
        Hamxz2 = TwoSpinOps(self.Kyy2Pair_, self.Kyy2coef_, self.sx * self.sz, self.sx * self.sz, self.Nsite)

        Hamyx2 = TwoSpinOps(self.Kzz2Pair_, self.Kzz2coef_, self.sy * self.sx, self.sy * self.sx, self.Nsite)
        Hamyz2 = TwoSpinOps(self.Kxx2Pair_, self.Kxx2coef_, self.sy * self.sz, self.sy * self.sz, self.Nsite)

        Hamzx2 = TwoSpinOps(self.Kyy2Pair_, self.Kyy2coef_, self.sz * self.sx, self.sz * self.sx, self.Nsite)
        Hamzy2 = TwoSpinOps(self.Kzz2Pair_, self.Kzz2coef_, self.sz * self.sy, self.sz * self.sy, self.Nsite)

        Ham = Hamx1 + Hamy1 + Hamz1 + Hamxy2 + Hamxz2 + Hamyx2 + Hamyz2 + Hamzx2 + Hamzy2
        Ham += Hamxxxx2 + Hamxxyy2 + Hamxxzz2

        # --------------------------- Add external field -------------------------

        hilsize = self.sx.shape[0]
        for i in range(0, self.Nsite):
            ida = sp.eye(hilsize ** i)
            idb = sp.eye(hilsize ** (self.Nsite - i - 1))
            Ham += sp.kron(ida, sp.kron(self.sx, idb)) * self.Hx
            Ham += sp.kron(ida, sp.kron(self.sy, idb)) * self.Hy
            Ham += sp.kron(ida, sp.kron(self.sz, idb)) * self.Hz

        # if self.Lat.Model == "AKLT":
        Ham += sp.eye(Ham.shape[0]) * 2.0 / 3.0 * len(self.Kxx1coef_)
        return Ham
