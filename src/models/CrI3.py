import numpy as np
import scipy.sparse as sp
from src.Dofs import Dofs
from src.Hamiltonian import Hamiltonian, PairConstructor, TwoSpinOps
from src.Observ import Observ
from src.Helper import matprint


class CrI3(Hamiltonian):
    def __init__(self, Lat, Para):
        super().__init__(Lat, Para)

        self.Hx = Para.parameters["Bxx"]
        self.Hy = Para.parameters["Byy"]
        self.Hz = Para.parameters["Bzz"]

        # ----------------------------- Kitaev

        try:
            self.muFlux = Para.parameters["muFlux"]
        except KeyError:
            self.muFlux = None

        try:
            self.pinning = Para.parameters["Pinning"]
        except KeyError:
            self.pinning = None

        try:
            self.option = Para.parameters["Option"]
        except KeyError:
            self.option = [None]

        if "threeSpin" in self.option:
            self.threeSpin = Para.parameters["threeSpin"]

        self.KxxPair_ = np.zeros(())  # pairwise non-zero coupling \\
        self.KyyPair_ = np.zeros(())  # 1st and 2nd cols are site indices
        self.KzzPair_ = np.zeros(())

        self.Kxxcoef_ = np.zeros(())  # pairwise non-zero coupling strength
        self.Kyycoef_ = np.zeros(())
        self.Kzzcoef_ = np.zeros(())

        self.Kxx = Para.parameters["Kxx"]  # coupling strength of x-bond
        self.Kyy = Para.parameters["Kyy"]
        self.Kzz = Para.parameters["Kzz"]

        self.Nsite = Lat.LLX * Lat.LLY * 2

        # ----------------------------- Heisenberg

        self.JxxPair_ = np.zeros(())  # pairwise non-zero coupling \\
        self.JyyPair_ = np.zeros(())  # 1st and 2nd cols are site indices
        self.JzzPair_ = np.zeros(())

        self.Jxxcoef_ = np.zeros(())  # pairwise non-zero coupling strength
        self.Jyycoef_ = np.zeros(())
        self.Jzzcoef_ = np.zeros(())

        self.Jxx = Para.parameters["Jxx"]  # coupling strength of x-bond
        self.Jyy = Para.parameters["Jyy"]
        self.Jzz = Para.parameters["Jzz"]

        self.JxxGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
        self.JyyGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
        self.JzzGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)

        # ----------------------------

        self.Ob = Observ(Lat, Para)
        self.Ham = self.BuildCrI3()

    def BuildFlux(self, IndexArray, SigmaArray):
        """
        Construct Flux operators in the full Hilbert space
        :param IndexArray: 2D Array, each row is a 6-element 1D array that represents 6 vertices in a plaquette
               SigmaArray: 2D Array, each row is a 6-element 1D array that represent 6 pauli matrices
        :return: 1D list that contains Flux operators as sparse matrix
        """
        numFlux = len(IndexArray)
        hilbsize = Dofs("SpinHalf").hilbsize

        FluxOperators = []
        for i in range(numFlux):
            S = sp.eye(hilbsize ** self.Nsite, dtype=complex)
            for s, site in enumerate(IndexArray[i, :]):
                if SigmaArray[i, s] == "x":
                    # print("site, component =", site, SigmaArray[i, s])
                    S *= self.Ob.LSxBuild(site, "SpinHalf") * 2  # *2 to recover pauli matrix from spin-1/2
                elif SigmaArray[i, s] == "y":
                    # print("site, component =", site, SigmaArray[i, s])
                    S *= self.Ob.LSyBuild(site, "SpinHalf") * 2
                elif SigmaArray[i, s] == "z":
                    # print("site, component =", site, SigmaArray[i, s])
                    S *= self.Ob.LSzBuild(site, "SpinHalf") * 2
                else:
                    raise ValueError("invalid name for spin components:", SigmaArray[i, s])
            FluxOperators.append(S)
        return FluxOperators


    def BuildCrI3(self):
        """
        Build CrI3 Hamiltonian
        :return: CrI3 Hamiltonian as sparse matrix
        """
        # --------------------------- Start with Kitaev term --------------------
        for i in range(0, self.Nsite):
            # Kxx_Conn
            j = self.Lat.nn_[i, 0]
            if i < j and j >= 0:
                self.KxxGraph_[i, j] = self.Kxx
                self.KxxGraph_[j, i] = self.Kxx

            # Kyy_Conn
            j = self.Lat.nn_[i, 1]
            if i < j and j >= 0:
                self.KyyGraph_[i, j] = self.Kyy
                self.KyyGraph_[j, i] = self.Kyy

            # Kzz_Conn
            j = self.Lat.nn_[i, 2]
            if i < j and j >= 0:
                self.KzzGraph_[i, j] = self.Kzz
                self.KzzGraph_[j, i] = self.Kzz

        print("\nKxxGraph_:")
        matprint(self.KxxGraph_)
        print("\nKyyGraph_:")
        matprint(self.KyyGraph_)
        print("\nKzzGraph_:")
        matprint(self.KzzGraph_)

        self.KxxPair_, self.Kxxcoef_ = PairConstructor(self.KxxGraph_, self.Nsite)
        self.KyyPair_, self.Kyycoef_ = PairConstructor(self.KyyGraph_, self.Nsite)
        self.KzzPair_, self.Kzzcoef_ = PairConstructor(self.KzzGraph_, self.Nsite)

        # ---------------------Build Hamiltonian as Sparse Matrix-------------------

        print("[Hamiltonian.py] Building Hamiltonian as Sparse Matrix...")

        Hamx = TwoSpinOps(self.KxxPair_, self.Kxxcoef_, self.sx, self.sx, self.Nsite)
        Hamy = TwoSpinOps(self.KyyPair_, self.Kyycoef_, self.sy, self.sy, self.Nsite)
        Hamz = TwoSpinOps(self.KzzPair_, self.Kzzcoef_, self.sz, self.sz, self.Nsite)

        Ham = Hamx + Hamy + Hamz

        # ---------------------- Add Heienberg term ---------------------
        lat = self.Lat

        for bond in range(0, lat.Number1neigh):
            for i in range(0, self.Nsite):
                j = lat.nn_[i, bond]
                # print(j)
                if i < j and j >= 0:
                    # Jxx_ij * S_i^x S_j^x
                    self.JxxGraph_[i, j] = self.Jxx
                    self.JxxGraph_[j, i] = self.Jxx

                    # Jyy_ij * S_i^y S_j^y
                    self.JyyGraph_[i, j] = self.Jyy
                    self.JyyGraph_[j, i] = self.Jyy

                    # Jzz_ij * S_i^z S_j^z
                    self.JzzGraph_[i, j] = self.Jzz
                    self.JzzGraph_[j, i] = self.Jzz

        print("\nJxxGraph_:")
        matprint(self.JxxGraph_)
        print("\nJyyGraph_:")
        matprint(self.JyyGraph_)
        print("\nJzzGraph_:")
        matprint(self.JzzGraph_)

        # matprint(self.JxxGraph_-self.JzzGraph_); matprint(self.JxxGraph_-self.JyyGraph_)

        self.JxxPair_, self.Jxxcoef_ = PairConstructor(self.JxxGraph_, self.Nsite)
        self.JyyPair_, self.Jyycoef_ = PairConstructor(self.JyyGraph_, self.Nsite)
        self.JzzPair_, self.Jzzcoef_ = PairConstructor(self.JzzGraph_, self.Nsite)
        # print(self.JzzPair_)
        # print(self.Jzzcoef_)
        # ---------------------Build Hamiltonian as Sparse Matrix-------------------

        print("[Hamiltonian.py] Building Hamiltonian as Sparse Matrix...")

        HamJx = TwoSpinOps(self.JxxPair_, self.Jxxcoef_, self.sx, self.sx, self.Nsite)
        HamJy = TwoSpinOps(self.JyyPair_, self.Jyycoef_, self.sy, self.sy, self.Nsite)
        HamJz = TwoSpinOps(self.JzzPair_, self.Jzzcoef_, self.sz, self.sz, self.Nsite)

        Ham += HamJx + HamJy + HamJz

        # --------------------------- Add external field -------------------------

        for i in range(0, self.Nsite):
            if self.option[0] is None:
                print("Adding Magnetic Field at site " + str(i) + "(Hx, Hy, Hz) = "
                      + str(self.Hx) + ", " + str(self.Hx) + ", " + str(self.Hx))
                ida = sp.eye(4 ** i)
                idb = sp.eye(4 ** (self.Nsite - i - 1))
                Ham += sp.kron(ida, sp.kron(self.sx, idb)) * self.Hx
                Ham += sp.kron(ida, sp.kron(self.sy, idb)) * self.Hy
                Ham += sp.kron(ida, sp.kron(self.sz, idb)) * self.Hz
            elif "zigzagField" in self.option:
                print("Adding Zigzag Field at site " + str(i))
                ida = sp.eye(4 ** i)
                idb = sp.eye(4 ** (self.Nsite - i - 1))
                if i%2 == 0:
                    Ham += sp.kron(ida, sp.kron(self.sx, idb)) * self.Hx
                    Ham += sp.kron(ida, sp.kron(self.sy, idb)) * self.Hy
                    Ham += sp.kron(ida, sp.kron(self.sz, idb)) * self.Hz
                elif i%2 != 0:
                    Ham -= sp.kron(ida, sp.kron(self.sx, idb)) * self.Hx
                    Ham -= sp.kron(ida, sp.kron(self.sy, idb)) * self.Hy
                    Ham -= sp.kron(ida, sp.kron(self.sz, idb)) * self.Hz


        # --------------------------- Add pinning field (to site 0) -------------------------

        if self.pinning is not None:
            site = 0
            ida = sp.eye(2 ** site)
            idb = sp.eye(2 ** (self.Nsite - site - 1))
            Ham += sp.kron(ida, sp.kron(self.sz, idb)) * self.pinning


        return Ham