import numpy as np
import scipy.sparse as sp
from src.Dofs import Dofs
from src.Hamiltonian import Hamiltonian, PairConstructor, TwoSpinOps
from src.Observ import Observ
from src.Helper import matprint


class Kitaev(Hamiltonian):
    def __init__(self, Lat, Para):
        super().__init__(Lat, Para)

        self.Hx = Para.parameters["Bxx"]
        self.Hy = Para.parameters["Byy"]
        self.Hz = Para.parameters["Bzz"]
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
            self.option = None

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
        self.Ob = Observ(Lat, Para)
        self.Ham = self.BuildKitaev()

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


    def BuildKitaev(self):
        """
        Build Kitaev Hamiltonian
        :return: Kitaev Hamiltonian as sparse matrix
        """
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

        # --------------------------- Add external field -------------------------

        for i in range(0, self.Nsite):
            if self.option is None:
                ida = sp.eye(2 ** i)
                idb = sp.eye(2 ** (self.Nsite - i - 1))
                Ham += sp.kron(ida, sp.kron(self.sx, idb)) * self.Hx
                Ham += sp.kron(ida, sp.kron(self.sy, idb)) * self.Hy
                Ham += sp.kron(ida, sp.kron(self.sz, idb)) * self.Hz
            elif self.option == "zigzagField":
                ida = sp.eye(2 ** i)
                idb = sp.eye(2 ** (self.Nsite - i - 1))
                if i%2 == 0:
                    Ham += sp.kron(ida, sp.kron(self.sx, idb)) * self.Hx
                    Ham += sp.kron(ida, sp.kron(self.sy, idb)) * self.Hy
                    Ham += sp.kron(ida, sp.kron(self.sz, idb)) * self.Hz
                elif i%2 != 0:
                    Ham -= sp.kron(ida, sp.kron(self.sx, idb)) * self.Hx
                    Ham -= sp.kron(ida, sp.kron(self.sy, idb)) * self.Hy
                    Ham -= sp.kron(ida, sp.kron(self.sz, idb)) * self.Hz

        # --------------------------- Add the threeSpin term -------------------------
        if self.option is not None:
            if "threeSpin" in self.option:
                for i in range(0, self.Nsite):
                    sxi = self.Ob.LSxBuild(i)
                    syi = self.Ob.LSyBuild(i)
                    szi = self.Ob.LSzBuild(i)

                    Ham += sxi * self.Ob.LSyBuild(self.Lat.nn_[i, 1]) * self.Ob.LSzBuild(
                        self.Lat.nn_[i, 2]) * self.threeSpin
                    Ham += syi * self.Ob.LSxBuild(self.Lat.nn_[i, 0]) * self.Ob.LSzBuild(
                        self.Lat.nn_[i, 2]) * self.threeSpin
                    Ham += szi * self.Ob.LSxBuild(self.Lat.nn_[i, 0]) * self.Ob.LSyBuild(
                        self.Lat.nn_[i, 1]) * self.threeSpin

                    # 4-majorana term
                    # Ham += self.Ob.LSxBuild(self.Lat.nn_[i, 0]) * self.Ob.LSyBuild(self.Lat.nn_[i, 1]) \
                    #        * self.Ob.LSzBuild(self.Lat.nn_[i, 2]) * self.threeSpin

        # --------------------------- Add pinning field (to site 0) -------------------------

        if self.pinning is not None:
            site = 0
            ida = sp.eye(2 ** site)
            idb = sp.eye(2 ** (self.Nsite - site - 1))
            Ham += sp.kron(ida, sp.kron(self.sz, idb)) * self.pinning

        # ------------------ Add Flux bias (for 18 with PBC only at this stage) -------------
        if self.muFlux is not None and self.Nsite == 18:
            print("Adding Flux bias")
            IndexArray = np.array(([
                [1, 2, 3, 8, 7, 6],
                [3, 4, 5, 10, 9, 8],
                [7, 8, 9, 14, 13, 12],
                [9, 10, 11, 16, 15, 14],
                [13, 14, 15, 2, 1, 0],
                [15, 16, 17, 4, 3, 2],
                [17, 12, 13, 0, 5, 4],
                [5, 0, 1, 6, 11, 10],
                [11, 6, 7, 12, 17, 16]
            ]), dtype=int)
            SigmaArray = np.array(([
                ["x", "y", "z", "x", "y", "z"],
                ["x", "y", "z", "x", "y", "z"],
                ["x", "y", "z", "x", "y", "z"],
                ["x", "y", "z", "x", "y", "z"],
                ["x", "y", "z", "x", "y", "z"],
                ["x", "y", "z", "x", "y", "z"],
                ["x", "y", "z", "x", "y", "z"],
                ["x", "y", "z", "x", "y", "z"],
                ["x", "y", "z", "x", "y", "z"]
            ]))

            fluxOperators = self.BuildFlux(IndexArray, SigmaArray)
            for i in range(len(IndexArray)):
                Ham += self.muFlux * fluxOperators[i]

        return Ham