import numpy as np
import scipy.sparse as sp
from src.Dofs import Dofs
from src.Hamiltonian import Hamiltonian, PairConstructor, TwoSpinOps
from src.Observ import Observ
from src.Helper import matprint



class Kitaev_Ladder(Hamiltonian):
    def __init__(self, Lat, Para):
        super().__init__(Lat, Para)

        self.Hx = Para.parameters["Bxx"]
        self.Hy = Para.parameters["Byy"]
        self.Hz = Para.parameters["Bzz"]

        try:
            self.option = Para.parameters["Option"]
        except KeyError:
            self.option = [None]

        self.KxxPair_ = np.zeros(())  # pairwise non-zero coupling \\
        self.KyyPair_ = np.zeros(())  # 1st and 2nd cols are site indices
        self.KzzPair_ = np.zeros(())

        self.Kxxcoef_ = np.zeros(())  # pairwise non-zero coupling strength
        self.Kyycoef_ = np.zeros(())
        self.Kzzcoef_ = np.zeros(())

        self.Kxx = Para.parameters["Kxx"]  # coupling strength of x-bond
        self.Kyy = Para.parameters["Kyy"]
        self.Kzz = Para.parameters["Kzz"]

        self.indx_ = Lat.indx_  
        self.indy_ = Lat.indy_ 

        # self.Nsite = Lat.CustomNsites
        self.Ob = Observ(Lat, Para)
        self.Ham = self.BuildKitaevLadder()

    def BuildFlux(self, IndexArray, SigmaArray):
        """
        Construct Flux operators in the full Hilbert space
        :param IndexArray: 2D Array, each row is a 6 or 4-element 1D array that represents 6 vertices in a plaquette
               SigmaArray: 2D Array, each row is a 6 or 4-element 1D array that represent 6 pauli matrices
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


    def BuildKitaevLadder(self):
        """
        Build Kitaev Hamiltonian
        1 - x - 3 - y - 5 - x - 7 - y - 9 - x - 11 -
        |       |       |       |       |       |            
        z       z       z       z       z       z              
        |       |       |       |       |       |          
        0 - y - 2 - x - 4 - y - 6 - x - 8 - y - 10 -
        :return: Kitaev Hamiltonian as sparse matrix
        """
        # count from even site
        for i in np.arange(0, self.Nsite, 2):
            # Kxx_Conn, x + 1
            if i % 4 == 2:
                j = self.Lat.nn_[i, 0]
                if j >= 0:
                    self.KxxGraph_[i, j] = self.Kxx
                    self.KxxGraph_[j, i] = self.Kxx

            # Kyy_Conn, x + 1
            if i % 4 == 0:
                j = self.Lat.nn_[i, 0]
                if j >= 0:
                    self.KyyGraph_[i, j] = self.Kyy
                    self.KyyGraph_[j, i] = self.Kyy

            # Kzz_Conn, y + 1
            j = self.Lat.nn_[i, 2]
            if j >= 0:
                self.KzzGraph_[i, j] = self.Kzz
                self.KzzGraph_[j, i] = self.Kzz

        # count from odd site
        for i in np.arange(1, self.Nsite, 2):
            # Kxx_Conn, x + 1
            if i % 4 == 1:
                j = self.Lat.nn_[i, 0]
                if j >= 0:
                    self.KxxGraph_[i, j] = self.Kxx
                    self.KxxGraph_[j, i] = self.Kxx

            # Kyy_Conn, x + 1
            if i % 4 == 3:
                j = self.Lat.nn_[i, 0]
                if j >= 0:
                    self.KyyGraph_[i, j] = self.Kyy
                    self.KyyGraph_[j, i] = self.Kyy


        # remove double counting
        # self.KxxGraph_ = np.triu(self.KxxGraph_)
        # self.KyyGraph_ = np.triu(self.KyyGraph_)
        # self.KzzGraph_ = np.triu(self.KzzGraph_)

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

        Ham = (Hamx + Hamy + Hamz) * 4

        # --------------------------- Add external field -------------------------

        for i in range(0, self.Nsite):
            if self.option[0] is None:
                print("Adding Magnetic Field at site " + str(i))
                ida = sp.eye(2 ** i)
                idb = sp.eye(2 ** (self.Nsite - i - 1))
                Ham += sp.kron(ida, sp.kron(self.sx, idb)) * self.Hx * 2
                Ham += sp.kron(ida, sp.kron(self.sy, idb)) * self.Hy * 2
                Ham += sp.kron(ida, sp.kron(self.sz, idb)) * self.Hz * 2
            elif "zigzagField" in self.option:
                print("Adding Zigzag Field at site " + str(i))
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
        return Ham

       


       
    

if __name__ == '__main__':
    from src.Parameter import Parameter
    from src.Lattice import Lattice

    param = Parameter("/Users/shifeng/Projects/ExactDiagPy/input.inp")
    lat = Lattice(param)
    test = Kitaev_Ladder(lat, param)
