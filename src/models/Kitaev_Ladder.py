import numpy as np
import scipy.sparse as sp
from src.Dofs import Dofs
from src.Hamiltonian import Hamiltonian, PairConstructor, TwoSpinOps
from src.Observ import Observ
from src.Helper import matprint, printfArray



class Kitaev_Ladder(Hamiltonian):
    def __init__(self, Lat, Para):
        super().__init__(Lat, Para)

        self.Lx = Lat.LLX  # Lattice dimensions
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
                    S *= self.Ob.LSxBuild(site, pauli=False, qm="SpinHalf") * 2  # *2 to recover pauli matrix from spin-1/2
                elif SigmaArray[i, s] == "y":
                    # print("site, component =", site, SigmaArray[i, s])
                    S *= self.Ob.LSyBuild(site, pauli=False, qm="SpinHalf") * 2
                elif SigmaArray[i, s] == "z":
                    # print("site, component =", site, SigmaArray[i, s])
                    S *= self.Ob.LSzBuild(site, pauli=False, qm="SpinHalf") * 2
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
        matprint(self.KxxGraph_, delimiter=' ', nospacing=False)
        print("\nKyyGraph_:")
        matprint(self.KyyGraph_, delimiter=' ', nospacing=False)
        print("\nKzzGraph_:")
        matprint(self.KzzGraph_, delimiter=' ', nospacing=False)

        printfArray(
            np.round(self.KxxGraph_).astype(int),
            filename="KxxGraph.txt",
            transpose=False
        ) 
        printfArray(
            np.round(self.KyyGraph_).astype(int),
            filename="KyyGraph.txt",
            transpose=False
        )
        printfArray(
            np.round(self.KzzGraph_).astype(int),
            filename="KzzGraph.txt",
            transpose=False
        )


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
                    Ham += sp.kron(ida, sp.kron(self.sx, idb)) * self.Hx * 2
                    Ham += sp.kron(ida, sp.kron(self.sy, idb)) * self.Hy * 2
                    Ham += sp.kron(ida, sp.kron(self.sz, idb)) * self.Hz * 2
                elif i%2 != 0:
                    Ham -= sp.kron(ida, sp.kron(self.sx, idb)) * self.Hx * 2
                    Ham -= sp.kron(ida, sp.kron(self.sy, idb)) * self.Hy * 2
                    Ham -= sp.kron(ida, sp.kron(self.sz, idb)) * self.Hz * 2
        return Ham

       
    def buildFluxList(self):
        """
        Measure the fluxes in the system
        :return: 1D list of fluxes
        1 - x - 3 - y - 5 - x - 7 - y - 9 - x - 11 -
        |       |       |       |       |       |            
        z  Ws1  z  Ws2  z  Ws1  z  Ws2  z  Ws1  z              
        |       |       |       |       |       |          
        0 - y - 2 - x - 4 - y - 6 - x - 8 - y - 10 -
        with 
        Ws1 = X0 Y1 X2 Y3 + translations
        Ws2 = Y2 X3 Y4 X5 + translations
        Ws1.Ws2 = Wp = X0 Y1 Z2 Z3 Y4 X5 + translations
        """
        # Define the Ws1 operators
        Ws1_Indx = []
        Ws1_Ops = []
        Ws2_Indx = []
        Ws2_Ops = []
        Wp_Indx = []
        Wp_Ops = []
        for i in np.arange(0, self.Lat.Nsite, 2):
            print(i)
            if i % 4 == 0:
                # Ws1 operator with lower-left corner on sites 0, 4, 8, ...
                j0 = i
                j1 = self.Lat.nn_[i, 2]
                j2 = self.Lat.nn_[i, 0] 
                j3 = self.Lat.nn_[j1,0]
                # print("[Ws1 sites] j0, j1, j2, j3 = ", j0, j1, j2, j3)
                if j0 >= 0 and j1 >= 0 and j2 >= 0 and j3 >= 0:
                    Ws1_Indx.append([j0, j1, j2, j3])
                    Ws1_Ops.append(["x", "y", "x", "y"])  # X0 Y1 X2 Y3, the order of operators in the plaquette
                # Wp operator with lower-left corner on sites 0, 4, 8, ...
                j4 = self.Lat.nn_[j2, 0] 
                j5 = self.Lat.nn_[j3, 0] 
                # print("[Wp sites] j0, j1, j2, j3, j4, j5 = ", j0, j1, j2, j3, j4, j5)
                if j0 >= 0 and j1 >= 0 and j2 >= 0 and j3 >= 0 and j4>= 0 and j5 >= 0:
                    Wp_Indx.append([j0, j1, j2, j3, j4, j5])
                    Wp_Ops.append(["x", "y", "z", "z", "y", "x"])  # X0 Y1 Z2 Z3 Y4 X5, the order of operators in the plaquette
            
            # Ws2 operator with lower-left corner on sites 2, 6, 10, ... 
            elif i % 4 == 2:
                j2 = i
                j3 = self.Lat.nn_[i, 2]
                j4 = self.Lat.nn_[i, 0] 
                j5 = self.Lat.nn_[j3,0]
                # print("[Ws2 sites] j2, j3, j4, j5 = ", j2, j3, j4, j5)
                if j2 >= 0 and j3 >= 0 and j4 >= 0 and j5 >= 0:
                    Ws2_Indx.append([j2, j3, j4, j5])
                    Ws2_Ops.append(["y", "x", "y", "x"]) 

        return np.array(Ws1_Indx), np.array(Ws1_Ops), np.array(Ws2_Indx), np.array(Ws2_Ops), np.array(Wp_Indx), np.array(Wp_Ops)
    


                
        

if __name__ == '__main__':
    from src.Parameter import Parameter
    from src.Lattice import Lattice
    import h5py
    from src.Observ import matele

    param = Parameter("input.inp")
    lat = Lattice(param)
    test = Kitaev_Ladder(lat, param)

    # test measuring fluxes 
    outputname = "dataSpec.hdf5"
    inputname = "input.inp"

    para = Parameter(inputname)
    Lat = Lattice(para)
    ob = Observ(Lat, para)
    print(para.parameters["Model"])
    dof = Dofs("SpinHalf")  # default spin-1/2

    # ------- Read dataSpec file -------
    rfile = h5py.File(outputname, 'r')
    evalset = rfile["3.Eigen"]["Eigen Values"]
    evecset = rfile["3.Eigen"]["Wavefunctions"]

    # extract eigen value and eigen vector from HDF5 rfile
    evals = np.zeros(para.parameters["Nstates"], dtype=float)
    evecs = np.zeros((dof.hilbsize ** Lat.Nsite, para.parameters["Nstates"]), dtype=complex)
    evalset.read_direct(evals)
    evecset.read_direct(evecs)
    rfile.close()
    gs = evecs[:, 0]

    # Build the flux list
    ws1_indx, ws1_ops, ws2_indx, ws2_ops, wp_indx, wp_ops = test.buildFluxList()
    print("ws1_indx, ws1_ops = ", ws1_indx, ws1_ops)
    print("ws2_indx, ws2_ops = ", ws2_indx, ws2_ops)
    print("wp_indx, wp_ops = ", wp_indx, wp_ops)

    # Build the flux operators and measure
    print("measuring ws1 fluxes:")
    ops = test.BuildFlux(ws1_indx, ws1_ops)
    for op in ops:
        result = matele(gs, op, gs).real
        print(result)

    print("measuring ws2 fluxes:")
    ops = test.BuildFlux(ws2_indx, ws2_ops)
    for op in ops:
        result = matele(gs, op, gs).real
        print(result)

    print("measuring wp fluxes:")
    ops = test.BuildFlux(wp_indx, wp_ops)
    for op in ops:
        result = matele(gs, op, gs).real
        print(result)






    
