import numpy as np
import scipy.sparse as sp
from src.Dofs import Dofs

class Hamiltonian:

    def __init__(self, Lat, Para):

        self.Lat = Lat
        self.Nsite = 1

        if Lat.Geometry == "Square":
            self.Nsite = Lat.LLX * Lat.LLY
        elif Lat.Geometry == "Honeycomb":
            self.Nsite = Lat.LLX * Lat.LLY * 2
        elif Lat.Geometry == "Chain":
            self.Nsite = Lat.LLX
        else:
            raise ValueError("Geometry not yet supported")

        if Para.parameters["Dof"] == "Fermion":
            Fermion = Dofs("Fermion")
            self.annih = Fermion.annih
            self.creat = Fermion.creat
            self.occup = Fermion.occup
            self.Id = Fermion.I

            self.tGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
            self.tPair_ = np.zeros(())
            self.tcoef_ = np.zeros(())

        elif Para.parameters["Dof"] == "Boson":
            Boson = Dofs("Boson", Para.parameters["maxOccupation"])
            self.annih = Boson.annih
            self.creat = Boson.creat
            self.occup = Boson.occup
            self.Id = Boson.I

            self.tGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
            self.tPair_ = np.zeros(())
            self.tcoef_ = np.zeros(())

        elif Para.parameters["Dof"] == "SpinHalf" or Para.parameters["Dof"] == "SpinOne":
            Spins = Dofs(Para.parameters["Dof"])
            self.sx = Spins.Sx
            self.sy = Spins.Sy
            self.sz = Spins.Sz

            if Para.parameters["Model"] == "BLBQ" or Para.parameters["Dof"] == "AKLT":
                self.Kxx1Graph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
                self.Kyy1Graph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
                self.Kzz1Graph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)

                self.Kxx2Graph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
                self.Kyy2Graph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
                self.Kzz2Graph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
            else:
                self.KxxGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
                self.KyyGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
                self.KzzGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)

        else:
            raise ValueError("Dof type not supported")

def PairConstructor(Graph_, Nsite):
    bonds = int(np.count_nonzero(Graph_) / 2)  # Number of non-zero bonds. Only half of matrix elements is needed
    PairInd_ = np.zeros((bonds, 2))  # Indices of pairs of sites that have non-zero coupling
    PairCoef_ = np.zeros(bonds)  # coupling constant of pairs

    # extract non-zero coupling pairs and their value
    counter = 0
    for i in range(0, Nsite):
        for j in range(i, Nsite):
            if Graph_[i, j] != 0:
                PairInd_[counter, 0] = i
                PairInd_[counter, 1] = j
                PairCoef_[counter] = Graph_[i, j]
                counter += 1

    return PairInd_, PairCoef_

# Two point operator by tensor product
def TwoSpinOps(PairInd_, PairCoef_, Dof1, Dof2, Nsite):
    Hilsize = Dof1.shape[0]
    Nbonds = len(PairCoef_)  # Number of non-zero bonds
    Hamtmp_ = sp.eye(Hilsize ** Nsite, dtype=complex) * 0
    for i in range(0, Nbonds):
        ia = PairInd_[i, 0]
        ib = PairInd_[i, 1]
        coef = PairCoef_[i]

        ida = sp.eye(Hilsize ** ia)
        idm = sp.eye(Hilsize ** (ib - ia - 1))
        idb = sp.eye(Hilsize ** (Nsite - ib - 1))

        tmp = sp.kron(sp.kron(sp.kron(sp.kron(ida, Dof1), idm), Dof2), idb)
        Hamtmp_ += tmp * coef

    return Hamtmp_

# para = Parameter("../input.inp")
# Lat = Lattice(para)
# Ham = Hamiltonian(Lat)
