import numpy as np
from src.Hamiltonian import Hamiltonian
from src.Helper import matprint


class KitaevQuadratic(Hamiltonian):
    def __init__(self, Lat, Para, FlipZ=None):
        super().__init__(Lat, Para)

        if FlipZ is None:
            FlipZ = []
        self.Kxx = Para.parameters["Kxx"]  # coupling strength of x-bond
        self.Kyy = Para.parameters["Kyy"]
        self.Kzz = Para.parameters["Kzz"]
        self.h = 2.0

        self.KxxGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
        self.KyyGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
        self.KzzGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)

        self.nnnGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)

        self.Nsite = Lat.LLX * Lat.LLY * 2
        self.FlipZ = FlipZ
        self.HamMatrix = self.Build()

    def Build(self):
        """
        Build Kitaev Hamiltonian in Majorana basis
        :return: Kitaev Hamiltonian as dense matrix
        """
        # Nearest Neighbor
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
                if i in self.FlipZ:
                    self.KzzGraph_[i, j] *= -1
                    self.KzzGraph_[j, i] *= -1

        matprint(self.KzzGraph_)
        Ham_TB = self.KzzGraph_ + self.KxxGraph_ + self.KyyGraph_

        # Next-nearest neighbor
        for i in range(0, self.Nsite):
            jx = self.Lat.nn_[i, 0]
            jy = self.Lat.nn_[i, 1]
            jz = self.Lat.nn_[i, 2]
            self.nnnGraph_[jx, jy] = self.h
            self.nnnGraph_[jy, jx] = self.h

            self.nnnGraph_[jy, jz] = self.h
            self.nnnGraph_[jz, jy] = self.h

            self.nnnGraph_[jz, jx] = self.h
            self.nnnGraph_[jx, jz] = self.h

        Ham_TB += self.nnnGraph_

        # print("\nHam_TB:")
        matprint(Ham_TB)
        return Ham_TB


if __name__ == '__main__':
    from src.Parameter import Parameter
    from src.Lattice import Lattice
    from scipy.linalg import eigh

    Para = Parameter("../../input.inp")  # import parameters from input.inp
    Lat = Lattice(Para)  # Build lattice

    FlipZ = []
    Hamil = KitaevQuadratic(Lat, Para, FlipZ).HamMatrix
    evals, evecs = eigh(Hamil)
    print(evals)  # [int(len(evals) / 2) - 1:int(len(evals) / 2) + 1]
