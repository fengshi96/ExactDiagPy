import numpy as np
from src.Hamiltonian import Hamiltonian
from src.Helper import matprint


class KitaevQuadratic(Hamiltonian):
    def __init__(self, Lat, Para, FlipX=[], FlipY=[], FlipZ=[]):
        super().__init__(Lat, Para)

        self.Kxx = Para.parameters["Kxx"]  # coupling strength of x-bond
        self.Kyy = Para.parameters["Kyy"]
        self.Kzz = Para.parameters["Kzz"]
        self.h = Para.parameters["h"]
        self.LLX = Para.parameters["LLX"]
        self.LLY = Para.parameters["LLY"]

        self.KxxGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
        self.KyyGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)
        self.KzzGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)

        self.nnnGraph_ = np.zeros((self.Nsite, self.Nsite), dtype=float)

        self.FlipX = FlipX
        self.FlipY = FlipY
        self.FlipZ = FlipZ
        self.Nsite = Lat.LLX * Lat.LLY * 2
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
                self.KxxGraph_[i, j] = -self.Kxx
            elif i > j and j >= 0:
                self.KxxGraph_[i, j] = self.Kxx

            # Kyy_Conn
            j = self.Lat.nn_[i, 1]
            if (i % 2 != 0 or int(i / (self.LLY * 2)) != 0) and (i % 2 != 1 or int(i / (self.LLY * 2)) != self.LLY - 1):
                if i < j and j >= 0:
                    self.KyyGraph_[i, j] = self.Kyy
                elif i > j and j >= 0:
                    self.KyyGraph_[i, j] = -self.Kyy
            else:
                if i < j and j >= 0:
                    self.KyyGraph_[i, j] = -self.Kyy
                elif i > j and j >= 0:
                    self.KyyGraph_[i, j] = self.Kyy

            # Kzz_Conn
            j = self.Lat.nn_[i, 2]
            if i % (self.LLY * 2) != self.LLY * 2 - 1 and i % (
                    self.LLY * 2) != 0:  # if not on y-boundary (not top nor bottom)
                if i < j and j >= 0:
                    self.KzzGraph_[i, j] = self.Kzz
                elif i > j and j >= 0:
                    self.KzzGraph_[i, j] = -self.Kzz
            else:  # if on the boundary
                if i < j and j >= 0:
                    self.KzzGraph_[i, j] = -self.Kzz
                elif i > j and j >= 0:
                    self.KzzGraph_[i, j] = self.Kzz

        # Flip X link
        for i in self.FlipX:
            j = self.Lat.nn_[i, 0]
            self.KxxGraph_[i, j] *= -1
            self.KxxGraph_[j, i] *= -1

        # Flip Y link
        for i in self.FlipY:
            j = self.Lat.nn_[i, 1]
            self.KyyGraph_[i, j] *= -1
            self.KyyGraph_[j, i] *= -1

        # Flip Z link
        for i in self.FlipZ:
            j = self.Lat.nn_[i, 2]
            self.KzzGraph_[i, j] *= -1
            self.KzzGraph_[j, i] *= -1

        Ham_TB = self.KzzGraph_ + self.KxxGraph_ + self.KyyGraph_

        # Next-nearest neighbor
        for i in range(0, self.Nsite):
            jx = self.Lat.nn_[i, 0]
            jy = self.Lat.nn_[i, 1]
            jz = self.Lat.nn_[i, 2]

            # x-z
            if i % 2 == 1:
                self.nnnGraph_[jx, jz] = self.h
                self.nnnGraph_[jz, jx] = -self.h
            elif i % 2 == 0:
                self.nnnGraph_[jx, jz] = self.h
                self.nnnGraph_[jz, jx] = -self.h

            # z-y
            if i % 2 == 1:
                self.nnnGraph_[jz, jy] = self.h
                self.nnnGraph_[jy, jz] = -self.h
            elif i % 2 == 0:
                self.nnnGraph_[jz, jy] = self.h
                self.nnnGraph_[jy, jz] = -self.h

            # y-x
            if i % 2 == 1:
                self.nnnGraph_[jy, jx] = self.h
                self.nnnGraph_[jx, jy] = -self.h
            elif i % 2 == 0:
                self.nnnGraph_[jy, jx] = self.h
                self.nnnGraph_[jx, jy] = -self.h

        Ham_TB += self.nnnGraph_

        # print("\nHam_TB:")
        #         matprint(Ham_TB)
        return Ham_TB * complex(0, 1)


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
