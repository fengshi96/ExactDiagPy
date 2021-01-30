import numpy as np
import math as m
from src.Helper import matprint, vecprint


class Lattice:

    def __init__(self, para):

        # Shared attributes
        self.LLX = para.LLX  # Number of unit cells in x
        self.LLY = para.LLY  # Number of unit cells in y
        self.IsPeriodicX = para.IsPeriodicX  # PBC (1) or OBC (0)
        self.IsPeriodicY = para.IsPeriodicY
        self.Model = para.Model  # Name of the model.
        if para.Option is not None:
            if "EE" in para.Option:
                self.SysIndx = para.SysIndx

        # Model-dependent attributes
        if para.Model == "Kitaev" or para.Model == "Heisenberg_Honeycomb":
            self.Nsite = self.LLX * self.LLY * 2
            self.Hx = para.Hx; self.Kxx = para.Kxx
            self.Hy = para.Hy; self.Kyy = para.Kyy
            self.Hz = para.Hz; self.Kzz = para.Kzz
            self.indx_ = np.zeros(self.Nsite, dtype=int)  # x coordinate in mesh
            self.indy_ = np.zeros(self.Nsite, dtype=int)
            self.Number1neigh = 3  # number of nearest neighbors
            self.nn_ = -np.ones((self.Nsite, self.Number1neigh), dtype=int)  # nearest neighbor matrix
            self.mesh_ = -np.ones((self.LLX * 2 + self.LLY, self.LLY * 2), dtype=int)  # declare mesh of the lattice
            self.BuildHoneycomb()  # build attributes in honeycomb lattice
        elif para.Model == "Heisenberg_Square":
            self.Nsite = self.LLX * self.LLY
            self.Hx = para.Hx; self.Kxx = para.Kxx
            self.Hy = para.Hy; self.Kyy = para.Kyy
            self.Hz = para.Hz; self.Kzz = para.Kzz
            self.indx_ = np.zeros(self.Nsite, dtype=int)
            self.indy_ = np.zeros(self.Nsite, dtype=int)
            self.Number1neigh = 4
            if self.LLX == 1 or self.LLY == 1:
                self.Number1neigh = 2
            self.nn_ = -np.ones((self.Nsite, self.Number1neigh), dtype=int)
            self.mesh_ = -np.ones((self.LLX, self.LLY), dtype=int)
            self.BuildSquare()  # build attributes in square lattice
        elif para.Model == "AKLT":
            self.Nsite = self.LLX
            self.Hx = para.Hx; self.Kxx1 = para.Kxx; self.Kxx2 = para.Kxx / 3.0
            self.Hy = para.Hy; self.Kyy1 = para.Kyy; self.Kyy2 = para.Kyy / 3.0
            self.Hz = para.Hz; self.Kzz1 = para.Kzz; self.Kzz2 = para.Kzz / 3.0
            self.indx_ = np.zeros(self.Nsite, dtype=int)
            self.Number1neigh = 2
            self.nn_ = -np.ones((self.Nsite, self.Number1neigh), dtype=int)
            self.mesh_ = -np.ones(self.LLX, dtype=int)
            self.BuildBLBQChain()  # build attributes in square lattice
        elif para.Model == "TFIM":
            self.Nsite = self.LLX
            self.Hx = para.Hx; self.Kxx1 = 0; self.Kxx2 = 0
            self.Hy = 0; self.Kyy1 = 0; self.Kyy2 = 0
            self.Hz = 0; self.Kzz1 = para.Kzz; self.Kzz2 = 0
            self.indx_ = np.zeros(self.Nsite, dtype=int)
            self.Number1neigh = 2
            self.nn_ = -np.ones((self.Nsite, self.Number1neigh), dtype=int)
            self.mesh_ = -np.ones(self.LLX, dtype=int)
            self.BuildBLBQChain()  # build attributes in square lattice
        elif para.Model == "Bose_Hubbard":
            self.Nsite = self.LLX * self.LLY
            self.t = para.t; self.U = para.U; self.mu = para.mu
            self.maxOccupation = para.maxOccupation
            self.indx_ = np.zeros(self.Nsite, dtype=int)
            self.indy_ = np.zeros(self.Nsite, dtype=int)
            self.Number1neigh = 4
            if self.LLX == 1 or self.LLY == 1:
                self.Number1neigh = 2
            self.nn_ = -np.ones((self.Nsite, self.Number1neigh), dtype=int)
            self.mesh_ = -np.ones((self.LLX, self.LLY), dtype=int)
            self.BuildSquare()  # build attributes in square lattice

        else:
            raise ValueError("Model not supported yet")

    def BuildBLBQChain(self):
        """
        Construct Spin-1 BLBQ Chain and nearest neighbor matrix
        Neighbor Label:  —— (1) ——  i  ——  (0) ——
        """
        print("[Lattice.py] building 1D BLBQ Chain...")
        counter = 0
        for ix in range(0, self.LLX):
            self.mesh_[ix] = counter
            self.indx_[counter] = ix
            counter += 1
        vecprint(self.mesh_)

        print("\n[Lattice.py] Looking for nearest neighbors...")
        for i in range(0, self.Nsite):
            ix = self.indx_[i]

            # ----------------------------OBC-----------------------------------
            # +x - 1neighbor 0
            if ix + 1 < self.LLX:  # if not at the bottom edge
                jx = ix + 1
                j = self.mesh_[jx]  # (+x)-neighbor index
                self.nn_[i, 0] = j

            # -x - 1neighbor 1
            if ix > 0:  # if not at the top edge
                jx = ix - 1
                j = self.mesh_[jx]  # (-x)-neighbor index
                self.nn_[i, 1] = j

            # ----------------------------OBC-----------------------------------

            # --------------------------Apply PBC-------------------------------
            if self.IsPeriodicX:
                # +x - 1neighbor 0
                if ix == self.LLX - 1:  # bottom edge
                    jx = 0
                    j = self.mesh_[jx]
                    self.nn_[i, 0] = j

                # -x - 1neighbor 1
                if ix == 0:  # top edge
                    jx = self.LLX - 1
                    j = self.mesh_[jx]
                    self.nn_[i, 1] = j

        matprint(self.nn_)

    def BuildHoneycomb(self):
        """
        Construct Honeycomb Lattice mesh and nearest neighbor matrix
        """

        print("[Lattice.py] building Honeycomb lattice...")
        scalex = 2
        scaley = 4.0 / m.sqrt(3)
        t1 = [1.0 * scalex, 0]
        t2 = [0.5 * scalex, m.sqrt(3) / 2.0 * scaley]

        xv = 0
        counter = 0

        for i in range(0, self.LLX):
            if i != 0:
                xv += t1[0]

            for j in range(0, self.LLY):
                xa = xv + j * t2[0]
                xb = (xv + 1) + j * t2[0]
                ya = j * t2[1]
                yb = 1 + j * t2[1]

                xa = int(xa)
                xb = int(xb)
                ya = int(ya)
                yb = int(yb)

                self.indx_[counter] = xa
                self.indy_[counter] = ya
                self.mesh_[xa, ya] = counter
                counter += 1

                self.indx_[counter] = xb
                self.indy_[counter] = yb
                self.mesh_[xb, yb] = counter
                counter += 1
        matprint(self.mesh_)

        print("\n[Lattice.py] Looking for nearest neighbors...")
        xmax = max(self.indx_)
        ymax = max(self.indy_)

        for i in range(0, self.Nsite):
            ix = self.indx_[i]  # coordinate of n-th site in matrix
            iy = self.indy_[i]

            # ----------------------------OBC-----------------------------------

            # n.n in x-bond
            jx = ix + 1
            jy = iy + 1  # move 1 step in x = (1,1) direction
            if jx <= xmax and jy <= ymax and self.mesh_[jx, jy] != -1:
                j = self.mesh_[jx, jy]  # site index of n.n. in x direction
                self.nn_[i, 0] = j
                self.nn_[j, 0] = i

            # n.n in y-bond
            jx = ix + 1
            jy = iy - 1  # move 1 step in x = (1,1) direction
            if jx <= xmax and ymax >= jy >= 0 and self.mesh_[jx, jy] != -1:
                j = self.mesh_[jx, jy]  # site index of n.n. in x direction
                self.nn_[i, 1] = j
                self.nn_[j, 1] = i

            # n.n in z-bond
            jx = ix
            jy = iy + 1  # move 1 step in x = (1,1) direction
            if jx <= xmax and jy <= ymax and self.mesh_[jx, jy] != -1:
                j = self.mesh_[jx, jy]  # site index of n.n. in x direction
                self.nn_[i, 2] = j
                self.nn_[j, 2] = i

            # ----------------------------OBC-----------------------------------

            # --------------------------Apply PBC-------------------------------

            if self.IsPeriodicY:
                # z-bond
                jx = ix - self.LLY
                jy = 0
                if jx >= 0 and iy == ymax and self.mesh_[jx, jy] != -1:
                    j = self.mesh_[jx, jy]
                    self.nn_[i, 2] = j
                    self.nn_[j, 2] = i

            if self.IsPeriodicX:
                # y-bond
                jx = ix + 2 * self.LLX - 1
                jy = iy + 1
                if jx <= xmax and iy <= ymax and iy % 2 == 0 and self.mesh_[jx, jy] != -1:
                    j = self.mesh_[jx, jy]
                    self.nn_[i, 1] = j
                    self.nn_[j, 1] = i

        matprint(self.nn_)

    def BuildSquare(self):
        """
        Construct Square Lattice mesh and nearest neighbor matrix
        ----------------------------------------------------------
                                 |
                                (1)
                                 |
                      —— (3) ——  i  ——  (2) ——
                                 |
                                (0)
                                 |
        ----------------------------------------------------------
        """

        print("[Lattice.py] building Square lattice...")
        # ----------------------------OBC-----------------------------------
        counter = 0
        for ix in range(0, self.LLX):
            for iy in range(0, self.LLY):
                self.mesh_[ix, iy] = counter
                self.indx_[counter] = ix
                self.indy_[counter] = iy
                counter += 1
        matprint(self.mesh_)

        print("\n[Lattice.py] Looking for nearest neighbors...")
        for i in range(0, self.Nsite):
            ix = self.indx_[i]
            iy = self.indy_[i]

            # +x - 1neighbor 0
            if ix + 1 < self.LLX:  # if not at the bottom edge
                jy = iy
                jx = ix + 1
                j = self.mesh_[jx, jy]  # (+x)-neighbor index
                self.nn_[i, 0] = j

            # -x - 1neighbor 1
            if ix > 0:  # if not at the top edge
                jy = iy
                jx = ix - 1
                j = self.mesh_[jx, jy]  # (-x)-neighbor index
                self.nn_[i, 1] = j

            # +y - 1neighbor 2
            if iy + 1 < self.LLY:  # if not at the right edge
                jy = iy + 1
                jx = ix
                j = self.mesh_[jx, jy]  # (+y)-neighbor index
                self.nn_[i, 2] = j

            # -y - 1neighbor 3
            if iy > 0:  # if not at the left edge
                jy = iy - 1
                jx = ix
                j = self.mesh_[jx, jy]  # (-y)-neighbor index
                self.nn_[i, 3] = j
            # ----------------------------OBC-----------------------------------

            # --------------------------Apply PBC-------------------------------
            if self.IsPeriodicX * self.LLX == 1:
                raise ValueError("PBC is ill-defined along X when LLY = 1. For 1D X-chain Plz set OBC for Y")

            if self.IsPeriodicY * self.LLY == 1:
                raise ValueError("PBC is ill-defined along Y when LLX = 1. For 1D Y-chain Plz set OBC for X")

            if self.IsPeriodicX:
                # +x - 1neighbor 0
                if ix == self.LLX - 1:  # bottom edge
                    jx = 0
                    jy = iy
                    j = self.mesh_[jx, jy]
                    self.nn_[i, 0] = j

                # -x - 1neighbor 1
                if ix == 0:  # top edge
                    jx = self.LLX - 1
                    jy = iy
                    j = self.mesh_[jx, jy]
                    self.nn_[i, 1] = j

            if self.IsPeriodicY:
                # +y - 1neighbor 2
                if iy == self.LLY - 1:  # right edge
                    jx = ix
                    jy = 0
                    j = self.mesh_[jx, jy]
                    self.nn_[i, 2] = j

            if self.IsPeriodicY:
                # -y - 1neighbor 3
                if iy == 0:  # left edge
                    jx = ix
                    jy = self.LLY - 1
                    j = self.mesh_[jx, jy]
                    self.nn_[i, 3] = j
        matprint(self.nn_)

# param = Parameter("../input.inp")
# lat = Lattice(param)

