import scipy.sparse as sp
import numpy as np
from src.Parameter import Parameter
from src.Hamiltonian import Hamiltonian
from src.Lattice import Lattice
from src.Dofs import Dofs


class Observ:
    def __init__(self, para, Lat):

        self.Oscurr_str = []  # on-site spin current
        self.Oscurrz = sp.eye(2 ** Lat.Nsite, dtype=complex) * 0
        self.Oscurry = sp.eye(2 ** Lat.Nsite, dtype=complex) * 0
        self.Oscurrx = sp.eye(2 ** Lat.Nsite, dtype=complex) * 0

        self.Tscurr_str = []  # total spin current
        self.Tscurrz = sp.eye(2 ** Lat.Nsite, dtype=complex) * 0
        self.Tscurry = sp.eye(2 ** Lat.Nsite, dtype=complex) * 0
        self.Tscurrx = sp.eye(2 ** Lat.Nsite, dtype=complex) * 0

    def OscurrBuild(self, site):  # onsite spin current

        if Lat.Model == "Kitaev":
            #print("Lat.Model", Lat.Model)
            Spins = Dofs("SpinHalf")
            Sx = Spins.Sx
            Sy = Spins.Sy
            Sz = Spins.Sz
            I = Spins.I

            nn_ = Lat.nn_[site, :]  # nn_[0]: i+x; nn_[1]: i+y; nn_[2]: i+z
            print(nn_)

            self.Oscurrx *= 0
            self.Oscurry *= 0
            self.Oscurrz *= 0
            self.Oscurr_str = []

            # ------------------------ current in x direction ---------------------------
            # 1st term -- i + z
            indxS = min(site, nn_[2])
            indxL = max(site, nn_[2])
            ida = sp.eye(2 ** indxS)
            idm = sp.eye(2 ** (indxL - indxS - 1))
            idb = sp.eye(2 ** (Lat.Nsite - indxL - 1))
            self.Oscurrx += para.Kzz * sp.kron(sp.kron(sp.kron(sp.kron(ida, Sy), idm), Sz), idb)
            stringx = "Kz*Sy[" + str(site) + "]"+"Sz["+str(nn_[2])+"]"  # "js_x["+str(site)+"] = "+

            # 2nd term -- i + y
            indxS = min(site, nn_[1])
            indxL = max(site, nn_[1])
            ida = sp.eye(2 ** indxS)
            idm = sp.eye(2 ** (indxL - indxS - 1))
            idb = sp.eye(2 ** (Lat.Nsite - indxL - 1))
            self.Oscurrx -= para.Kyy * sp.kron(sp.kron(sp.kron(sp.kron(ida, Sz), idm), Sy), idb)
            stringx += " - Ky*Sz[" + str(site) + "]"+"Sy["+str(nn_[1])+"]"

            # 3rd & 4th term
            ida = sp.eye(2 ** site)
            idb = sp.eye(2 ** (Lat.Nsite - site - 1))
            self.Oscurrx += para.Hy * sp.kron(ida, sp.kron(Sz, idb))
            self.Oscurrx -= para.Hz * sp.kron(ida, sp.kron(Sy, idb))
            stringx += " + Hy*Sz["+str(site)+"]"
            stringx += " - Hz*Sy[" + str(site) + "]"

            self.Oscurr_str.append(stringx)
            #print(stringx)

            # ------------------------ current in y direction ---------------------------
            # 1st term -- i + x
            indxS = min(site, nn_[0])
            indxL = max(site, nn_[0])
            ida = sp.eye(2 ** indxS)
            idm = sp.eye(2 ** (indxL - indxS - 1))
            idb = sp.eye(2 ** (Lat.Nsite - indxL - 1))
            self.Oscurry += para.Kxx * sp.kron(sp.kron(sp.kron(sp.kron(ida, Sz), idm), Sx), idb)
            stringy = "Kx*Sz[" + str(site) + "]" + "Sx[" + str(nn_[0]) + "]"  # "js_y[" + str(site) + "] = " +

            # 2nd term -- i + z
            indxS = min(site, nn_[2])
            indxL = max(site, nn_[2])
            ida = sp.eye(2 ** indxS)
            idm = sp.eye(2 ** (indxL - indxS - 1))
            idb = sp.eye(2 ** (Lat.Nsite - indxL - 1))
            self.Oscurry -= para.Kzz * sp.kron(sp.kron(sp.kron(sp.kron(ida, Sx), idm), Sz), idb)
            stringy += " - Kz*Sx[" + str(site) + "]" + "Sz[" + str(nn_[2]) + "]"

            # 3rd & 4th term
            ida = sp.eye(2 ** site)
            idb = sp.eye(2 ** (Lat.Nsite - site - 1))
            self.Oscurry -= para.Hx * sp.kron(ida, sp.kron(Sz, idb))
            self.Oscurry += para.Hz * sp.kron(ida, sp.kron(Sx, idb))
            stringy += " - Hx*Sz["+str(site)+"]"
            stringy += " + Hz*Sx[" + str(site) + "]"

            self.Oscurr_str.append(stringy)
            #print(stringy)

            # ------------------------ current in z direction ---------------------------
            # 1st term -- i + y
            indxS = min(site, nn_[1])
            indxL = max(site, nn_[1])
            ida = sp.eye(2 ** indxS)
            idm = sp.eye(2 ** (indxL - indxS - 1))
            idb = sp.eye(2 ** (Lat.Nsite - indxL - 1))
            self.Oscurrz += para.Kyy * sp.kron(sp.kron(sp.kron(sp.kron(ida, Sx), idm), Sy), idb)
            stringz = "Ky*Sx[" + str(site) + "]" + "Sy[" + str(nn_[1]) + "]"  # "js_z[" + str(site) + "] = " +

            # 2nd term -- i + x
            indxS = min(site, nn_[0])
            indxL = max(site, nn_[0])
            ida = sp.eye(2 ** indxS)
            idm = sp.eye(2 ** (indxL - indxS - 1))
            idb = sp.eye(2 ** (Lat.Nsite - indxL - 1))
            self.Oscurrz -= para.Kxx * sp.kron(sp.kron(sp.kron(sp.kron(ida, Sy), idm), Sx), idb)
            stringz += " - Kx*Sy[" + str(site) + "]" + "Sx[" + str(nn_[0]) + "]"

            # 3rd & 4th term
            ida = sp.eye(2 ** site)
            idb = sp.eye(2 ** (Lat.Nsite - site - 1))
            self.Oscurrz += para.Hx * sp.kron(ida, sp.kron(Sy, idb))
            self.Oscurrz -= para.Hy * sp.kron(ida, sp.kron(Sx, idb))
            stringz += " + Hx*Sy[" + str(site) + "]"
            stringz += " - Hy*Sx[" + str(site) + "]"

            self.Oscurr_str.append(stringz)
            #print(stringz)

            return self.Oscurrx, self.Oscurry, self.Oscurrz

        if Lat.Model == "Heisenberg":
            pass

    def TscurrBuild(self):

        self.Tscurrx *= 0
        self.Tscurry *= 0
        self.Tscurrz *= 0

        if Lat.Model == "Kitaev":

            for site in range(0, Lat.Nsite):
                Oscurrxtmp, Oscurrytmp, Oscurrztmp = self.OscurrBuild(site)
                self.Tscurrx += Oscurrxtmp
                self.Tscurry += Oscurrytmp
                self.Tscurrz += Oscurrztmp



        elif Lat.Model == "Heisenberg":
            pass
        else:
            pass

        return self.Tscurrx, self.Tscurry, self.Tscurrz


para = Parameter("../input.inp")
Lat = Lattice(para)
ob = Observ(para, Lat)
#Oscurrx, Oscurry, Oscurrz = ob.OscurrBuild(0)
Tscurrx, Tscurry, Tscurrz = ob.TscurrBuild()
print("\n")
print(*ob.Oscurr_str, sep="\n")

