import scipy.sparse as sp
from src.Dofs import Dofs


class Observ:
    def __init__(self, Lat):

        self.Lat = Lat

        self.Oscurr_str = []  # on-site spin current
        self.Oscurrz = sp.eye(2 ** self.Lat.Nsite, dtype=complex) * 0
        self.Oscurry = sp.eye(2 ** self.Lat.Nsite, dtype=complex) * 0
        self.Oscurrx = sp.eye(2 ** self.Lat.Nsite, dtype=complex) * 0

        self.Tscurr_str = [[], [], []]  # total spin current in x,y,z
        self.Tscurrz = sp.eye(2 ** self.Lat.Nsite, dtype=complex) * 0
        self.Tscurry = sp.eye(2 ** self.Lat.Nsite, dtype=complex) * 0
        self.Tscurrx = sp.eye(2 ** self.Lat.Nsite, dtype=complex) * 0

    def OscurrBuild(self, site):  # onsite spin current

        Lat = self.Lat
        if Lat.Model == "Kitaev":
            # print("Lat.Model", Lat.Model)
            Spins = Dofs("SpinHalf")
            Sx = Spins.Sx
            Sy = Spins.Sy
            Sz = Spins.Sz
            I = Spins.I

            nn_ = Lat.nn_[site, :]  # nn_[0]: i+x; nn_[1]: i+y; nn_[2]: i+z
            # print(nn_)

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
            self.Oscurrx += Lat.Kzz * sp.kron(sp.kron(sp.kron(sp.kron(ida, Sy), idm), Sz), idb)
            stringx = "Kz*Sy[" + str(site) + "]*" + "Sz[" + str(nn_[2]) + "]*|gs>"  # "js_x["+str(site)+"] = "+

            # 2nd term -- i + y
            indxS = min(site, nn_[1])
            indxL = max(site, nn_[1])
            ida = sp.eye(2 ** indxS)
            idm = sp.eye(2 ** (indxL - indxS - 1))
            idb = sp.eye(2 ** (Lat.Nsite - indxL - 1))
            self.Oscurrx -= Lat.Kyy * sp.kron(sp.kron(sp.kron(sp.kron(ida, Sz), idm), Sy), idb)
            stringx += " - Ky*Sz[" + str(site) + "]*" + "Sy[" + str(nn_[1]) + "]*|gs>"

            # 3rd & 4th term
            ida = sp.eye(2 ** site)
            idb = sp.eye(2 ** (Lat.Nsite - site - 1))
            self.Oscurrx += Lat.Hy * sp.kron(ida, sp.kron(Sz, idb))
            self.Oscurrx -= Lat.Hz * sp.kron(ida, sp.kron(Sy, idb))
            stringx += " + Hy*Sz[" + str(site) + "]*|gs>"
            stringx += " - Hz*Sy[" + str(site) + "]*|gs>"

            self.Oscurr_str.append(stringx)
            # print(stringx)

            # ------------------------ current in y direction ---------------------------
            # 1st term -- i + x
            indxS = min(site, nn_[0])
            indxL = max(site, nn_[0])
            ida = sp.eye(2 ** indxS)
            idm = sp.eye(2 ** (indxL - indxS - 1))
            idb = sp.eye(2 ** (Lat.Nsite - indxL - 1))
            self.Oscurry += Lat.Kxx * sp.kron(sp.kron(sp.kron(sp.kron(ida, Sz), idm), Sx), idb)
            stringy = "Kx*Sz[" + str(site) + "]*" + "Sx[" + str(nn_[0]) + "]*|gs>"  # "js_y[" + str(site) + "] = " +

            # 2nd term -- i + z
            indxS = min(site, nn_[2])
            indxL = max(site, nn_[2])
            ida = sp.eye(2 ** indxS)
            idm = sp.eye(2 ** (indxL - indxS - 1))
            idb = sp.eye(2 ** (Lat.Nsite - indxL - 1))
            self.Oscurry -= Lat.Kzz * sp.kron(sp.kron(sp.kron(sp.kron(ida, Sx), idm), Sz), idb)
            stringy += " - Kz*Sx[" + str(site) + "]*" + "Sz[" + str(nn_[2]) + "]*|gs>"

            # 3rd & 4th term
            ida = sp.eye(2 ** site)
            idb = sp.eye(2 ** (Lat.Nsite - site - 1))
            self.Oscurry -= Lat.Hx * sp.kron(ida, sp.kron(Sz, idb))
            self.Oscurry += Lat.Hz * sp.kron(ida, sp.kron(Sx, idb))
            stringy += " - Hx*Sz[" + str(site) + "]*|gs>"
            stringy += " + Hz*Sx[" + str(site) + "]*|gs>"

            self.Oscurr_str.append(stringy)
            # print(stringy)

            # ------------------------ current in z direction ---------------------------
            # 1st term -- i + y
            indxS = min(site, nn_[1])
            indxL = max(site, nn_[1])
            ida = sp.eye(2 ** indxS)
            idm = sp.eye(2 ** (indxL - indxS - 1))
            idb = sp.eye(2 ** (Lat.Nsite - indxL - 1))
            self.Oscurrz += Lat.Kyy * sp.kron(sp.kron(sp.kron(sp.kron(ida, Sx), idm), Sy), idb)
            stringz = "Ky*Sx[" + str(site) + "]*" + "Sy[" + str(nn_[1]) + "]*|gs>"  # "js_z[" + str(site) + "] = " +

            # 2nd term -- i + x
            indxS = min(site, nn_[0])
            indxL = max(site, nn_[0])
            ida = sp.eye(2 ** indxS)
            idm = sp.eye(2 ** (indxL - indxS - 1))
            idb = sp.eye(2 ** (Lat.Nsite - indxL - 1))
            self.Oscurrz -= Lat.Kxx * sp.kron(sp.kron(sp.kron(sp.kron(ida, Sy), idm), Sx), idb)
            stringz += " - Kx*Sy[" + str(site) + "]*" + "Sx[" + str(nn_[0]) + "]*|gs>"

            # 3rd & 4th term
            ida = sp.eye(2 ** site)
            idb = sp.eye(2 ** (Lat.Nsite - site - 1))
            self.Oscurrz += Lat.Hx * sp.kron(ida, sp.kron(Sy, idb))
            self.Oscurrz -= Lat.Hy * sp.kron(ida, sp.kron(Sx, idb))
            stringz += " + Hx*Sy[" + str(site) + "]*|gs>"
            stringz += " - Hy*Sx[" + str(site) + "]*|gs>"

            self.Oscurr_str.append(stringz)
            # print(stringz)

            return self.Oscurrx, self.Oscurry, self.Oscurrz

        if Lat.Model == "Heisenberg":
            pass

    def TscurrBuild(self):

        Lat = self.Lat

        self.Tscurrx *= 0
        self.Tscurry *= 0
        self.Tscurrz *= 0
        self.Tscurr_str = [[], [], []]  # jx, jy, jz

        if Lat.Model == "Kitaev":

            for site in range(0, Lat.Nsite):
                Oscurrxtmp, Oscurrytmp, Oscurrztmp = self.OscurrBuild(site)
                self.Tscurrx += Oscurrxtmp
                self.Tscurry += Oscurrytmp
                self.Tscurrz += Oscurrztmp
                self.Tscurr_str[0].append(self.Oscurr_str[0])
                self.Tscurr_str[1].append(self.Oscurr_str[1])
                self.Tscurr_str[2].append(self.Oscurr_str[2])

        elif Lat.Model == "Heisenberg":
            pass
        else:
            pass

        self.Tscurr_str[0] = " + ".join(self.Tscurr_str[0])
        self.Tscurr_str[1] = " + ".join(self.Tscurr_str[1])
        self.Tscurr_str[2] = " + ".join(self.Tscurr_str[2])
        return self.Tscurrx, self.Tscurry, self.Tscurrz

# para = Parameter("../input.inp")
# Lat = Lattice(para)
# ob = Observ(Lat)
# #Oscurrx, Oscurry, Oscurrz = ob.OscurrBuild(0)
# Tscurrx, Tscurry, Tscurrz = ob.TscurrBuild()
# print("\n")
# #print(*ob.Oscurr_str, sep="\n")
# print("\n", *ob.Tscurr_str, sep="\n")
#
# #print("\n", " + ".join(ob.Tscurr_str[0]))
# #print("\n", " + ".join(ob.Tscurr_str[1]))
# #print("\n", " + ".join(ob.Tscurr_str[2]))