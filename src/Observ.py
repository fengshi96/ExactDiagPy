import scipy.sparse as sp
import numpy as np
from src.Dofs import Dofs


def matele(A, Op, B):
    """
    Calculate matrix element: <A|Op|B>
    """
    bra = A.conj().T
    ket = B
    if bra.shape != ket.shape:
        print("bra, ket.shape = ", bra.shape, ket.shape)
        raise ValueError("mel(bra,Op,ket): bra & ket must be vectors of the same dimension")
    if bra.shape[0] != Op.shape[0] or ket.shape[0] != Op.shape[0]:
        raise ValueError("mel(bra,Op,ket): Dimension mismatch")

    # return np.dot(bra, np.dot(Op, ket))
    return bra.dot(Op.dot(ket))


class Observ:
    """
    This class is designed for construction of all kinds of Observable operators and their evaluation
    """

    def __init__(self, Lat):

        self.Lat = Lat

        self.Oscurr_str = []  # on-site spin current string
        self.Tscurr_str = [[], [], []]  # total spin current string of x,y,z

    def LscurrBuild(self, site):  # onsite spin current

        Lat = self.Lat

        self.Oscurr_str = []  # on-site spin current
        Oscurrz = sp.eye(2 ** Lat.Nsite, dtype=complex) * 0
        Oscurry = sp.eye(2 ** Lat.Nsite, dtype=complex) * 0
        Oscurrx = sp.eye(2 ** Lat.Nsite, dtype=complex) * 0

        stringx = ""
        stringy = ""
        stringz = ""

        if Lat.Model == "Kitaev":
            # print("Lat.Model", Lat.Model)
            Spins = Dofs("SpinHalf")
            Sx = Spins.Sx
            Sy = Spins.Sy
            Sz = Spins.Sz
            I = Spins.I

            nn_ = Lat.nn_[site, :]  # nn_[0]: i+x; nn_[1]: i+y; nn_[2]: i+z
            # print(nn_)

            # ------------------------ current in x direction ---------------------------
            # 1st term -- i + z
            indxS = min(site, nn_[2])
            indxL = max(site, nn_[2])
            if indxS >= 0:
                ida = sp.eye(2 ** indxS)
                idm = sp.eye(2 ** (indxL - indxS - 1))
                idb = sp.eye(2 ** (Lat.Nsite - indxL - 1))
                Oscurrx += Lat.Kzz * sp.kron(sp.kron(sp.kron(sp.kron(ida, Sy), idm), Sz), idb)
                stringx = "Kz*Sy[" + str(site) + "]*" + "Sz[" + str(nn_[2]) + "]*|gs>"  # "js_x["+str(site)+"] = "+

            # 2nd term -- i + y
            indxS = min(site, nn_[1])
            indxL = max(site, nn_[1])
            if indxS >= 0:
                ida = sp.eye(2 ** indxS)
                idm = sp.eye(2 ** (indxL - indxS - 1))
                idb = sp.eye(2 ** (Lat.Nsite - indxL - 1))
                Oscurrx -= Lat.Kyy * sp.kron(sp.kron(sp.kron(sp.kron(ida, Sz), idm), Sy), idb)
                stringx += " - Ky*Sz[" + str(site) + "]*" + "Sy[" + str(nn_[1]) + "]*|gs>"

            # 3rd & 4th term
            ida = sp.eye(2 ** site)
            idb = sp.eye(2 ** (Lat.Nsite - site - 1))
            Oscurrx += Lat.Hy * sp.kron(ida, sp.kron(Sz, idb))
            Oscurrx -= Lat.Hz * sp.kron(ida, sp.kron(Sy, idb))
            stringx += " + Hy*Sz[" + str(site) + "]*|gs>"
            stringx += " - Hz*Sy[" + str(site) + "]*|gs>"

            self.Oscurr_str.append(stringx)
            # print(stringx)

            # ------------------------ current in y direction ---------------------------
            # 1st term -- i + x
            indxS = min(site, nn_[0])
            indxL = max(site, nn_[0])
            if indxS >= 0:  # if nn_[0] is not out of boundary
                ida = sp.eye(2 ** indxS)
                idm = sp.eye(2 ** (indxL - indxS - 1))
                idb = sp.eye(2 ** (Lat.Nsite - indxL - 1))
                Oscurry += Lat.Kxx * sp.kron(sp.kron(sp.kron(sp.kron(ida, Sz), idm), Sx), idb)
                stringy = "Kx*Sz[" + str(site) + "]*" + "Sx[" + str(nn_[0]) + "]*|gs>"  # "js_y[" + str(site) + "] = " +

            # 2nd term -- i + z
            indxS = min(site, nn_[2])
            indxL = max(site, nn_[2])
            if indxS >= 0:
                ida = sp.eye(2 ** indxS)
                idm = sp.eye(2 ** (indxL - indxS - 1))
                idb = sp.eye(2 ** (Lat.Nsite - indxL - 1))
                Oscurry -= Lat.Kzz * sp.kron(sp.kron(sp.kron(sp.kron(ida, Sx), idm), Sz), idb)
                stringy += " - Kz*Sx[" + str(site) + "]*" + "Sz[" + str(nn_[2]) + "]*|gs>"

            # 3rd & 4th term
            ida = sp.eye(2 ** site)
            idb = sp.eye(2 ** (Lat.Nsite - site - 1))
            Oscurry -= Lat.Hx * sp.kron(ida, sp.kron(Sz, idb))
            Oscurry += Lat.Hz * sp.kron(ida, sp.kron(Sx, idb))
            stringy += " - Hx*Sz[" + str(site) + "]*|gs>"
            stringy += " + Hz*Sx[" + str(site) + "]*|gs>"

            self.Oscurr_str.append(stringy)
            # print(stringy)

            # ------------------------ current in z direction ---------------------------
            # 1st term -- i + y
            indxS = min(site, nn_[1])
            indxL = max(site, nn_[1])
            if indxS >= 0:
                ida = sp.eye(2 ** indxS)
                idm = sp.eye(2 ** (indxL - indxS - 1))
                idb = sp.eye(2 ** (Lat.Nsite - indxL - 1))
                Oscurrz += Lat.Kyy * sp.kron(sp.kron(sp.kron(sp.kron(ida, Sx), idm), Sy), idb)
                stringz = "Ky*Sx[" + str(site) + "]*" + "Sy[" + str(nn_[1]) + "]*|gs>"  # "js_z[" + str(site) + "] = " +

            # 2nd term -- i + x
            indxS = min(site, nn_[0])
            indxL = max(site, nn_[0])
            if indxS >= 0:
                ida = sp.eye(2 ** indxS)
                idm = sp.eye(2 ** (indxL - indxS - 1))
                idb = sp.eye(2 ** (Lat.Nsite - indxL - 1))
                Oscurrz -= Lat.Kxx * sp.kron(sp.kron(sp.kron(sp.kron(ida, Sy), idm), Sx), idb)
                stringz += " - Kx*Sy[" + str(site) + "]*" + "Sx[" + str(nn_[0]) + "]*|gs>"

            # 3rd & 4th term
            ida = sp.eye(2 ** site)
            idb = sp.eye(2 ** (Lat.Nsite - site - 1))
            Oscurrz += Lat.Hx * sp.kron(ida, sp.kron(Sy, idb))
            Oscurrz -= Lat.Hy * sp.kron(ida, sp.kron(Sx, idb))
            stringz += " + Hx*Sy[" + str(site) + "]*|gs>"
            stringz += " - Hy*Sx[" + str(site) + "]*|gs>"

            self.Oscurr_str.append(stringz)
            # print(stringz)

            return Oscurrx, Oscurry, Oscurrz

        if Lat.Model == "Heisenberg":
            pass

    def TscurrBuild(self):

        Lat = self.Lat

        self.Tscurr_str = [[], [], []]  # total spin current in x,y,z
        Tscurrz = sp.eye(2 ** Lat.Nsite, dtype=complex) * 0
        Tscurry = sp.eye(2 ** Lat.Nsite, dtype=complex) * 0
        Tscurrx = sp.eye(2 ** Lat.Nsite, dtype=complex) * 0

        if Lat.Model == "Kitaev":

            for site in range(0, Lat.Nsite):
                Oscurrxtmp, Oscurrytmp, Oscurrztmp = self.LscurrBuild(site)
                Tscurrx += Oscurrxtmp
                Tscurry += Oscurrytmp
                Tscurrz += Oscurrztmp
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
        return Tscurrx, Tscurry, Tscurrz

    # ----------------------------- Local dofs ----------------------------------
    # ----------------------------- Local dofs----------------------------------
    # ----------------------------- Build local operators ----------------------------------
    def LSxBuild(self, site):
        """
        Build local spin x operator in full Hilbert space
        """
        sx = Dofs("SpinHalf").Sx
        ida = sp.eye(2 ** site)
        idb = sp.eye(2 ** (self.Lat.Nsite - site - 1))
        Sx = sp.kron(ida, sp.kron(sx, idb))
        return Sx

    def LSyBuild(self, site):
        """
        Build local spin y operator in full Hilbert space
        """
        sy = Dofs("SpinHalf").Sy
        ida = sp.eye(2 ** site)
        idb = sp.eye(2 ** (self.Lat.Nsite - site - 1))
        Sy = sp.kron(ida, sp.kron(sy, idb))
        return Sy

    def LSzBuild(self, site):
        """
        Build local spin z operator in full Hilbert space
        """
        sz = Dofs("SpinHalf").Sz
        ida = sp.eye(2 ** site)
        idb = sp.eye(2 ** (self.Lat.Nsite - site - 1))
        Sz = sp.kron(ida, sp.kron(sz, idb))
        return Sz

    # ----------------------------- evaluate local operators ----------------------------------
    def mLocal(self, state, Opstr, site):
        """
        Measure local variables. Opstr = Sx, Sy, Sz
        """
        if Opstr == "Sx":
            self.mLocalSx(state, site)
        elif Opstr == "Sy":
            self.mLocalSy(state, site)
        elif Opstr == "Sz":
            self.mLocalSz(state, site)
        elif Opstr == "S":  # local spin in x, y, z
            self.mLocalS(state, site)
        elif Opstr == "Js":  # local spin currents in x, y, z
            self.mLocalJs(state, site)
        else:
            print("Opstr = ", Opstr)
            raise ValueError("Opstr must be one of [S, (Sx, Sy, Sz); J]")

    # ------------ local spin measurements -----------------
    def mLocalSx(self, state, site):
        """
        Measure local <Sx>
        """
        Sx = self.LSxBuild(site)
        Sxeval = matele(state, Sx, state)
        return Sxeval

    def mLocalSy(self, state, site):
        """
        Measure local <Sy>
        """
        Sy = self.LSyBuild(site)
        Syeval = matele(state, Sy, state)
        return Syeval

    def mLocalSz(self, state, site):
        """
        Measure local <Sz>
        """
        Sz = self.LSzBuild(site)
        Szeval = matele(state, Sz, state)
        return Szeval

    def mLocalS(self, state, site):
        """
        Measure together local <Sx>, <Sy>, <Sz>
        """
        Sxeval = self.mLocalSx(state, site)
        Syeval = self.mLocalSy(state, site)
        Szeval = self.mLocalSz(state, site)
        return Sxeval, Syeval, Szeval

    # ------------ End: local spin measurement -----------------

    # ------------ local current measurement -----------------
    def mLocalJs(self, state, site):
        """
        Measure together all local spin current <Jsx>, <Jsy>, <Jsz>
        """
        Jsx, Jsy, Jsz = self.LscurrBuild(site)
        Jsxeval = matele(state, Jsx, state)
        Jsyeval = matele(state, Jsx, state)
        Jszeval = matele(state, Jsx, state)
        return Jsxeval, Jsyeval, Jszeval

    def mLocalJe(self, state, site):
        """
        Measure together all local energy current <Jex>, <Jey>, <Jez>
        """
        pass

    # ------------ End: local current measurement -----------------

    # ----------------------------- Transport measurement ----------------------------------
    # ----------------------------- Transport measurement ----------------------------------
    # ----------------------------- Transport measurement ----------------------------------
    def SpRe(self, evals, evecs):
        """
        Measure together spin response S(\omega)
        Parameter: evals, 1d array of eigen energies
                   evecs, 2d array, with cols being eigen vectors corresponding to evals
        """
        Lat = self.Lat
        Nstates = len(evals)

        gs = evecs[:, 0]  # ground state
        Eg = evals[0]  # ground state energy

        omegasteps = 200
        domega = 0.005
        eta = 0.009

        # for each omega, define a matrix Mel_{si,mi}
        Melx = np.zeros((Lat.Nsite, Nstates), dtype=complex)  # <m|S_i^x|gs>
        Mely = np.zeros((Lat.Nsite, Nstates), dtype=complex)  # <m|S_i^y|gs>
        Melz = np.zeros((Lat.Nsite, Nstates), dtype=complex)  # <m|S_i^z|gs>

        for mi in range(0, Nstates):
            for si in range(0, Lat.Nsite):
                Sxi = self.LSxBuild(si)
                Syi = self.LSyBuild(si)
                Szi = self.LSzBuild(si)

                # <m|S_i^a|gs>
                Melx[si, mi] = matele(evecs[:, mi], Sxi, gs)
                Mely[si, mi] = matele(evecs[:, mi], Syi, gs)
                Melz[si, mi] = matele(evecs[:, mi], Szi, gs)
        # print(np.real(Melx))
        # print(np.real(Mely))
        # print(np.real(Melz))

        # elastic contribution <Sa>_i
        mSx = np.zeros(Lat.Nsite, dtype=complex)
        mSy = np.zeros(Lat.Nsite, dtype=complex)
        mSz = np.zeros(Lat.Nsite, dtype=complex)
        for si in range(0, Lat.Nsite):
            mSx[si] = self.mLocalSx(gs, si)
            mSy[si] = self.mLocalSy(gs, si)
            mSz[si] = self.mLocalSz(gs, si)

        Sr = np.zeros((omegasteps, 2), dtype=float)  # spin response
        # begin fill in vector Sr(\omega)
        omegacounter = 0
        for oi in range(0, omegasteps):
            omega = domega * oi
            Itensity = np.zeros((3, 3), dtype=complex)

            denom1 = complex(omega, -eta)
            # for each omega, define a matrix Mel_{si,mi}
            for si in range(0, Lat.Nsite):
                for mi in range(0, Nstates):
                    Em = evals[mi]
                    denom2 = complex(omega - (Em - Eg), -eta)

                    # <m|S_i^a|gs>
                    tmpx = Melx[si, mi]
                    tmpy = Mely[si, mi]
                    tmpz = Melz[si, mi]

                    # <gs|S_i^a|m><m|S_i^b|gs>
                    tmpxx = tmpx.conjugate() * tmpx  # Six Six
                    tmpxy = tmpx.conjugate() * tmpy  # Six Siy
                    tmpxz = tmpx.conjugate() * tmpz  # Six Siz

                    tmpyy = tmpy.conjugate() * tmpy  # Siy Siy
                    tmpyz = tmpy.conjugate() * tmpz  # Siy Siz
                    tmpzz = tmpz.conjugate() * tmpz  # Siz Siz

                    # update polarization matrix
                    Itensity[0, 0] += tmpxx / denom2
                    Itensity[0, 1] += tmpxy / denom2
                    Itensity[0, 2] += tmpxz / denom2
                    Itensity[1, 1] += tmpyy / denom2
                    Itensity[1, 2] += tmpyz / denom2
                    Itensity[2, 2] += tmpzz / denom2

                # remove elastic contribution
                Itensity[0, 0] -= mSx[si].conjugate() * mSx[si] / denom1
                Itensity[0, 1] -= mSx[si].conjugate() * mSy[si] / denom1
                Itensity[0, 2] -= mSx[si].conjugate() * mSz[si] / denom1
                Itensity[1, 1] -= mSy[si].conjugate() * mSy[si] / denom1
                Itensity[1, 2] -= mSy[si].conjugate() * mSz[si] / denom1
                Itensity[2, 2] -= mSz[si].conjugate() * mSz[si] / denom1

            Sr[omegacounter, 0] = round(omega, 4)
            Sr[omegacounter, 1] = Itensity.sum().imag / np.pi
            omegacounter += 1
        #print(Itensity)

        return Sr

    def Econd(self, state, site):
        """
        Measure together energy conductivity
        """
        pass

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
