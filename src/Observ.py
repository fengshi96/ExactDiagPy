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

    def LscurrBuild(self, site, qm="SpinHalf"):  # onsite spin current

        if qm not in ["SpinHalf", "SpinOne"]:
            raise ValueError("Must be SpinHalf or SpinOne")

        Lat = self.Lat

        Spins = Dofs(qm)
        Sx = Spins.Sx
        Sy = Spins.Sy
        Sz = Spins.Sz
        hilbsize = Spins.hilbsize

        self.Oscurr_str = []  # on-site spin current
        Oscurrz = sp.eye(hilbsize ** Lat.Nsite, dtype=complex) * 0
        Oscurry = sp.eye(hilbsize ** Lat.Nsite, dtype=complex) * 0
        Oscurrx = sp.eye(hilbsize ** Lat.Nsite, dtype=complex) * 0

        stringx = ""
        stringy = ""
        stringz = ""

        if Lat.Model == "Kitaev":

            nn_ = Lat.nn_[site, :]  # nn_[0]: i+x; nn_[1]: i+y; nn_[2]: i+z
            # print(nn_)

            # ------------------------ current in x direction ---------------------------
            # 1st term -- i + z
            indxS = min(site, nn_[2])
            indxL = max(site, nn_[2])
            if indxS >= 0:
                ida = sp.eye(hilbsize ** indxS)
                idm = sp.eye(hilbsize ** (indxL - indxS - 1))
                idb = sp.eye(hilbsize ** (Lat.Nsite - indxL - 1))
                Oscurrx += Lat.Kzz * sp.kron(sp.kron(sp.kron(sp.kron(ida, Sy), idm), Sz), idb)
                stringx = "Kz*Sy[" + str(site) + "]*" + "Sz[" + str(nn_[2]) + "]*|gs>"  # "js_x["+str(site)+"] = "+

            # 2nd term -- i + y
            indxS = min(site, nn_[1])
            indxL = max(site, nn_[1])
            if indxS >= 0:
                ida = sp.eye(hilbsize ** indxS)
                idm = sp.eye(hilbsize ** (indxL - indxS - 1))
                idb = sp.eye(hilbsize ** (Lat.Nsite - indxL - 1))
                Oscurrx -= Lat.Kyy * sp.kron(sp.kron(sp.kron(sp.kron(ida, Sz), idm), Sy), idb)
                stringx += " - Ky*Sz[" + str(site) + "]*" + "Sy[" + str(nn_[1]) + "]*|gs>"

            # 3rd & 4th term
            ida = sp.eye(hilbsize ** site)
            idb = sp.eye(hilbsize ** (Lat.Nsite - site - 1))
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
                ida = sp.eye(hilbsize ** indxS)
                idm = sp.eye(hilbsize ** (indxL - indxS - 1))
                idb = sp.eye(hilbsize ** (Lat.Nsite - indxL - 1))
                Oscurry += Lat.Kxx * sp.kron(sp.kron(sp.kron(sp.kron(ida, Sz), idm), Sx), idb)
                stringy = "Kx*Sz[" + str(site) + "]*" + "Sx[" + str(nn_[0]) + "]*|gs>"  # "js_y[" + str(site) + "] = " +

            # 2nd term -- i + z
            indxS = min(site, nn_[2])
            indxL = max(site, nn_[2])
            if indxS >= 0:
                ida = sp.eye(hilbsize ** indxS)
                idm = sp.eye(hilbsize ** (indxL - indxS - 1))
                idb = sp.eye(hilbsize ** (Lat.Nsite - indxL - 1))
                Oscurry -= Lat.Kzz * sp.kron(sp.kron(sp.kron(sp.kron(ida, Sx), idm), Sz), idb)
                stringy += " - Kz*Sx[" + str(site) + "]*" + "Sz[" + str(nn_[2]) + "]*|gs>"

            # 3rd & 4th term
            ida = sp.eye(hilbsize ** site)
            idb = sp.eye(hilbsize ** (Lat.Nsite - site - 1))
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
                ida = sp.eye(hilbsize ** indxS)
                idm = sp.eye(hilbsize ** (indxL - indxS - 1))
                idb = sp.eye(hilbsize ** (Lat.Nsite - indxL - 1))
                Oscurrz += Lat.Kyy * sp.kron(sp.kron(sp.kron(sp.kron(ida, Sx), idm), Sy), idb)
                stringz = "Ky*Sx[" + str(site) + "]*" + "Sy[" + str(nn_[1]) + "]*|gs>"  # "js_z[" + str(site) + "] = " +

            # 2nd term -- i + x
            indxS = min(site, nn_[0])
            indxL = max(site, nn_[0])
            if indxS >= 0:
                ida = sp.eye(hilbsize ** indxS)
                idm = sp.eye(hilbsize ** (indxL - indxS - 1))
                idb = sp.eye(hilbsize ** (Lat.Nsite - indxL - 1))
                Oscurrz -= Lat.Kxx * sp.kron(sp.kron(sp.kron(sp.kron(ida, Sy), idm), Sx), idb)
                stringz += " - Kx*Sy[" + str(site) + "]*" + "Sx[" + str(nn_[0]) + "]*|gs>"

            # 3rd & 4th term
            ida = sp.eye(hilbsize ** site)
            idb = sp.eye(hilbsize ** (Lat.Nsite - site - 1))
            Oscurrz += Lat.Hx * sp.kron(ida, sp.kron(Sy, idb))
            Oscurrz -= Lat.Hy * sp.kron(ida, sp.kron(Sx, idb))
            stringz += " + Hx*Sy[" + str(site) + "]*|gs>"
            stringz += " - Hy*Sx[" + str(site) + "]*|gs>"

            self.Oscurr_str.append(stringz)
            # print(stringz)

            return Oscurrx, Oscurry, Oscurrz

        if Lat.Model == "Heisenberg":
            pass

    def TscurrBuild(self, qm="SpinHalf"):

        if qm not in ["SpinHalf", "SpinOne"]:
            raise ValueError("Must be SpinHalf or SpinOne")
        Lat = self.Lat
        hilbsize = Dofs(qm).hilbsize

        self.Tscurr_str = [[], [], []]  # total spin current in x,y,z
        Tscurrz = sp.eye(hilbsize ** Lat.Nsite, dtype=complex) * 0
        Tscurry = sp.eye(hilbsize ** Lat.Nsite, dtype=complex) * 0
        Tscurrx = sp.eye(hilbsize ** Lat.Nsite, dtype=complex) * 0

        if Lat.Model == "Kitaev":

            for site in range(0, Lat.Nsite):
                Oscurrxtmp, Oscurrytmp, Oscurrztmp = self.LscurrBuild(site, qm)
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
    def LSxBuild(self, site, qm="SpinHalf"):
        """
        Build local spin x operator in full Hilbert space
        """
        if qm not in ["SpinHalf", "SpinOne"]:
            raise ValueError("Must be SpinHalf or SpinOne")

        dof = Dofs(qm)
        sx = dof.Sx
        hilbsize = dof.hilbsize

        ida = sp.eye(hilbsize ** site)
        idb = sp.eye(hilbsize ** (self.Lat.Nsite - site - 1))
        Sx = sp.kron(ida, sp.kron(sx, idb))
        return Sx

    def LSyBuild(self, site, qm="SpinHalf"):
        """
        Build local spin y operator in full Hilbert space
        """
        if qm not in ["SpinHalf", "SpinOne"]:
            raise ValueError("Must be SpinHalf or SpinOne")

        dof = Dofs(qm)
        sy = dof.Sy
        hilbsize = dof.hilbsize

        ida = sp.eye(hilbsize ** site)
        idb = sp.eye(hilbsize ** (self.Lat.Nsite - site - 1))
        Sy = sp.kron(ida, sp.kron(sy, idb))
        return Sy

    def LSzBuild(self, site, qm="SpinHalf"):
        """
        Build local spin z operator in full Hilbert space
        """
        if qm not in ["SpinHalf", "SpinOne"]:
            raise ValueError("Must be SpinHalf or SpinOne")

        dof = Dofs(qm)
        sz = dof.Sz
        hilbsize = dof.hilbsize

        ida = sp.eye(hilbsize ** site)
        idb = sp.eye(hilbsize ** (self.Lat.Nsite - site - 1))
        Sz = sp.kron(ida, sp.kron(sz, idb))
        return Sz

    def LSpBuild(self, site, qm="SpinHalf"):
        """
        Build local spin plus operator in full Hilbert space
        """
        if qm not in ["SpinHalf", "SpinOne"]:
            raise ValueError("Must be SpinHalf or SpinOne")

        dof = Dofs(qm)
        splux = dof.Sp
        hilbsize = dof.hilbsize

        ida = sp.eye(hilbsize ** site)
        idb = sp.eye(hilbsize ** (self.Lat.Nsite - site - 1))
        Sp = sp.kron(ida, sp.kron(splux, idb))
        return Sp

    def LSmBuild(self, site, qm="SpinHalf"):
        """
        Build local spin minus operator in full Hilbert space
        """
        if qm not in ["SpinHalf", "SpinOne"]:
            raise ValueError("Must be SpinHalf or SpinOne")

        dof = Dofs(qm)
        sminus = dof.Sm
        hilbsize = dof.hilbsize

        ida = sp.eye(hilbsize ** site)
        idb = sp.eye(hilbsize ** (self.Lat.Nsite - site - 1))
        Sm = sp.kron(ida, sp.kron(sminus, idb))
        return Sm

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
    def mLocalSx(self, state, site, qm="SpinHalf"):
        """
        Measure local <Sx>
        """
        Sx = self.LSxBuild(site, qm)
        Sxeval = matele(state, Sx, state)
        return Sxeval

    def mLocalSy(self, state, site, qm="SpinHalf"):
        """
        Measure local <Sy>
        """
        Sy = self.LSyBuild(site, qm)
        Syeval = matele(state, Sy, state)
        return Syeval

    def mLocalSz(self, state, site, qm="SpinHalf"):
        """
        Measure local <Sz>
        """
        Sz = self.LSzBuild(site, qm)
        Szeval = matele(state, Sz, state)
        return Szeval

    def mLocalSp(self, state, site, qm="SpinHalf"):
        """
        Measure local <Sz>
        """
        Sp = self.LSpBuild(site, qm)
        Speval = matele(state, Sp, state)
        return Speval

    def mLocalSm(self, state, site, qm="SpinHalf"):
        """
        Measure local <Sz>
        """
        Sm = self.LSmBuild(site, qm)
        Smeval = matele(state, Sm, state)
        return Smeval

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
    def mLocalJs(self, state, site, qm="SpinHalf"):
        """
        Measure together all local spin current <Jsx>, <Jsy>, <Jsz>
        """
        Jsx, Jsy, Jsz = self.LscurrBuild(site, qm)
        Jsxeval = matele(state, Jsx, state)
        Jsyeval = matele(state, Jsx, state)
        Jszeval = matele(state, Jsx, state)
        return Jsxeval, Jsyeval, Jszeval

    def mLocalJe(self, state, site, qm="SpinHalf"):
        """
        Measure together all local energy current <Jex>, <Jey>, <Jez>
        """
        pass

    # ------------ End: local current measurement -----------------

    # ----------------------------- Transport measurement ----------------------------------
    # ----------------------------- Transport measurement ----------------------------------
    # ----------------------------- Transport measurement ----------------------------------
    def SpRe(self, evals, evecs, qm="SpinHalf"):
        """
        Measure together spin response S(omega)
        Parameter: evals, 1d array of eigen energies
                   evecs, 2d array, with cols being eigen vectors corresponding to evals
        """
        Lat = self.Lat
        Nstates = len(evals)

        gs = evecs[:, 0]  # ground state
        Eg = evals[0]  # ground state energy

        omegasteps = 400
        domega = 0.005
        eta = 0.009

        # for each omega, define a matrix Mel_{si,mi}
        Melx = np.zeros((Lat.Nsite, Nstates), dtype=complex)  # <m|S_i^x|gs>
        Mely = np.zeros((Lat.Nsite, Nstates), dtype=complex)  # <m|S_i^y|gs>
        Melz = np.zeros((Lat.Nsite, Nstates), dtype=complex)  # <m|S_i^z|gs>

        for mi in range(0, Nstates):
            for si in range(0, Lat.Nsite):
                Sxi = self.LSxBuild(si, qm)
                Syi = self.LSyBuild(si, qm)
                Szi = self.LSzBuild(si, qm)

                # <m|S_i^a|gs>
                Melx[si, mi] = matele(evecs[:, mi], Sxi, gs)
                Mely[si, mi] = matele(evecs[:, mi], Syi, gs)
                Melz[si, mi] = matele(evecs[:, mi], Szi, gs)

        # elastic contribution <Sa>_i
        mSx = np.zeros(Lat.Nsite, dtype=complex)
        mSy = np.zeros(Lat.Nsite, dtype=complex)
        mSz = np.zeros(Lat.Nsite, dtype=complex)
        for si in range(0, Lat.Nsite):
            mSx[si] = self.mLocalSx(gs, si, qm)
            mSy[si] = self.mLocalSy(gs, si, qm)
            mSz[si] = self.mLocalSz(gs, si, qm)

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
        # print(Itensity)

        return Sr

    def Econd(self, state, site):
        """
        Measure together energy conductivity
        """
        pass

    def SingleMagnon(self, evals, evecs, qm="SpinHalf"):
        """
        Measure Single Magnon DOS
        """
        Lat = self.Lat
        Nstates = len(evals)

        gs = evecs[:, 0]  # ground state
        Eg = evals[0]  # ground state energy

        omegasteps = 400
        domega = 0.005
        eta = 0.009

        # for each omega, define a matrix Mel_{si,mi}
        Melp = np.zeros((Lat.Nsite, Nstates), dtype=complex)  # <m|S_i^+|gs>
        Melm = np.zeros((Lat.Nsite, Nstates), dtype=complex)  # <m|S_i^-|gs>
        Melz = np.zeros((Lat.Nsite, Nstates), dtype=complex)  # <m|S_i^z|gs>

        for mi in range(0, Nstates):
            for si in range(0, Lat.Nsite):
                Spi = self.LSpBuild(si, qm)
                Smi = self.LSmBuild(si, qm)
                Szi = self.LSzBuild(si, qm)

                # <m|S_i^a|gs>
                Melp[si, mi] = matele(evecs[:, mi], Spi, gs)
                Melm[si, mi] = matele(evecs[:, mi], Smi, gs)
                Melz[si, mi] = matele(evecs[:, mi], Szi, gs)

        SM = np.zeros((omegasteps, 2), dtype=float)  # spin response
        # begin fill in vector Sr(\omega)
        omegacounter = 0
        for oi in range(0, omegasteps):
            omega = domega * oi
            Itensity = np.zeros((3, 3), dtype=complex)

            # for each omega, define a matrix Mel_{si,mi}
            for si in range(0, Lat.Nsite):
                for mi in range(1, Nstates):
                    Em = evals[mi]
                    denom2 = complex(omega - (Em - Eg), -eta)

                    # <m|S_i^a|gs>
                    tmpp = Melp[si, mi]
                    tmpm = Melm[si, mi]
                    tmpz = Melz[si, mi]

                    # <gs|S_i^a|m><m|S_i^b|gs>
                    tmppm = tmpp.conjugate() * tmpm  # Sip Sim
                    tmpmp = tmpm.conjugate() * tmpp  # Sim Sip
                    tmpzz = tmpz.conjugate() * tmpz  # Siz Siz

                    # update polarization matrix
                    Itensity[0, 1] += tmppm / denom2
                    Itensity[1, 0] += tmpmp / denom2
                    Itensity[2, 2] += tmpzz / denom2

            SM[omegacounter, 0] = round(omega, 4)
            SM[omegacounter, 1] = Itensity.sum().imag / (np.pi * Lat.Nsite)
            omegacounter += 1
        # print(Itensity)

        return SM

    def SzSz(self, evals, evecs, qm="SpinHalf"):
        """
        Measure Single < Sz Sz >
        """
        Lat = self.Lat
        Nstates = len(evals)

        gs = evecs[:, 0]  # ground state
        Eg = evals[0]  # ground state energy

        omegasteps = 400
        domega = 0.005
        eta = 0.009

        # for each omega, define a matrix Mel_{si,mi}
        #Melp = np.zeros((Lat.Nsite, Nstates), dtype=complex)  # <m|S_i^+|gs>
        #Melm = np.zeros((Lat.Nsite, Nstates), dtype=complex)  # <m|S_i^-|gs>
        Melz = np.zeros((Lat.Nsite, Nstates), dtype=complex)  # <m|S_i^z|gs>

        for mi in range(0, Nstates):
            for si in range(0, Lat.Nsite):
                Spi = self.LSpBuild(si, qm)
                Smi = self.LSmBuild(si, qm)
                Szi = self.LSzBuild(si, qm)

                # <m|S_i^a|gs>
                #Melp[si, mi] = matele(evecs[:, mi], Spi, gs)
                #Melm[si, mi] = matele(evecs[:, mi], Smi, gs)
                Melz[si, mi] = matele(evecs[:, mi], Szi, gs)

        SzSz = np.zeros((omegasteps, 2), dtype=float)  # spin response
        # begin fill in vector Sr(\omega)
        omegacounter = 0
        for oi in range(0, omegasteps):
            omega = domega * oi
            Itensity = np.zeros((3, 3), dtype=complex)

            # for each omega, define a matrix Mel_{si,mi}
            for si in range(0, Lat.Nsite):
                for mi in range(1, Nstates):
                    Em = evals[mi]
                    denom2 = complex(omega - (Em - Eg), -eta)

                    # <m|S_i^a|gs>
                    #tmpp = Melp[si, mi]
                    #tmpm = Melm[si, mi]
                    tmpz = Melz[si, mi]

                    # <gs|S_i^a|m><m|S_i^b|gs>
                    #tmppm = tmpp.conjugate() * tmpm  # Sip Sim
                    #tmpmp = tmpm.conjugate() * tmpp  # Sim Sip
                    tmpzz = tmpz.conjugate() * tmpz  # Siz Siz

                    # update polarization matrix
                    #Itensity[0, 1] += tmppm / denom2
                    #Itensity[1, 0] += tmpmp / denom2
                    Itensity[2, 2] += tmpzz / denom2

            SzSz[omegacounter, 0] = round(omega, 4)
            SzSz[omegacounter, 1] = Itensity.sum().imag / (np.pi * Lat.Nsite)
            omegacounter += 1
        # print(Itensity)

        return SzSz

    def SpSm(self, evals, evecs, qm="SpinHalf"):
        """
        Measure Single < Sz Sz >
        """
        Lat = self.Lat
        Nstates = len(evals)

        gs = evecs[:, 0]  # ground state
        Eg = evals[0]  # ground state energy

        omegasteps = 400
        domega = 0.005
        eta = 0.009

        # for each omega, define a matrix Mel_{si,mi}
        Melp = np.zeros((Lat.Nsite, Nstates), dtype=complex)  # <m|S_i^+|gs>
        Melm = np.zeros((Lat.Nsite, Nstates), dtype=complex)  # <m|S_i^-|gs>
        # Melz = np.zeros((Lat.Nsite, Nstates), dtype=complex)  # <m|S_i^z|gs>

        for mi in range(0, Nstates):
            for si in range(0, Lat.Nsite):
                Spi = self.LSpBuild(si, qm)
                Smi = self.LSmBuild(si, qm)
                Szi = self.LSzBuild(si, qm)

                # <m|S_i^a|gs>
                Melp[si, mi] = matele(evecs[:, mi], Spi, gs)
                Melm[si, mi] = matele(evecs[:, mi], Smi, gs)
                # Melz[si, mi] = matele(evecs[:, mi], Szi, gs)

        SpSm = np.zeros((omegasteps, 2), dtype=float)  # spin response
        # begin fill in vector Sr(\omega)
        omegacounter = 0
        for oi in range(0, omegasteps):
            omega = domega * oi
            Itensity = np.zeros((3, 3), dtype=complex)

            # for each omega, define a matrix Mel_{si,mi}
            for si in range(0, Lat.Nsite):
                for mi in range(1, Nstates):
                    Em = evals[mi]
                    denom2 = complex(omega - (Em - Eg), -eta)

                    # <m|S_i^a|gs>
                    tmpp = Melp[si, mi]
                    tmpm = Melm[si, mi]
                    #tmpz = Melz[si, mi]

                    # <gs|S_i^a|m><m|S_i^b|gs>
                    tmppm = tmpp.conjugate() * tmpm  # Sip Sim
                    #tmpmp = tmpm.conjugate() * tmpp  # Sim Sip
                    #tmpzz = tmpz.conjugate() * tmpz  # Siz Siz

                    # update polarization matrix
                    Itensity[0, 1] += tmppm / denom2
                    #Itensity[1, 0] += tmpmp / denom2
                    #Itensity[2, 2] += tmpzz / denom2

            SpSm[omegacounter, 0] = round(omega, 4)
            SpSm[omegacounter, 1] = Itensity.sum().imag / (np.pi * Lat.Nsite)
            omegacounter += 1
        # print(Itensity)

        return SpSm

    def ecurrent1(self, evals, evecs, site):
        """
        Measure energy current at a given site in TFIM
        """

        Lat = self.Lat
        si = site; qm ="SpinHalf"

        if Lat.Model != "TFIM":
            raise ValueError("ecurrent1 is designed for TFIM exclusively")
        elif si+1 > Lat.Nsite or si-1 < 0:
            raise ValueError("I'm falling out of boundary!")

        gs = evecs[:, 0]  # ground state
        Eg = evals[0]  # ground state energy
        # Build local spin operator in full Hilbert space
        Syi = self.LSyBuild(si, qm)
        Szi = self.LSzBuild(si, qm)  # Sz(i)
        Szim1 = self.LSzBuild(si - 1, qm)  #Sz(i-1)
        Szip1 = self.LSzBuild(si + 1, qm)  #Sz(i+1)

        # Build current operator    -Jh z(i-1)y(i)                  Not J[  z(i-1)y(i) + y(i)z(i+1)  ]  - h y(i)z(i+1)
        J = Lat.Kzz1; H = Lat.Hx
        # ECurr = J * H * (Szim1 * Syi + Syi * Szip1) - H * Syi * Szip1
        ECurr = - J * H * (Szim1 * Syi)

        # Measure current at ground state
        mECurr = matele(gs, ECurr, gs)
        return mECurr

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
