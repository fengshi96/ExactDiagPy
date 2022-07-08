import sys
import time
import numpy as np
import h5py
import scipy.sparse as sp
from src.Wavefunction import *
from src.Helper import matprintos
from src.Dofs import Dofs
from src.Parameter import Parameter
from src.Observ import matele, Observ
from src.Lattice import Lattice
from src.Helper import matprintos, printfArray

pi = np.pi


class KitaevObserv(Observ):
    def __init__(self, Lat, Para):
        super().__init__(Lat, Para)

    def Single(self, site, evals, evecs, omegasteps, domega, eta, qm="SpinHalf"):
        """
        Single site density of states i.e. S(omega) = sum_m |<0|O_i|m>|^2 delta(omega - (Em - E0))
        :param site: the site i of O_i
        :param evals: 1D array of eigen values
        :param evecs: 2D array of eigen vectors; m-th column corresponds to m-th eigenvalue
        :param omegasteps: number of omegas to scan
        :param domega: step of omega scan
        :param eta: broadening factor
        :param qm: type of dof
        :return: 3 Arrays of DOS for O_i = sigma_x, sigma_y, sigma_z respectively
        """
        Nstates = len(evals)

        gs = evecs[:, 0]  # ground state
        Eg = evals[0]  # ground state energy

        # for each omega, define a matrix Mel_{si,mi}
        Mel1 = np.zeros(Nstates, dtype=complex)  # <m|D_x(1)|gs>
        Mel2 = np.zeros(Nstates, dtype=complex)  # <m|D_y(2)|gs>
        Mel3 = np.zeros(Nstates, dtype=complex)  # <m|Gz|gs>

        Sx = self.LSxBuild(site)
        Sy = self.LSyBuild(site)
        Sz = self.LSzBuild(site)

        for mi in range(0, Nstates):
            Mel1[mi] = matele(evecs[:, mi], Sx, gs)
            Mel2[mi] = matele(evecs[:, mi], Sy, gs)
            Mel3[mi] = matele(evecs[:, mi], Sz, gs)

        CSx = np.zeros(omegasteps, dtype=float)  # spin response
        CSy = np.zeros(omegasteps, dtype=float)  # spin response
        CSz = np.zeros(omegasteps, dtype=float)  # spin response

        omegacounter = 0
        for oi in range(0, omegasteps):
            omega = domega * oi

            # for each omega, define a matrix Mel_{si,mi}
            for mi in range(0, Nstates):
                Em = evals[mi]
                if Em != Eg:  # rule out gs
                    denom = eta / ((omega - (Em - Eg)) ** 2 + eta ** 2)

                    # <m|O|gs>
                    tmp1 = Mel1[mi]
                    tmp2 = Mel2[mi]
                    tmp3 = Mel3[mi]

                    # <gs|O*|m><m|O|gs>
                    tmp11 = (tmp1.conjugate() * tmp1).real  # D1 D1
                    tmp22 = (tmp2.conjugate() * tmp2).real  # D2 D2
                    tmp33 = (tmp3.conjugate() * tmp3).real  # D3 D3

                    # update polarization matrix
                    CSx[omegacounter] += tmp11 * denom
                    CSy[omegacounter] += tmp22 * denom
                    CSz[omegacounter] += tmp33 * denom

            omegacounter += 1
        # print(Itensity)

        return CSx * 2 * self.Lat.Nsite, CSy * 2 * self.Lat.Nsite, CSz * 2 * self.Lat.Nsite

    def Double(self, site, evals, evecs, omegasteps, domega, eta, qm="SpinHalf"):
        """
        Double site density of states i.e. S(omega) = sum_m |<0|A_i B_j|m>|^2 delta(omega - (Em - E0))
        :param site: the site i of O_i
        :param evals: 1D array of eigen values
        :param evecs: 2D array of eigen vectors; m-th column corresponds to m-th eigenvalue
        :param omegasteps: number of omegas to scan
        :param domega: step of omega scan
        :param eta: broadening factor
        :param qm: type of dof
        :return: 3 Arrays of DOS for Majorana pair, Majorana+f bound and Majorana+m bound
        """
        Nstates = len(evals)

        gs = evecs[:, 0]  # ground state
        Eg = evals[0]  # ground state energy

        # for each omega, define a matrix Mel_{si,mi}
        Mel1 = np.zeros(Nstates, dtype=complex)  # <m|D_x(1)|gs>
        Mel2 = np.zeros(Nstates, dtype=complex)  # <m|D_y(2)|gs>
        Mel3 = np.zeros(Nstates, dtype=complex)  # <m|Gz|gs>

        nn_ = self.Lat.nn_[site, :]
        CC = self.LSxBuild(site) * self.LSxBuild(nn_[0])
        CCE = self.LSyBuild(site) * self.LSzBuild(nn_[2])
        CCM = self.LSxBuild(site) * self.LSyBuild(nn_[1])

        for mi in range(0, Nstates):
            Mel1[mi] = matele(evecs[:, mi], CC, gs)
            Mel2[mi] = matele(evecs[:, mi], CCE, gs)
            Mel3[mi] = matele(evecs[:, mi], CCM, gs)

        intCC = np.zeros(omegasteps, dtype=float)  # intensity of CC
        intCCE = np.zeros(omegasteps, dtype=float)
        intCCM = np.zeros(omegasteps, dtype=float)

        omegacounter = 0
        for oi in range(0, omegasteps):
            omega = domega * oi

            # for each omega, define a matrix Mel_{si,mi}
            for mi in range(0, Nstates):
                Em = evals[mi]
                if Em != Eg:  # rule out gs
                    denom = eta / ((omega - (Em - Eg)) ** 2 + eta ** 2)

                    # <m|O|gs>
                    tmp1 = Mel1[mi]
                    tmp2 = Mel2[mi]
                    tmp3 = Mel3[mi]

                    # <gs|O*|m><m|O|gs>
                    tmp11 = (tmp1.conjugate() * tmp1).real  # D1 D1
                    tmp22 = (tmp2.conjugate() * tmp2).real  # D2 D2
                    tmp33 = (tmp3.conjugate() * tmp3).real  # D3 D3

                    # update polarization matrix
                    intCC[omegacounter] += tmp11 * denom
                    intCCE[omegacounter] += tmp22 * denom
                    intCCM[omegacounter] += tmp33 * denom

            omegacounter += 1
        # print(Itensity)

        return intCC * 2 * self.Lat.Nsite, intCCE * 2 * self.Lat.Nsite, intCCM * 2 * self.Lat.Nsite

    def DynCorrelation(self, siteRef, evals, evecs, omegasteps, domega, eta, qm="SpinHalf"):
        """
        For dynamical correlation i.e. S(c,j, omega) = sum_m sum_j <0|O_R|m><m|O_j|0> delta(omega - (Em - E0))
        :param siteRef: the site of reference: R of A_R
        :param evals: 1D array of eigen values
        :param evecs: 2D array of eigen vectors; m-th column corresponds to m-th eigenvalue
        :param omegasteps: number of omegas to scan
        :param domega: step of omega scan
        :param eta: broadening factor
        :param qm: type of dof
        :return: 2D array: len(omega) x Nsites
        """
        print("Calculating DynCorrelator for site: " + str(siteRef))
        CSx = np.zeros((omegasteps, Lat.Nsite+1), dtype=float)
        CSy = np.zeros((omegasteps, Lat.Nsite+1), dtype=float)
        CSz = np.zeros((omegasteps, Lat.Nsite+1), dtype=float)

        Nstates = len(evals)
        gs = evecs[:, 0]  # ground state
        Eg = evals[0]  # ground state energy

        SxR = self.LSxBuild(siteRef) # dofs on the site of reference
        SyR = self.LSyBuild(siteRef)
        SzR = self.LSzBuild(siteRef)

        omegacounter = 0
        for oi in range(0, omegasteps):
            omega = domega * oi
            CSx[omegacounter, 0] = omega
            CSy[omegacounter, 0] = omega
            CSz[omegacounter, 0] = omega
            for si in range(Lat.Nsite):
                Sxi = self.LSxBuild(si)  # dofs on the site i
                Syi = self.LSyBuild(si)
                Szi = self.LSzBuild(si)

                for mi in range(0, Nstates):
                    Em = evals[mi]
                    if Em != Eg:  # rule out gs
                        MelRx = matele(evecs[:, mi], SxR, gs)
                        MelRy = matele(evecs[:, mi], SyR, gs)
                        MelRz = matele(evecs[:, mi], SzR, gs)
                        Melix = matele(evecs[:, mi], Sxi, gs)
                        Meliy = matele(evecs[:, mi], Syi, gs)
                        Meliz = matele(evecs[:, mi], Szi, gs)

                        denom = 1 / (omega - (Em - Eg) - complex(0, 1) * eta)

                        # <gs|O_R|m><m|O_i|gs>
                        tmp11 = MelRx.conjugate() * Melix * denom  # D1 D1
                        tmp22 = MelRy.conjugate() * Meliy * denom  # D2 D2
                        tmp33 = MelRz.conjugate() * Meliz * denom  # D3 D3

                        # update polarization matrix
                        CSx[omegacounter, si+1] += tmp11.imag
                        CSy[omegacounter, si+1] += tmp22.imag
                        CSz[omegacounter, si+1] += tmp33.imag

            omegacounter += 1

        return CSx, CSy, CSz

    def DynDoubleCorrelation(self, siteRef, bond, evals, evecs, omegasteps, domega, eta, qm="SpinHalf"):
        """
        For dynamical correlation i.e. S(c,j, omega) = sum_m sum_j <0|O_R O_{R+b}|m><m|O_j O_{j+b}|0> delta(omega - (Em - E0))
        :param siteRef: the site of reference: R of A_R
        :param evals: 1D array of eigen values
        :param bond: integer that indicate bond direction; 0 = x, 1 = y, 2 = z
        :param evecs: 2D array of eigen vectors; m-th column corresponds to m-th eigenvalue
        :param omegasteps: number of omegas to scan
        :param domega: step of omega scan
        :param eta: broadening factor
        :param qm: type of dof
        :return: 2D array: len(omega) x #unit_cells =  len(omega) x Nsites/2
        """
        print("Calculating DynDoubleCorrelation for the dimer: " + str(siteRef) + " and " + str(Lat.nn_[siteRef, bond]))
        CSxx = np.zeros((omegasteps, int(Lat.Nsite/2)+1), dtype=float)
        CSyy = np.zeros((omegasteps, int(Lat.Nsite/2)+1), dtype=float)
        CSzz = np.zeros((omegasteps, int(Lat.Nsite/2)+1), dtype=float)

        Nstates = len(evals)
        gs = evecs[:, 0]  # ground state
        Eg = evals[0]  # ground state energy

        SxxR = self.LSxBuild(siteRef) * self.LSxBuild(Lat.nn_[siteRef, bond]) # dofs on the site of reference
        SyyR = self.LSyBuild(siteRef) * self.LSyBuild(Lat.nn_[siteRef, bond])
        SzzR = self.LSzBuild(siteRef) * self.LSzBuild(Lat.nn_[siteRef, bond])

        omegacounter = 0
        for oi in range(0, omegasteps):
            omega = domega * oi
            CSxx[omegacounter, 0] = omega
            CSyy[omegacounter, 0] = omega
            CSzz[omegacounter, 0] = omega
            for si in range(0, Lat.Nsite, 2):
                Sxxi = self.LSxBuild(si) * self.LSxBuild(Lat.nn_[si, bond])  # dofs on the site i
                Syyi = self.LSyBuild(si) * self.LSyBuild(Lat.nn_[si, bond])
                Szzi = self.LSzBuild(si) * self.LSzBuild(Lat.nn_[si, bond])

                for mi in range(0, Nstates):
                    Em = evals[mi]
                    if Em != Eg:  # rule out gs
                        MelRx = matele(evecs[:, mi], SxxR, gs)
                        MelRy = matele(evecs[:, mi], SyyR, gs)
                        MelRz = matele(evecs[:, mi], SzzR, gs)
                        Melix = matele(evecs[:, mi], Sxxi, gs)
                        Meliy = matele(evecs[:, mi], Syyi, gs)
                        Meliz = matele(evecs[:, mi], Szzi, gs)

                        denom = 1 / (omega - (Em - Eg) - complex(0, 1) * eta)

                        # <gs|O_R|m><m|O_i|gs>
                        tmp11 = MelRx.conjugate() * Melix * denom  # D1 D1
                        tmp22 = MelRy.conjugate() * Meliy * denom  # D2 D2
                        tmp33 = MelRz.conjugate() * Meliz * denom  # D3 D3

                        # update polarization matrix
                        CSxx[omegacounter, int(si/2)+1] += tmp11.imag
                        CSyy[omegacounter, int(si/2)+1] += tmp22.imag
                        CSzz[omegacounter, int(si/2)+1] += tmp33.imag

            omegacounter += 1

        return CSxx, CSyy, CSzz


if __name__ == '__main__':
    inputname = "../../input.inp"
    para = Parameter(inputname)
    Lat = Lattice(para)
    ob = KitaevObserv(Lat, para)
    print(para.parameters["Model"])
    dof = Dofs("SpinHalf")  # default spin-1/2

    # ------- Read dataSpec file -------
    rfile = h5py.File('../../dataSpec.hdf5', 'r')
    evalset = rfile["3.Eigen"]["Eigen Values"]
    evecset = rfile["3.Eigen"]["Wavefunctions"]

    # extract eigen value and eigen vector from HDF5 rfile
    print(dof.hilbsize ** Lat.Nsite)
    evals = np.zeros(para.parameters["Nstates"], dtype=float)
    evecs = np.zeros((dof.hilbsize ** Lat.Nsite, para.parameters["Nstates"]), dtype=complex)
    evalset.read_direct(evals)
    evecset.read_direct(evecs)
    rfile.close()

    omegasteps = 500
    domega = 0.00004
    eta = 0.00010
    B = np.abs(np.ones(omegasteps) * para.parameters["Bxx"])
    Omega = np.round(np.linspace(0, (omegasteps - 1) * domega, omegasteps), 5)

    CSx, CSy, CSz = ob.DynCorrelation(11, evals, evecs, omegasteps, domega, eta)
    printfArray(CSx, "Csx.dat")
    printfArray(CSy, "Csy.dat")
    printfArray(CSz, "Csz.dat")

    # CSxx, CSyy, CSzz = ob.DynDoubleCorrelation(11, 12, evals, evecs, omegasteps, domega, eta)
    # printfArray(CSxx, "Csxx-zbond.dat")
    # printfArray(CSyy, "Csyy-zbond.dat")
    # printfArray(CSzz, "Cszz-zbond.dat")
    #
    # CSxx, CSyy, Czz = ob.DynDoubleCorrelation(11, 10, evals, evecs, omegasteps, domega, eta)
    # printfArray(CSxx, "Csxx-xbond.dat")
    # printfArray(CSyy, "Csyy-xbond.dat")
    # printfArray(CSzz, "Cszz-xbond.dat")