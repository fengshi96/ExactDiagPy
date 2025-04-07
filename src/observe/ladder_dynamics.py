import sys
sys.path.append('/Users/shifeng/Projects/ExactDiagPy')
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
from src.Helper import matprint
from src.models.Kitaev_Ladder import Kitaev_Ladder

pi = np.pi


class KitaevLadderObserv(Observ):
    def __init__(self, Lat, Para):
        super().__init__(Lat, Para)
        self.Hx = Para.parameters["Bxx"]
        self.Hy = Para.parameters["Byy"]
        self.Hz = Para.parameters["Bzz"]

    
    def buildEnergyCurrent(self):
        print(self.Lat.nn_)
        print(self.Hx)
        self.ecurr_str = []  # [[(const, site, op), (const, site, op), (const, site, op)], ...]
        for i in np.arange(2, self.Lat.Nsite, 4):
            print("Building energy current for site: " + str(i))
            site2 = i
            site3 = self.Lat.nn_[i, 2]
            site4 = self.Lat.nn_[i, 0]
            site5 = self.Lat.nn_[site4, 2]
            site6 = self.Lat.nn_[site4, 0]
            site7 = self.Lat.nn_[site6, 2]
            print(site2, site3, site4, site5, site6, site7)

            # 6 three-spin terms
            for j in range(0, 6):
                self.ecurr_str.append([(2, site2, 'sx'), (2, site4, 'sz'), (2, site6, 'sy')])
                self.ecurr_str.append([(-2, site3, 'sy'), (-2, site5, 'sz'), (-2, site7, 'sx')])
                self.ecurr_str.append([(1, site4, 'sz'), (1, site5, 'sy'), (1, site7, 'sx')])
                self.ecurr_str.append([(1, site3, 'sy'), (1, site4, 'sz'), (1, site5, 'sx')])
                self.ecurr_str.append([(-1, site2, 'sx'), (-1, site4, 'sy'), (-1, site5, 'sz')])
                self.ecurr_str.append([(-1, site4, 'sx'), (-1, site5, 'sz'), (-1, site6, 'sy')])

            # 8 two-spin terms
            h = self.Hx # for [111] field
            if h != 0:
                for j in range(0, 8):
                    self.ecurr_str.append([(h, site3, 'sy'), (h, site5, 'sx')])
                    self.ecurr_str.append([(-h, site2, 'sx'), (-h, site4, 'sy')])
                    self.ecurr_str.append([(h, site5, 'sy'), (h, site7, 'sx')])
                    self.ecurr_str.append([(-h, site4, 'sx'), (-h, site6, 'sy')])
                    self.ecurr_str.append([(h, site2, 'sx'), (h, site4, 'sz')])
                    self.ecurr_str.append([(h, site4, 'sz'), (h, site6, 'sy')])
                    self.ecurr_str.append([(-h, site5, 'sz'), (-h, site7, 'sx')])
                    self.ecurr_str.append([(-h, site3, 'sy'), (-h, site5, 'sz')])

            if self.Hx != 0:
                items_to_print = self.ecurr_str[-14:]
            else:
                items_to_print = self.ecurr_str[-6:]

            for sublist in items_to_print:
                # Create a formatted string for each tuple in the sublist
                formatted_line = ", ".join(f"({const}, {site}, {op})" for const, site, op in sublist)
                print(f"[{formatted_line}]")
            
            print('finished building energy current for site: ' + str(i) + '\n')

        # construct the sparse matrix for the total energy current operator
        Spins = Dofs("SpinHalf")
        X = Spins.Sx * 2
        Y = Spins.Sy * 2
        Z = Spins.Sz * 2
        hilbsize = Spins.hilbsize
        dict = {'sx': X, 'sy': Y, 'sz': Z}

        self.Oscurr_str = []  # on-site spin current
        Ecurr = sp.eye(hilbsize ** self.Lat.Nsite, dtype=complex) * 0

        for sublist in self.ecurr_str:
            sorted_row = sorted(sublist, key=lambda x: x[1])
            # print(sorted_row)
            
            # for three spin terms
            if len(sorted_row) == 3:
                first_site = sorted_row[0][1]
                second_site = sorted_row[1][1]
                third_site = sorted_row[2][1]
                print(first_site, second_site, third_site)
                if first_site > 0:
                    ida = sp.eye(hilbsize ** first_site)
                else:
                    ida = sp.eye(1)
                idm1 = sp.eye(hilbsize ** (second_site - first_site - 1))
                idm2 = sp.eye(hilbsize ** (third_site - second_site - 1))
                if third_site < self.Lat.Nsite - 1:
                    idb = sp.eye(hilbsize ** (self.Lat.Nsite - third_site - 1))
                else:
                    idb = sp.eye(1)

                op1 = sorted_row[0][2]
                op2 = sorted_row[1][2]
                op3 = sorted_row[2][2]
                const = sorted_row[0][0]
                Ecurr += sp.kron(sp.kron(sp.kron(sp.kron(sp.kron(
                        sp.kron(ida, dict[op1]), idm1), dict[op2]), idm2), dict[op3]), idb) * const * 1j
                
            # for two spin terms
            elif len(sorted_row) == 2:
                first_site = sorted_row[0][1]
                second_site = sorted_row[1][1]
                if first_site > 0:
                    ida = sp.eye(hilbsize ** first_site)
                else:
                    ida = sp.eye(1)
                idm = sp.eye(hilbsize ** (second_site - first_site - 1))
                if second_site < self.Lat.Nsite - 1:
                    idb = sp.eye(hilbsize ** (self.Lat.Nsite - second_site - 1))
                else:
                    idb = sp.eye(1)

                op1 = sorted_row[0][2]
                op2 = sorted_row[1][2]
                const = sorted_row[0][0]
                Ecurr += sp.kron(sp.kron(sp.kron(sp.kron(ida, dict[op1]), idm), dict[op2]), idb) * const * 1j


    def DynDoubleCorrelation(self, siteRef, bond, evals, evecs, omegaList, eta, qm="SpinHalf"):
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
        Lat = self.Lat
        print("Calculating DynDoubleCorrelation for the dimer: " + str(siteRef) + " and " + str(Lat.nn_[siteRef, bond]))
        CSxx = np.zeros((len(omegaList), int(Lat.Nsite/2)+1), dtype=float)
        CSyy = np.zeros((len(omegaList), int(Lat.Nsite/2)+1), dtype=float)
        CSzz = np.zeros((len(omegaList), int(Lat.Nsite/2)+1), dtype=float)

        Nstates = len(evals)
        gs = evecs[:, 0]  # ground state
        Eg = evals[0]  # ground state energy

        SxxR = self.LSxBuild(siteRef) * self.LSxBuild(Lat.nn_[siteRef, bond]) # dofs on the site of reference
        SyyR = self.LSyBuild(siteRef) * self.LSyBuild(Lat.nn_[siteRef, bond])
        SzzR = self.LSzBuild(siteRef) * self.LSzBuild(Lat.nn_[siteRef, bond])

        omegacounter = 0
        for oi in range(0, len(omegaList)):
            omega = omegaList[oi]
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



def observe(total, cmdargs):
    inputname = "input.inp" 
    observname = "TBW"
    para = Parameter(inputname)
    Lat = Lattice(para)
    ob = KitaevLadderObserv(Lat, para)
    print(para.parameters["Model"])
    dof = Dofs("SpinHalf")  # default spin-1/2

    # ------- Read dataSpec file -------
    rfile = h5py.File('dataSpec.hdf5', 'r')
    evalset = rfile["3.Eigen"]["Eigen Values"]
    evecset = rfile["3.Eigen"]["Wavefunctions"]

    # extract eigen value and eigen vector from HDF5 rfile
    print(dof.hilbsize ** Lat.Nsite)
    evals = np.zeros(para.parameters["Nstates"], dtype=float)
    evecs = np.zeros((dof.hilbsize ** Lat.Nsite, para.parameters["Nstates"]), dtype=complex)
    evalset.read_direct(evals)
    evecset.read_direct(evecs)
    rfile.close()

    ob.buildEnergyCurrent()

    # ------- Calculate dynamical correlation --------
    if observname == "dynCorrelation":   
        eta = 0.00010
        omegaList = np.array([0.00798])
        CSx, CSy, CSz = ob.DynCorrelation(11, evals, evecs, omegaList, eta)
        matprintos(CSx, "Csx.dat")
        matprintos(CSy, "Csy.dat")
        matprintos(CSz, "Csz.dat")

        



if __name__ == '__main__':
    sys.argv  ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    observe(total, cmdargs)