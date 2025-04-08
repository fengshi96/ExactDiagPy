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
from matplotlib import pyplot as plt

pi = np.pi


class KitaevLadderObserv(Observ):
    def __init__(self, Lat, Para):
        super().__init__(Lat, Para)
        self.Hx = Para.parameters["Bxx"]
        self.Hy = Para.parameters["Byy"]
        self.Hz = Para.parameters["Bzz"]

    
    def generate_ecurr_string(self):
        """
        Generate a string representation of ecurr_str.
        
        Each summand (a sublist) is represented as:
        constant*op1[site1]*op2[site2]*...*|gs>
        
        For example:
        [(2, 14, 'sx'), (2, 0, 'sz'), (2, 2, 'sy')]
        becomes:
        "2*sx[14]*sz[0]*sy[2]*|gs>"
        """
        print(self.Lat.nn_)
        print(self.Hx)
        self.ecurr_str = []  # [[(const, site, op), (const, site, op), (const, site, op)], ...]
        for i in np.arange(2, self.Lat.Nsite, 4):
            print("Building local energy current operator for site: " + str(i))
            site2 = i
            site3 = self.Lat.nn_[i, 2]
            site4 = self.Lat.nn_[i, 0]
            site5 = self.Lat.nn_[site4, 2]
            site6 = self.Lat.nn_[site4, 0]
            site7 = self.Lat.nn_[site6, 2]
            print(site2, site3, site4, site5, site6, site7)

            # 6 three-spin terms
            self.ecurr_str.append([(2, site2, 'sx'), (2, site4, 'sz'), (2, site6, 'sy')])
            self.ecurr_str.append([(-2, site3, 'sy'), (-2, site5, 'sz'), (-2, site7, 'sx')])
            self.ecurr_str.append([(1, site4, 'sz'), (1, site5, 'sy'), (1, site7, 'sx')])
            self.ecurr_str.append([(1, site3, 'sy'), (1, site4, 'sz'), (1, site5, 'sx')])
            self.ecurr_str.append([(-1, site2, 'sx'), (-1, site4, 'sy'), (-1, site5, 'sz')])
            self.ecurr_str.append([(-1, site4, 'sx'), (-1, site5, 'sz'), (-1, site6, 'sy')])

            # 8 two-spin terms
            h = self.Hx # for [111] field
            if h != 0:
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

        terms = []
        for summand in self.ecurr_str:
            # Get the constant from the first tuple in the summand.
            constant = summand[0][0]
            # Build operator factors: each tuple gives op[site]
            op_factors = []
            for tup in summand:
                # tup is (const, site, op) ; ignore the repeated constant in subsequent tuples.
                # Convert the operator to a string (if not already) and build e.g. "sx[14]"
                op_str = f"{tup[2]}[{tup[1]}]"
                op_factors.append(op_str)
            # Build the term: constant then multiplied by all operator factors, then "|gs>" at the end.
            # You might want to adjust whether you include a "*" between constant and the first op.
            term = f"{constant}*" + "*".join(op_factors) + "*|gs>"
            terms.append(term)
        # Join each term with a plus sign.
        result = "+".join(terms)
        print(result)
        return result


    def buildEnergyCurrent(self):
        print(self.Lat.nn_)
        print(self.Hx)
        self.ecurr_str = []  # [[(const, site, op), (const, site, op), (const, site, op)], ...]
        for i in np.arange(2, self.Lat.Nsite, 4):
            print("Building local energy current operator for site: " + str(i))
            site2 = i
            site3 = self.Lat.nn_[i, 2]
            site4 = self.Lat.nn_[i, 0]
            site5 = self.Lat.nn_[site4, 2]
            site6 = self.Lat.nn_[site4, 0]
            site7 = self.Lat.nn_[site6, 2]
            print(site2, site3, site4, site5, site6, site7)

            # 6 three-spin terms
            self.ecurr_str.append([(2, site2, 'sx'), (2, site4, 'sz'), (2, site6, 'sy')])
            self.ecurr_str.append([(-2, site3, 'sy'), (-2, site5, 'sz'), (-2, site7, 'sx')])
            self.ecurr_str.append([(1, site4, 'sz'), (1, site5, 'sy'), (1, site7, 'sx')])
            self.ecurr_str.append([(1, site3, 'sy'), (1, site4, 'sz'), (1, site5, 'sx')])
            self.ecurr_str.append([(-1, site2, 'sx'), (-1, site4, 'sy'), (-1, site5, 'sz')])
            self.ecurr_str.append([(-1, site4, 'sx'), (-1, site5, 'sz'), (-1, site6, 'sy')])

            # 8 two-spin terms
            h = self.Hx # for [111] field
            if h != 0:
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
        print("building the total energy current operator for q = 0")
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
                # print(first_site, second_site, third_site)
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
                        sp.kron(ida, dict[op1]), idm1), dict[op2]), idm2), dict[op3]), idb) * const
                
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
                Ecurr += sp.kron(sp.kron(sp.kron(sp.kron(ida, dict[op1]), idm), dict[op2]), idb) * const

        return Ecurr

    def CurrentCorrelation(self, evals, evecs, omegaList, eta, qm="SpinHalf"):
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
        ECC = np.zeros((len(omegaList), 2), dtype=float)


        Nstates = len(evals)
        gs = evecs[:, 0]  # ground state
        Eg = evals[0]  # ground state energy

        ECurr = self.buildEnergyCurrent()

        omegacounter = 0
        for oi in range(0, len(omegaList)):
            omega = omegaList[oi]
            ECC[omegacounter, 0] = omega

            for mi in range(0, Nstates):
                Em = evals[mi]
                if Em != Eg:  # rule out gs
                    denom = 1 / (omega - (Em - Eg) - complex(0, 1) * eta)
                    MelE = matele(evecs[:, mi], ECurr, gs)
                    weight = MelE.conjugate() * MelE * denom  
                    weight = weight.imag
                    ECC[omegacounter, 1] += weight

            omegacounter += 1

        return ECC


    def CurrentCorrelationFiniteTemp(self, evals, evecs, omegaList, eta, T):

        ECC = np.zeros((len(omegaList), 2), dtype=float)

        Nstates = len(evals)
        gs = evecs[:, 0]  # ground state
        Eg = evals[0]  # ground state energy

        ECurr = self.buildEnergyCurrent()

        Z = np.sum(np.exp(-evals / T))  # partition function
        print("Z = ", Z)

        omegacounter = 0
        for oi in range(0, len(omegaList)):
            omega = omegaList[oi]
            ECC[omegacounter, 0] = omega
            print("omega = ", omega)

            for ni in range(0, Nstates):
                En = evals[ni]
                prob = np.exp(- En / T) / Z
                for mi in range(Nstates):
                    Em = evals[mi]
                    # print("prob = ", prob)
                    if True: #Em != Eg and En != Eg:  # rule out gs
                        denom = eta / ((omega - (Em - En)) ** 2 + eta ** 2)
                        Melmn = matele(evecs[:, mi], ECurr, evecs[:, ni])
                        Melnm = matele(evecs[:, ni], ECurr, evecs[:, mi])
                        # print(Melmn,Melmn.conjugate(), Melnm, Melnm.conjugate())
                        weight = Melmn * Melnm * prob * denom  
                        weight = weight.real
                        ECC[omegacounter, 1] += weight 

            omegacounter += 1
            print("ECC = ", ECC[omegacounter, 1])
        ECC[:, 1] /= T
        

        return ECC     
  


    def SpinSpectral(self, evals, evecs, omegaList, eta, qm="SpinHalf"):
            """
            Measure together spin response S(omega)
            Parameter: evals, 1d array of eigen energies
                    evecs, 2d array, with cols being eigen vectors corresponding to evals
            """
            Nstates = len(evals)

            gs = evecs[:, 0]  # ground state
            Eg = evals[0]  # ground state energy

            hilbsize = Dofs(qm).hilbsize
            # STotal = sp.eye(hilbsize ** self.Lat.Nsite, dtype=complex) * 0
            # for i in range(0, self.Lat.Nsite):
            #     STotal += (-1) ** (i//2) * self.LSyBuild(i, qm)
            #     STotal += (-1) ** (i//2) * self.LSzBuild(i, qm)
            #     STotal += (-1) ** (i//2) * self.LSxBuild(i, qm)
            # STotal = self.LSxBuild(self.Lat.Nsite//2, qm) + self.LSyBuild(self.Lat.Nsite//2, qm) + self.LSzBuild(self.Lat.Nsite//2, qm)
            STotal = self.LSxBuild(self.Lat.Nsite//2, qm) + 1j * self.LSyBuild(self.Lat.Nsite//2, qm) 
            Sr = np.zeros((len(omegaList), 2), dtype=float)  # spin response
            omegacounter = 0
            for oi in range(0, len(omegaList)):
                omega = omegaList[oi]
                Sr[oi, 0] = omega
                for mi in range(0, Nstates):
                    Em = evals[mi]
                    if Em != Eg:
                        broaden = (1 / complex(omega - (Em - Eg), -eta)).imag
                        mel = matele(evecs[:, mi], STotal, gs)
                        MelS = mel.conjugate() * mel * broaden / np.pi
                        Sr[omegacounter, 1] += MelS.real
                omegacounter += 1
            return Sr

def observe(total, cmdargs):
    inputname = "input.inp" 
    observname = "CurrentCorrelation"
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

    # ob.buildEnergyCurrent()
    # ob.generate_ecurr_string()

    # ------- Calculate dynamical correlation --------
    if observname == "CurrentCorrelation":   
        eta = 0.025
        # omegaList = evals[:100] - evals[0]
        omegaList = np.arange(0.00, 0.3, 0.01)
        ECC = ob.CurrentCorrelation(evals, evecs, omegaList, eta)
        SCC = ob.SpinSpectral(evals, evecs, omegaList, eta)
        # ECC = ob.CurrentCorrelationFiniteTemp(evals, evecs, omegaList, eta, 100)
        # print(ECC)
        matprintos(ECC, "ECC.txt")
        matprintos(ECC, "SCC.txt")



        # ------- draw figure for ECC --------
        figure = plt.figure()
        ax = figure.add_subplot(111)
        # ax.plot(SCC[:, 0], SCC[:, 1])
        ax.plot(ECC[:, 0], ECC[:, 1])
        ax.set_xlabel("$\omega$")
        ax.set_ylabel("$C(\omega)$")
        figure.savefig("ECC.pdf", dpi=300, bbox_inches='tight')

        



if __name__ == '__main__':
    sys.argv  ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    observe(total, cmdargs)