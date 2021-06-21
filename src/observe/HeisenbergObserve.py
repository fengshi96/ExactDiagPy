import sys, math
import numpy as np
import time
import primme
from src.Dofs import Dofs
import scipy.sparse as sp
from src.models.Heisenberg import Heisenberg
from src.Observ import Observ, matele
from src.Parameter import Parameter
from src.Lattice import Lattice
from src.Helper import Logger, sort, hd5Storage


class HeisenbergObserve(Observ):
    def __init__(self, Lat, Para):
        super().__init__(Lat, Para)
        self.Lat = Lat
        self.Para = Para

    def spinCurr(self, direction, site, qm="SpinHalf"):  # on-site spin current x operator
        if qm not in ["SpinHalf", "SpinOne"]:
            raise ValueError("Must be SpinHalf or SpinOne")

        Lat = self.Lat

        Spins = Dofs(qm)
        Sp = Spins.Sp
        Sm = Spins.Sm
        hilbsize = Spins.hilbsize

        self.Oscurr_str = []  # on-site spin current
        Oscurr = sp.eye(hilbsize ** Lat.Nsite, dtype=complex) * 0  # on-site spin current

        nn_ = Lat.nn_[site, :]  # nn_[0]: i+x; nn_[1]: i+y; nn_[2]: i+z
        # print(nn_)
        string = ""

        if direction == "x" or direction == "X":
            indxS = min(site, nn_[0])  # the smaller index
            indxL = max(site, nn_[0])  # the larger index
            if indxS >= 0:
                ida = sp.eye(hilbsize ** indxS)
                idm = sp.eye(hilbsize ** (indxL - indxS - 1))
                idb = sp.eye(hilbsize ** (Lat.Nsite - indxL - 1))
                Oscurr += 0.5 * complex(0, 1) * self.Para.Kxx * sp.kron(sp.kron(sp.kron(sp.kron(ida, Sp), idm), Sm),
                                                                        idb)
                Oscurr -= Oscurr.conjugate()
                string = "0.5i*Kx*(sp[" + str(site) + "]*" + "sm[" + str(nn_[0]) + "]" \
                         + "-sm[" + str(site) + "]*sp[" + str(nn_[0]) + "])"  # "js_x["+str(site)+"] = "+

                print(string)
            return Oscurr, string

        elif direction == "y" or direction == "Y":
            indxS = min(site, nn_[1])  # the smaller index
            indxL = max(site, nn_[1])  # the larger index
            if indxS >= 0:
                ida = sp.eye(hilbsize ** indxS)
                idm = sp.eye(hilbsize ** (indxL - indxS - 1))
                idb = sp.eye(hilbsize ** (Lat.Nsite - indxL - 1))
                Oscurr += 0.5 * complex(0, 1) * self.Para.Kxx * sp.kron(sp.kron(sp.kron(sp.kron(ida, Sp), idm), Sm),
                                                                        idb)
                Oscurr -= Oscurr.conjugate()
                string = "0.5i*Ky*(sp[" + str(site) + "]*" + "sm[" + str(nn_[1]) + "]" \
                         + "-sm[" + str(site) + "]*sp[" + str(nn_[1]) + "])"  # "js_x["+str(site)+"] = "+

                print(string)
            return Oscurr, string

        elif direction == "z" or direction == "Z":
            indxS = min(site, nn_[2])  # the smaller index
            indxL = max(site, nn_[2])  # the larger index
            if indxS >= 0:
                ida = sp.eye(hilbsize ** indxS)
                idm = sp.eye(hilbsize ** (indxL - indxS - 1))
                idb = sp.eye(hilbsize ** (Lat.Nsite - indxL - 1))
                Oscurr += 0.5 * complex(0, 1) * self.Para.Kxx * sp.kron(sp.kron(sp.kron(sp.kron(ida, Sp), idm), Sm),
                                                                        idb)
                Oscurr -= Oscurr.conjugate()
                string = "0.5i*Kz*(sp[" + str(site) + "]*" + "sm[" + str(nn_[2]) + "]" \
                         + "-sm[" + str(site) + "]*sp[" + str(nn_[2]) + "])"  # "js_x["+str(site)+"] = "+

                print(string)
            return Oscurr, string

        else:
            raise ValueError("direction not valid")

    def spinCurrTotal(self, direction, qm="SpinHalf"):  # on-site spin current
        print("[HeisenbergObserve.py] Building total spin current in ", direction, "...")
        if qm not in ["SpinHalf", "SpinOne"]:
            raise ValueError("Must be SpinHalf or SpinOne")
        if direction not in ["x", "y", "z", "X", "Y", "Z"]:
            raise ValueError("direction not valid")

        Lat = self.Lat
        hilbsize = Dofs(qm).hilbsize

        Tscurr_str = []  # total spin current in x,y,z
        scurrTotal = sp.eye(hilbsize ** Lat.Nsite, dtype=complex) * 0

        if direction == "x" or direction == "X":
            for site in range(0, Lat.Nsite, 2):
                tmp, string = self.spinCurr("x", site, qm)
                scurrTotal += tmp
                Tscurr_str.append(string)

            stringtot = "+".join(Tscurr_str)
            # print(stringtot)
            return scurrTotal, stringtot

        elif direction == "y" or direction == "Y":
            for site in range(0, Lat.Nsite, 2):
                tmp, string = self.spinCurr("y", site, qm)
                scurrTotal += tmp
                Tscurr_str.append(string)

            stringtot = "+".join(Tscurr_str)
            # print(stringtot)
            return scurrTotal, stringtot

        elif direction == "z" or direction == "Z":
            for site in range(0, Lat.Nsite, 2):
                tmp, string = self.spinCurr("z", site, qm)
                scurrTotal += tmp
                Tscurr_str.append(string)

            stringtot = "+".join(Tscurr_str)
            # print(stringtot)
            return scurrTotal, stringtot

    # kinetic += 1 / (2 * self.Para.Nsites) * \
    #            self.Para.Kxx * sp.kron(sp.kron(sp.kron(sp.kron(ida, Sp), idm), Sm), idb)
    # kinetic += kinetic

    def kinetic(self, direction, site, qm="SpinHalf"):
        if qm not in ["SpinHalf", "SpinOne"]:
            raise ValueError("Must be SpinHalf or SpinOne")

        Lat = self.Lat
        Spins = Dofs(qm)
        Sp = Spins.Sp
        Sm = Spins.Sm
        hilbsize = Spins.hilbsize

        self.kinetic_str = []  # on-site spin current
        kinetic = sp.eye(hilbsize ** Lat.Nsite, dtype=complex) * 0  # on-site spin current

        nn_ = Lat.nn_[site, :]  # nn_[0]: i+x; nn_[1]: i+y; nn_[2]: i+z
        # print(nn_)
        string = ""

        if direction == "x" or direction == "X":
            indxS = min(site, nn_[0])  # the smaller index
            indxL = max(site, nn_[0])  # the larger index
            if indxS >= 0:
                ida = sp.eye(hilbsize ** indxS)
                idm = sp.eye(hilbsize ** (indxL - indxS - 1))
                idb = sp.eye(hilbsize ** (Lat.Nsite - indxL - 1))
                kinetic += 1 / (2 * Lat.Nsite) * \
                           self.Para.Kxx * sp.kron(sp.kron(sp.kron(sp.kron(ida, Sp), idm), Sm), idb)
                kinetic += kinetic.conjugate()
                string = "1/(2*N)*Kx*(sp[" + str(site) + "]*" + "sm[" + str(nn_[0]) + "]" \
                         + "+sm[" + str(site) + "]*sp[" + str(nn_[0]) + "])"  # "js_x["+str(site)+"] = "+

                print(string)
            return kinetic, string

        elif direction == "y" or direction == "Y":
            indxS = min(site, nn_[1])  # the smaller index
            indxL = max(site, nn_[1])  # the larger index
            if indxS >= 0:
                ida = sp.eye(hilbsize ** indxS)
                idm = sp.eye(hilbsize ** (indxL - indxS - 1))
                idb = sp.eye(hilbsize ** (Lat.Nsite - indxL - 1))
                kinetic += 1 / (2 * Lat.Nsite) * \
                           self.Para.Kyy * sp.kron(sp.kron(sp.kron(sp.kron(ida, Sp), idm), Sm), idb)
                kinetic += kinetic.conjugate()
                string = "1/(2*N)*Ky*(sp[" + str(site) + "]*" + "sm[" + str(nn_[0]) + "]" \
                         + "+sm[" + str(site) + "]*sp[" + str(nn_[0]) + "])"  # "js_x["+str(site)+"] = "+

                print(string)
            return kinetic, string

        elif direction == "z" or direction == "Z":
            indxS = min(site, nn_[2])  # the smaller index
            indxL = max(site, nn_[2])  # the larger index
            if indxS >= 0:
                ida = sp.eye(hilbsize ** indxS)
                idm = sp.eye(hilbsize ** (indxL - indxS - 1))
                idb = sp.eye(hilbsize ** (Lat.Nsite - indxL - 1))
                kinetic += 1 / (2 * Lat.Nsite) * \
                           self.Para.Kzz * sp.kron(sp.kron(sp.kron(sp.kron(ida, Sp), idm), Sm), idb)
                kinetic += kinetic.conjugate()
                string = "1/(2*N)*Kz*(sp[" + str(site) + "]*" + "sm[" + str(nn_[0]) + "]" \
                         + "+sm[" + str(site) + "]*sp[" + str(nn_[0]) + "])"  # "js_x["+str(site)+"] = "+

                print(string)
            return kinetic, string

        else:
            raise ValueError("direction not valid")

    def kineticTotal(self, direction, qm="SpinHalf"):
        print("[HeisenbergObserve.py] Building total Kinetic term in ", direction, "...")
        if qm not in ["SpinHalf", "SpinOne"]:
            raise ValueError("Must be SpinHalf or SpinOne")
        if direction not in ["x", "y", "z", "X", "Y", "Z"]:
            raise ValueError("direction not valid")

        hilbsize = Dofs(qm).hilbsize
        kinetic_str = []  # on-site spin current
        kineticTot = sp.eye(hilbsize ** self.Lat.Nsite, dtype=complex) * 0  # on-site spin current

        if direction == "x" or direction == "X":
            for site in range(0, self.Lat.Nsite, 2):
                tmp, string = self.kinetic("x", site, qm)
                kineticTot += tmp
                kinetic_str.append(string)
            stringtot = "+".join(kinetic_str)
            # print(stringtot)
            return kineticTot, stringtot

        if direction == "y" or direction == "Y":
            for site in range(0, self.Lat.Nsite, 2):
                tmp, string = self.kinetic("y", site, qm)
                kineticTot += tmp
                kinetic_str.append(string)
            stringtot = "+".join(kinetic_str)
            # print(stringtot)
            return kineticTot, stringtot

        elif direction == "z" or direction == "Z":
            for site in range(0, self.Lat.Nsite, 2):
                tmp, string = self.kinetic("z", site, qm)
                kineticTot += tmp
                kinetic_str.append(string)
            stringtot = "+".join(kinetic_str)
            # print(stringtot)
            return kineticTot, stringtot

    def Lambda(self, evals, evecs, direction, omegasteps=500, domega=0.002, eta=0.01, qm="SpinHalf"):
        """
        Measure \Lambda_ii(\omega)
        Parameter: evals, 1d array of eigen energies
                   evecs, 2d array, with cols being eigen vectors corresponding to evals
                   direction, current direction, x,y or z
        """
        gs = evecs[:, 0]  # ground state
        Eg = evals[0]  # ground state energy
        Lambda = np.zeros(omegasteps, dtype=complex)

        J = sp.eye(0)
        if direction == "x" or direction == "X":
            J, _ = self.spinCurrTotal("x", qm)
        elif direction == "y" or direction == "Y":
            J, _ = self.spinCurrTotal("y", qm)
        elif direction == "z" or direction == "Z":
            J, _ = self.spinCurrTotal("z", qm)

        for oi in range(0, omegasteps):
            omega = domega * oi

            for m in range(0, self.Para.Nstates):
                J0m = matele(gs, J, evecs[:, m])
                Jm0 = np.conj(J0m)

                denominator = omega - (evals[m] - Eg) + complex(0, 1) * eta
                Lambda[oi] += J0m * Jm0 / denominator

            Lambda[oi] = Lambda[oi].imag
            Lambda[oi] *= - complex(0, 2) / self.Lat.Nsite

        return Lambda

    def SpinConductivity(self, evals, evecs, direction, omegasteps=500, domega=0.002, eta=0.01, qm="SpinHalf"):
        """
        Measure together spin conductivity \sigma(omega)
        Parameter: evals, 1d array of eigen energies
                   evecs, 2d array, with cols being eigen vectors corresponding to evals
                   direction, current direction, x,y or z
        """
        gs = evecs[:, 0]  # ground state
        Eg = evals[0]  # ground state energy

        Lambda = self.Lambda(evals, evecs, direction, omegasteps, domega, eta, qm)

        K = sp.eye(0)  # kinetic Ki
        if direction == "x" or direction == "X":
            K, _ = self.kineticTotal("x", qm)
        elif direction == "y" or direction == "Y":
            K, _ = self.kineticTotal("y", qm)
        elif direction == "z" or direction == "Z":
            K, _ = self.kineticTotal("z", qm)

        # Kinetic part
        Ki = matele(gs, K, gs)

        sigma = Lambda + Ki * np.ones(omegasteps)
        for oi in range(omegasteps):
            sigma[oi] /= complex(0, 1) * (oi + complex(0, 1) * eta)

        return sigma


def main():
    Para = Parameter("../../input.inp")
    Lat = Lattice(Para)

    Hamil = Heisenberg(Lat, Para)
    ham = Hamil.Ham

    evals, evecs = primme.eigsh(ham, Para.Nstates, tol=Para.tolerance, which='SA')
    evals, evecs = sort(evals, evecs)

    ob = HeisenbergObserve(Lat, Para)
    Lambda = ob.Lambda(evals, evecs, "z",  1000, 0.001, 0.01)
    sigma = ob.SpinConductivity(evals, evecs, "z",  1000, 0.001, 0.01)

    Ki, _= ob.kineticTotal("z")

    regular = np.imag(Lambda)
    for oi in range(1000):
        regular[oi] /= oi

    print(matele(evecs[:, 0], Ki, evecs[:, 0]))
    print("\n-----------Conductivity-----------\n", *sigma, sep='\n')
    print("\n-----------Lambda-----------\n", *Lambda, sep='\n')
    print("\n-----------Regular-----------\n", *regular, sep='\n')


if __name__ == '__main__':
    main()
