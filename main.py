import sys, math
import numpy as np
import time
import primme
from src.Parameter import Parameter
from src.models.Kitaev import Kitaev
from src.models.Kitaev_Ladder import Kitaev_Ladder
from src.models.Heisenberg import Heisenberg
from src.Observ import Observ
from src.Lattice import Lattice
from src.Helper import Logger, sort, hd5Storage
import scipy.sparse.linalg

from src.models.ToricCode import ToricCode

pi = np.pi


def main(total, cmdargs):
    if total != 2:
        print(" ".join(str(x) for x in cmdargs))
        raise ValueError('Missing arguments')
    inputname = cmdargs[1]
    sys.stdout = Logger()  # for exporting the logfile

    # ----------------------- Build and Diagonalize ---------------------------------
    Para = Parameter(inputname)  # import parameters from input.inp
    Lat = Lattice(Para)  # Build lattice

    tic = time.perf_counter()
    #######################################
    if Para.parameters["Model"] == "Heisenberg":
        Hamil = Heisenberg(Lat, Para)    # Build Hamiltonian object
    elif Para.parameters["Model"] == "Kitaev":
        Hamil = Kitaev(Lat, Para)
    elif Para.parameters["Model"] == "ToricCode":
        Hamil = ToricCode(Lat, Para)
    elif Para.parameters["Model"] == "Kitaev_Ladder":
        Hamil = Kitaev_Ladder(Lat, Para)
    #######################################
    ham = Hamil.Ham  # mount in Hamiltonian as sparse matrix
    toc = time.perf_counter()
    print(f"Hamiltonian construction time = {toc - tic:0.4f} sec")

    tic = time.perf_counter()
    evals, evecs = primme.eigsh(ham, Para.parameters["Nstates"], tol=Para.parameters["tolerance"], which='SA')
    # evals, evecs = scipy.sparse.linalg.eigsh(ham, Para.parameters["Nstates"], which='SA', tol=Para.parameters["tolerance"], ncv=2*Para.parameters["Nstates"])
    evals, evecs = sort(evals, evecs)
    toc = time.perf_counter()

    evals = np.round(evals, 10)
    print("\n-----------Beg of Eigen Values-----------\n", *evals, sep='\n')
    print("\n-----------End of Eigen Values-----------")
    print(f"Diagonalization time = {toc - tic:0.4f} sec \n\n")
    hd5Storage(Para, Lat, Hamil, [evals, evecs])

    ob = Observ(Lat, Para)  # initialize observables

    # -------------- Entanglement properties of GS -----------
    if "Option" in Para.parameters.keys():
        if "EE" in Para.parameters["Option"]:
            EntS, Entvec = ob.EntSpec(evecs[:, 0])  # Entanglement spectrum and vector
            EntS = np.round(EntS, decimals=10)
            EntS_log = np.zeros(len(EntS))
            for i in range(len(EntS)):
                if EntS[i] <= 0:
                    EntS_log[i] = 0
                else:
                    EntS_log[i] = math.log(EntS[i])
            EE = - np.around(np.dot(EntS, EntS_log), decimals=8)
            print("Entanglement Spectrum=\n", EntS)
            print("Entanglement Entropy=", EE)

        # ------------------------ Ascii Ostream for EE--------
        # # For a U-scan of Bose-Hubbard model
        # file = open("entspec.dat", "w")
        # if Para.Model == "Bose_Hubbard":
        #     for i in range(EntS.size):
        #         file.write(str(abs(Para.U)) + " " + str(EntS[i]))
        #         file.write("\n")
        # else:
        #     for i in range(EntS.size):
        #         file.write(str(abs(Para.Hx)) + " " + str(EntS[i]))
        #         file.write("\n")






if __name__ == '__main__':
    sys.argv  ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    main(total, cmdargs)
