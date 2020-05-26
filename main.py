import sys, re, math, random
import numpy as np
import scipy.sparse as sp
import primme
from src.Parameter import Parameter
from src.Hamiltonian import Hamiltonian
from src.Observ import Observ, matele
from src.Lattice import Lattice
from src.Helper import matprint, matprintos

pi = np.pi


def main(total, cmdargs):
    if total != 2:
        print(" ".join(str(x) for x in cmdargs))
        raise ValueError('Missing arguments')
    inputname = cmdargs[1]
    # ---------------------------------------------------------------

    para = Parameter(inputname)  # import parameters from input.inp
    Lat = Lattice(para)  # Build lattice
    Hamil = Hamiltonian(Lat)  # Build Hamiltonian
    ham = Hamil.Ham  # Hamiltonian as sparse matrix

    Nstates = para.Nstates  # Number of eigenstates to keep
    evals, evecs = primme.eigsh(ham, Nstates, tol=1e-6, which='SA')
    print("\nEigen Values:-----------\n", *evals, sep='\n')
    print("\nEnd of Eigen Values-----------\n\n")

    ob = Observ(Lat)  # creat Observable object
    # Tscurrx, Tscurry, Tscurrz = ob.TscurrBuild()  # Build total spin current operators in 3 directions
    # print("\n\nTotal spin current in x,y,z:", *ob.Tscurr_str, sep="\n")  # print string

    tmpsr = ob.SpRe(evals, evecs)
    matprintos(tmpsr, "SpinRes.dat")


if __name__ == '__main__':
    sys.argv  ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    main(total, cmdargs)
