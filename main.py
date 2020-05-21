import sys, re, math, random
import numpy as np
import scipy.sparse as sp
import primme
from src.Parameter import Parameter
from src.Hamiltonian import Hamiltonian
from src.Observ import Observ
from src.Lattice import Lattice


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
    print("\nEigen Values:\n", *evals, sep='\n')

    ob = Observ(Lat)  # creat Observable object
    Tscurrx, Tscurry, Tscurrz = ob.TscurrBuild()  # Build total spin current operators in 3 directions
    print("\n\nTotal spin current in x,y,z:", *ob.Tscurr_str, sep="\n")  # print string


if __name__ == '__main__':
    sys.argv  ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    main(total, cmdargs)
