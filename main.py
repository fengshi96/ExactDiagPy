import sys, re, math, random
import numpy as np
import scipy.sparse as sp
import primme
from src.Parameter import Parameter
from src.Hamiltonian import Hamiltonian
from src.Observ import Observ, matele
from src.Lattice import Lattice
# from src.Observ import mel
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
    print("\nEigen Values:\n", *evals, sep='\n')

    ob = Observ(Lat)  # creat Observable object
    Tscurrx, Tscurry, Tscurrz = ob.TscurrBuild()  # Build total spin current operators in 3 directions
    print("\n\nTotal spin current in x,y,z:", *ob.Tscurr_str, sep="\n")  # print string

    gs = evecs[:, 0]  # ground state
    Eg = evals[0]  # ground state energy

    print(matele(gs, Tscurrx, gs))

    omegasteps = 200
    domega = 0.005
    eta = 0.009

    # Sr = np.zeros(omegasteps)  # spin response
    # for oi in range(0, omegasteps):
    #     omega = domega * oi
    #     Itensity = np.zeros((3, 3))
    #
    #     # for each omega, define a matrix Mel_{si,mi}
    #     Melx = np.zeros((Lat.Nsite, para.Nstates))  # <m|S_i^x|gs>
    #     Mely = np.zeros((Lat.Nsite, para.Nstates))  # <m|S_i^y|gs>
    #     Melz = np.zeros((Lat.Nsite, para.Nstates))  # <m|S_i^z|gs>
    #     for mi in range(0, Nstates):
    #         Em = evals[mi]
    #         for si in range(0, Lat.Nsite):
    #             denom = complex(omega - (Em - Eg), -eta)
    #
    #             # <m|S_i^a|gs>
    #             Melx[si, mi] = mel(evecs[:, mi], ob.Sx(si), gs)
    #             Melx[si, mi] = mel(evecs[:, mi], ob.Sy(si), gs)
    #             Melx[si, mi] = mel(evecs[:, mi], ob.Sz(si), gs)
    #
    #             # <gs|S_i^a|m><m|S_i^b|gs>
    #             Melxx = Melx[si, mi].conjugate() * Melx[si, mi]
    #             Melxy = Melx[si, mi].conjugate() * Mely[si, mi]
    #             Melxz = Melx[si, mi].conjugate() * Melz[si, mi]
    #
    #             Melyx = Mely[si, mi].conjugate() * Melx[si, mi]
    #             Melyy = Mely[si, mi].conjugate() * Mely[si, mi]
    #             Melyz = Mely[si, mi].conjugate() * Melz[si, mi]
    #
    #             Melzx = Melz[si, mi].conjugate() * Melx[si, mi]
    #             Melzy = Melz[si, mi].conjugate() * Mely[si, mi]
    #             Melzz = Melz[si, mi].conjugate() * Melz[si, mi]









if __name__ == '__main__':
    sys.argv  ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    main(total, cmdargs)
