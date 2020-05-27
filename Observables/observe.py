import sys, re, math, random
import numpy as np
import h5py
import scipy.sparse as sp
import primme
sys.path.insert(0, '../')
from src.Parameter import Parameter
from src.Hamiltonian import Hamiltonian
from src.Observ import Observ, matele
from src.Lattice import Lattice
from src.Helper import matprint, matprintos

pi = np.pi


def observe(total, cmdargs):
    if total < 2:
        print(" ".join(str(x) for x in cmdargs))
        raise ValueError('Missing arguments')
    inputname = cmdargs[1]
    outputname = "../observe.dat"
    # outputname = "observe.dat"  ### uncomment for production !!!!!!!!!!!!!!!!!!!!!!!!!
    if total == 3:
        outputname = cmdargs[2]
    # ---------------------------------------------------------------

    para = Parameter(inputname)
    Lat = Lattice(para)
    ob = Observ(Lat)

    # Read HDF5 file
    file = h5py.File('../dataSpec.hdf5', 'r')
    # file = h5py.File('dataSpec.hdf5', 'r')  ### uncomment for production !!!!!!!!!!!!!!!!!!!!!!!!!
    group = file["3.Eigen"]
    evalset = group["Eigen Values"]
    evecset = group["Wavefunctions"]

    # extract eigen value and eigen vector from HDF5 file
    evals = np.zeros(para.Nstates, dtype=float)
    evecs = np.zeros((2 ** Lat.Nsite, para.Nstates), dtype=complex)
    evalset.read_direct(evals)
    evecset.read_direct(evecs)

    #  Calculate spin response S(\omega)
    SpinRes = ob.SpRe(evals, evecs)
    matprintos(SpinRes, outputname)


if __name__ == '__main__':
    sys.argv  ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    observe(total, cmdargs)
