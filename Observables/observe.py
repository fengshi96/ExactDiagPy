import sys, re, math, random
import numpy as np
import h5py
import scipy.sparse as sp
sys.path.insert(0, '/fs/byo/trivedi/feng.934/2.Projects_2020/3.Kitaev_Transport/2.ED/ExactDiagPy')
from src.Parameter import Parameter
from src.Hamiltonian import Hamiltonian
from src.Observ import Observ, matele
from src.Lattice import Lattice
from src.Helper import matprint, matprintos

pi = np.pi


def observe(total, cmdargs):
    if total < 3:
        print(" ".join(str(x) for x in cmdargs))
        raise ValueError('Missing arguments')
    inputname = cmdargs[1]
    observname = cmdargs[2]
    # outputname = "../dataObserve.hdf5"
    outputname = "dataObserve.hdf5"  ### uncomment for production !!!!!!!!!!!!!!!!!!!!!!!!!
    if total == 4:
        outputname = cmdargs[3]
    # ---------------------------------------------------------------

    para = Parameter(inputname)
    Lat = Lattice(para)
    ob = Observ(Lat)

    # ------- Read dataSpec file -------
    # rfile = h5py.File('../dataSpec.hdf5', 'r')
    rfile = h5py.File('dataSpec.hdf5', 'r')  ### uncomment for production !!!!!!!!!!!!!!!!!!!!!!!!!
    group = rfile["3.Eigen"]
    evalset = group["Eigen Values"]
    evecset = group["Wavefunctions"]

    # extract eigen value and eigen vector from HDF5 rfile
    evals = np.zeros(para.Nstates, dtype=float)
    evecs = np.zeros((2 ** Lat.Nsite, para.Nstates), dtype=complex)
    evalset.read_direct(evals)
    evecset.read_direct(evecs)

    rfile.close()

    # ------- Calculate spin response S(\omega) & write to HDF5---------
    if observname == "spin_response":
        SpinRes = ob.SpRe(evals, evecs)
        wfile = h5py.File(outputname, 'w')
        wfile.create_dataset("Spin Response", data=SpinRes)
        wfile.close()
    # ------- Calculate spin conductivity & write to HDF5---------
    elif observname == "spin_cond":
        pass
    # ------- Calculate energy conductivity & write to HDF5---------
    elif observname == "energy_cond":
        pass
    else:
        raise ValueError("Observable not supported yet")


if __name__ == '__main__':
    sys.argv  ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    observe(total, cmdargs)
