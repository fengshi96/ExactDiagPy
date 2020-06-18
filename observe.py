import sys, re, math, random
import time
import numpy as np
import h5py
import scipy.sparse as sp

from src.Dofs import Dofs
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
    outputname = "dataObserve.hdf5"
    if total == 4:
        outputname = cmdargs[3]
    # ---------------------------------------------------------------

    para = Parameter(inputname)
    Lat = Lattice(para)
    ob = Observ(Lat)

    dof = Dofs("SpinHalf")  # default spin-1/2
    if Lat.Model == "AKLT":
        dof = Dofs("SpinOne")

    # ------- Read dataSpec file -------
    rfile = h5py.File('dataSpec.hdf5', 'r')
    evalset = rfile["3.Eigen"]["Eigen Values"]
    evecset = rfile["3.Eigen"]["Wavefunctions"]

    # extract eigen value and eigen vector from HDF5 rfile
    evals = np.zeros(para.Nstates, dtype=float)
    evecs = np.zeros((dof.hilbsize ** Lat.Nsite, para.Nstates), dtype=complex)
    evalset.read_direct(evals)
    evecset.read_direct(evecs)

    rfile.close()

    # ------- Calculate spin response S(\omega) & write to HDF5---------
    if observname == "spin_response":
        print("Calculating S(omega)...")
        tic = time.perf_counter()
        SpinRes = ob.SpRe(evals, evecs, dof.type)
        toc = time.perf_counter()
        print(f"time = {toc-tic:0.4f} sec")

        wfile = h5py.File(outputname, 'a')
        if "Spin Response" in list(wfile.keys()):
            wfile.__delitem__("Spin Response")  # over write if existed
        wfile.create_dataset("Spin Response", data=SpinRes)
        wfile["Spin Response"].attrs["time"] = toc - tic

        wfile.close()
    # ------- Calculate spin conductivity & write to HDF5---------
    elif observname == "spin_cond":
        pass
    # ------- Calculate energy conductivity & write to HDF5---------
    elif observname == "energy_cond":
        pass
    # ------- Calculate Single-Magnon DOS & write to HDF5---------
    elif observname == "singlemagnon":
        print("Calculating Single-Magnon DOS...")
        tic = time.perf_counter()
        SMag = ob.SingleMagnon(evals, evecs, dof.type)
        toc = time.perf_counter()
        print(f"time = {toc-tic:0.4f} sec")

        wfile = h5py.File(outputname, 'a')
        if "Single-Magnon DOS" in list(wfile.keys()):
            wfile.__delitem__("Single-Magnon DOS")  # over write if existed
        wfile.create_dataset("Single-Magnon DOS", data=SMag)
        wfile["Single-Magnon DOS"].attrs["time"] = toc - tic

        wfile.close()

    # ------- Calculate SzSz DOS & write to HDF5---------
    elif observname == "szsz":
        print("Calculating SzSz spectrum...")
        tic = time.perf_counter()
        SZSZ = ob.SzSz(evals, evecs, dof.type)
        toc = time.perf_counter()
        print(f"time = {toc-tic:0.4f} sec")

        wfile = h5py.File(outputname, 'a')
        if "SzSz" in list(wfile.keys()):
            wfile.__delitem__("SzSz")  # over write if existed
        wfile.create_dataset("SzSz", data=SZSZ)
        wfile["SzSz"].attrs["time"] = toc - tic

        wfile.close()
    else:
        raise ValueError("Observable not supported yet")


if __name__ == '__main__':
    sys.argv  ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    observe(total, cmdargs)
