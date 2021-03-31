import sys
import time
import numpy as np
import h5py
from src.Dofs import Dofs
from src.Parameter import Parameter
from src.Observ import Observ
from src.Lattice import Lattice

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
    print(Lat.Model)
    dof = Dofs("SpinHalf")  # default spin-1/2
    if Lat.Model == "AKLT":
        dof = Dofs("SpinOne")

    # ------- Read dataSpec file -------
    rfile = h5py.File('dataSpec.hdf5', 'r')
    evalset = rfile["3.Eigen"]["Eigen Values"]
    evecset = rfile["3.Eigen"]["Wavefunctions"]

    # extract eigen value and eigen vector from HDF5 rfile
    print(dof.hilbsize ** Lat.Nsite)
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
    elif observname == "energy_condx":
        print("Calculating energy conductivity in x...")
        tic = time.perf_counter()
        Econdx = ob.EcondLocal(evals, evecs)
        toc = time.perf_counter()
        print(f"time = {toc-tic:0.4f} sec")

        file = open("energy_condx.dat", 'w')
        for i in range(0, 400):
            string = str(Econdx[i, 0]) + " " + str(Econdx[i, 1]) + "\n"
            file.write(string)
        file.close()

        # wfile = h5py.File(outputname, 'a')
        # if "energy_condx" in list(wfile.keys()):
        #     wfile.__delitem__("energy_condx")  # over write if existed
        # wfile.create_dataset("energy_condx", data=Econdx)
        # wfile["energy_condx"].attrs["time"] = toc - tic
        #
        # wfile.close()

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

        # ------- Calculate SpSm DOS & write to HDF5---------
    elif observname == "spsm":
        print("Calculating SpSm spectrum...")
        tic = time.perf_counter()
        SPSM = ob.SpSm(evals, evecs, dof.type)
        toc = time.perf_counter()
        print(f"time = {toc - tic:0.4f} sec")

        wfile = h5py.File(outputname, 'a')
        if "SpSm" in list(wfile.keys()):
            wfile.__delitem__("SpSm")  # over write if existed
        wfile.create_dataset("SpSm", data=SPSM)
        wfile["SpSm"].attrs["time"] = toc - tic

        wfile.close()
        # ------- Calculate energy current of TFIM & write to HDF5---------
    elif observname == "ecurrent1":
        site = 2
        if para.Model != "TFIM":
            raise ValueError("ecurrent1 is designed for TFIM exclusively")
        print("Calculating energy current of TFIM...")
        tic = time.perf_counter()
        mEcurr = ob.ecurrent1(evals, evecs, site)
        toc = time.perf_counter()
        print(f"time = {toc - tic:0.4f} sec")
        print("Current at site=", site, "is ", mEcurr)

    else:
        raise ValueError("Observable not supported yet")


if __name__ == '__main__':
    sys.argv  ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    observe(total, cmdargs)
