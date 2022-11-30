import sys
import time
import numpy as np
import h5py
from src.Dofs import Dofs
from src.Parameter import Parameter
from src.Observ import Observ
from src.Lattice import Lattice
from src.Helper import printfArray

pi = np.pi


def observe(total, cmdargs):
    localSite = None
    if total < 3:
        print(" ".join(str(x) for x in cmdargs))
        raise ValueError('Missing arguments')
    inputname = cmdargs[1]
    observname = cmdargs[2]
    # Needed for loacl information input
    if total > 3:
        localSite = int(cmdargs[3])
    outputname = "dataObserve.hdf5"
    #if total == 4:
    #   outputname = cmdargs[3]
    # ---------------------------------------------------------------

    para = Parameter(inputname)
    Lat = Lattice(para)
    ob = Observ(Lat, para)
    print(para.parameters["Model"])
    dof = Dofs("SpinHalf")  # default spin-1/2
    if para.parameters["Model"] == "AKLT":
        dof = Dofs("SpinOne")

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

    # ------- Calculate local Sx Sy Sz (for gs) & write to ASCII---------
    if observname == "local_spins":
        localSpins = np.zeros((Lat.Nsite, 4))
        print("Calculating local_Spins...")
        tic = time.perf_counter()
        for localSite in range(Lat.Nsite):
            local_Sx = round(ob.mLocalSx(evecs[:, 0], localSite), 9)
            local_Sy = round(ob.mLocalSy(evecs[:, 0], localSite), 9)
            local_Sz = round(ob.mLocalSz(evecs[:, 0], localSite), 9)
            localSpins[localSite, 0] = int(localSite)
            localSpins[localSite, 1] = local_Sx
            localSpins[localSite, 2] = local_Sy
            localSpins[localSite, 3] = local_Sz
        toc = time.perf_counter()
        printfArray(localSpins, "localSpins.dat")
        print(f"time = {toc-tic:0.4f} sec")

    # ------- Calculate static correlations (for gs) & write to ASCII---------
    elif observname == "Cz":
        """
        3 input in termial: input name, observe name, and site of reference 
        """
        print("Calculating Static Sz-Sz...")
        tic = time.perf_counter()
        Czc = ob.Czc(localSite, evecs[:, 0])
        toc = time.perf_counter()
        print(f"time = {toc-tic:0.4f} sec")
        file = open("Czc.dat", 'w')
        for c in Czc:
            file.write(str(c) + "\n")
        file.close()
    elif observname == "Cx":
        print("Calculating Static Sx-Sx...")
        tic = time.perf_counter()
        Cxc = ob.Cxc(localSite, evecs[:, 0])
        toc = time.perf_counter()
        print(f"time = {toc-tic:0.4f} sec")
        file = open("Cxc.dat", 'w')
        for c in Cxc:
            file.write(str(c) + "\n")
        file.close()
    elif observname == "Cy":
        print("Calculating Static Sy-Sy...")
        tic = time.perf_counter()
        Cyc = ob.Cyc(localSite, evecs[:, 0])
        toc = time.perf_counter()
        print(f"time = {toc-tic:0.4f} sec")
        file = open("Cyc.dat", 'w')
        for c in Cyc:
            file.write(str(c) + "\n")
        file.close()

    elif observname == "CC_all":
        print("Calculating Static correlation for all pairs...")
        tic = time.perf_counter()
        Ccxx, Ccxy, Ccxz, Ccyx, Ccyy, Ccyz, Cczx, Cczy, Cczz = ob.CC_all(localSite, evecs[:, 0])
        toc = time.perf_counter()
        print(f"time = {toc-tic:0.4f} sec")
        Ccs = np.vstack([Ccxx, Ccxy, Ccxz, Ccyx, Ccyy, Ccyz, Cczx, Cczy, Cczz])
        printfArray(Ccs, "CCs.dat")

    # ------- Calculate 4-spin correlations (for gs) & write to ASCII---------
    elif observname == "fourSpin":
        """
        3 input in termial: input name, observe name, and site of reference 
        """
        print("Calculating test 4-spin correlator...")
        tic = time.perf_counter()
        C4 = ob.fourSpin(localSite, evecs[:, 0])
        toc = time.perf_counter()
        print(f"time = {toc-tic:0.4f} sec")
        file = open("C4.dat", 'w')
        file.write(str(np.real(C4)))
        file.close()
        print("C4 =", np.real(C4))

    # ------- Calculate spin response S(\omega) & write to HDF5---------
    elif observname == "spin_response":
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

    # ------- Calculate TotalS & write to ascii---------
    elif observname == "totalS":
        print("Calculating totalS...")
        tic = time.perf_counter()
        St = ob.TotalS(evecs[:, 0])
        toc = time.perf_counter()
        print(f"time = {toc-tic:0.4f} sec\n")
        print("Magnitization=", St)

    # ------- Calculate magnetization (Total Sz) & write to ascii---------
    elif observname == "magnetization":
        print("Calculating magnetization...")
        tic = time.perf_counter()
        Magx = ob.TotalSx(evecs[:, 0])
        Magy = ob.TotalSy(evecs[:, 0])
        Magz = ob.TotalSz(evecs[:, 0])
        toc = time.perf_counter()
        print(f"time = {toc-tic:0.4f} sec\n")
        print(f"totalSx = {Magx.real:.6f}")
        print(f"totalSy = {Magy.real:.6f}")
        print(f"totalSz = {Magz.real:.6f}")
        print(f"totalSe3 = {Magx.real + Magy.real + Magz.real:.6f}")

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
        if para.parameters["Model"] != "TFIM":
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
