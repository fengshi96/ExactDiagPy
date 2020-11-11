import sys
import numpy as np
import time
import primme
import h5py
from src.Parameter import Parameter
from src.Hamiltonian import Hamiltonian
from src.Observ import Observ
from src.Lattice import Lattice
from src.Helper import Logger

pi = np.pi


def main(total, cmdargs):
    if total != 2:
        print(" ".join(str(x) for x in cmdargs))
        raise ValueError('Missing arguments')
    inputname = cmdargs[1]
    sys.stdout = Logger()  # for exporting the logfile

    # ----------------------- Build and Diagonalize ---------------------------------
    para = Parameter(inputname)  # import parameters from input.inp
    Lat = Lattice(para)  # Build lattice
    tic = time.perf_counter()
    Hamil = Hamiltonian(Lat)  # Build Hamiltonian
    ham = Hamil.Ham  # Hamiltonian as sparse matrix
    toc = time.perf_counter()
    print(f"Hamiltonian construction time = {toc - tic:0.4f} sec")

    tic = time.perf_counter()
    evals, evecs = primme.eigsh(ham, para.Nstates, tol=1e-6, which='SA')
    toc = time.perf_counter()

    evals = np.round(evals, 10)
    print("\n-----------Beg of Eigen Values-----------\n", *evals, sep='\n')
    print("\n-----------End of Eigen Values-----------")
    print(f"Diagonalization time = {toc - tic:0.4f} sec \n\n")

    # ----------------------- Entanglement properties of GS ---------------------------------
    ob = Observ(Lat)
    ob.EntSpec(evecs[:, 0])








    # ------------------------ Hdf5 Ostream ------------------------------
    # ------------------------ Hdf5 Ostream ------------------------------
    # ------------------------ Hdf5 Ostream ------------------------------
    tic = time.perf_counter()
    file = h5py.File('dataSpec.hdf5', 'w')
    file.attrs["LLX"] = para.LLX; file.attrs["LLY"] = para.LLY
    file.attrs["IsPeriodicX"] = para.IsPeriodicX; file.attrs["IsPeriodicY"] = para.IsPeriodicY
    file.attrs["Kx"] = para.Kxx; file.attrs["Ky"] = para.Kyy; file.attrs["Kz"] = para.Kzz
    file.attrs["Hx"] = para.Hz; file.attrs["Hy"] = para.Hy; file.attrs["Hz"] = para.Hz
    file.attrs["#States2Keep"] = para.Nstates
    file.attrs["Model"] = para.Model
    file.attrs["Nsites"] = Lat.Nsite


    LatGrp = file.create_group("1.Lattice")
    LatGrp.create_dataset("Mesh", data=Lat.mesh_)
    LatGrp.create_dataset("Nearest Neighbors", data=Lat.nn_)

    ConnGrp = file.create_group("2.Connectors")
    # # uncomment for Kitaev, Heisenberg
    # ConnGrp.create_dataset("KxxGraph", data=Hamil.KxxGraph_)
    # ConnGrp.create_dataset("KyyGraph", data=Hamil.KyyGraph_)
    # ConnGrp.create_dataset("KzzGraph", data=Hamil.KzzGraph_)

    # uncomment for Heisenberg/AKLT
    ConnGrp.create_dataset("Kxx1Graph", data=Hamil.Kxx1Graph_)
    ConnGrp.create_dataset("Kyy1Graph", data=Hamil.Kyy1Graph_)
    ConnGrp.create_dataset("Kzz1Graph", data=Hamil.Kzz1Graph_)
    ConnGrp.create_dataset("Kxx2Graph", data=Hamil.Kxx2Graph_)
    ConnGrp.create_dataset("Kyy2Graph", data=Hamil.Kyy2Graph_)
    ConnGrp.create_dataset("Kzz2Graph", data=Hamil.Kzz2Graph_)

    EigGrp = file.create_group("3.Eigen")
    EigGrp.create_dataset("Eigen Values", data=evals)
    EigGrp.create_dataset("Wavefunctions", data=evecs)

    file.close()
    toc = time.perf_counter()
    print(f"\nHDF5 time = {toc - tic:0.4f} sec")
    # ------------------------ End: Hdf5 Ostream ------------------------------
    # ------------------------ End: Hdf5 Ostream ------------------------------
    # ------------------------ End: Hdf5 Ostream ------------------------------




if __name__ == '__main__':
    sys.argv  ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    main(total, cmdargs)
