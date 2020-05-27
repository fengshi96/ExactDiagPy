import sys, re, math, random
import numpy as np
import scipy.sparse as sp
import primme
import h5py
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

    evals, evecs = primme.eigsh(ham, para.Nstates, tol=1e-6, which='SA')
    print("\nEigen Values:-----------\n", *evals, sep='\n')
    print("\nEnd of Eigen Values-----------\n\n")

    # ------------------------ Hdf5 Ostream ------------------------------
    file = h5py.File('dataSpec.hdf5', 'w')
    file.attrs["LLX"] = para.LLX
    file.attrs["LLY"] = para.LLY
    file.attrs["IsPeriodicX"] = para.IsPeriodicX
    file.attrs["IsPeriodicY"] = para.IsPeriodicY
    file.attrs["Kx"] = para.Kxx
    file.attrs["Ky"] = para.Kyy
    file.attrs["Kz"] = para.Kzz
    file.attrs["Hx"] = para.Hz
    file.attrs["Hy"] = para.Hy
    file.attrs["Hz"] = para.Hz
    file.attrs["#States2Keep"] = para.Nstates
    file.attrs["Model"] = para.Model
    file.attrs["Nsites"] = Lat.Nsite


    LatGrp = file.create_group("1.Lattice")
    LatGrp.create_dataset("Mesh", data=Lat.mesh_)
    LatGrp.create_dataset("Nearest Neighbors", data=Lat.nn_)

    ConnGrp = file.create_group("2.Connectors")
    ConnGrp.create_dataset("KxxGraph", data=Hamil.KxxGraph_)
    ConnGrp.create_dataset("KyyGraph", data=Hamil.KyyGraph_)
    ConnGrp.create_dataset("KzzGraph", data=Hamil.KzzGraph_)

    EigGrp = file.create_group("3.Eigen")
    EigGrp.create_dataset("Eigen Values", data=evals)
    EigGrp.create_dataset("Wavefunctions", data=evecs)

    file.close()
    # ------------------------ End: Hdf5 Ostream ------------------------------




if __name__ == '__main__':
    sys.argv  ## get the input argument
    total = len(sys.argv)
    cmdargs = sys.argv
    main(total, cmdargs)
