import numpy as np
import sys
import h5py
import time


def hd5Storage(Para, Lat, Hamil, Eigen):
    tic = time.perf_counter()
    file = h5py.File('dataSpec.hdf5', 'w')
    file.attrs["LLX"] = Para.parameters["LLX"]
    file.attrs["LLY"] = Para.parameters["LLY"]

    file.attrs["IsPeriodicX"] = Para.parameters["IsPeriodicX"]
    file.attrs["IsPeriodicY"] = Para.parameters["IsPeriodicY"]

    file.attrs["#States2Keep"] = Para.parameters["Nstates"]
    file.attrs["Dof"] = Para.parameters["Dof"]
    file.attrs["Geometry"] = Para.parameters["Geometry"]
    file.attrs["Model"] = Para.parameters["Model"]
    file.attrs["Nsites"] = Lat.Nsite

    ConnGrp = file.create_group("2.Connectors")

    if "Spin" in Para.parameters["Dof"]:
        file.attrs["Kx"] = Para.parameters["Kxx"]
        file.attrs["Ky"] = Para.parameters["Kyy"]
        file.attrs["Kz"] = Para.parameters["Kzz"]

        file.attrs["Hx"] = Para.parameters["Bxx"]
        file.attrs["Hy"] = Para.parameters["Byy"]
        file.attrs["Hz"] = Para.parameters["Bzz"]

        if Para.parameters["Model"] == "Kitaev":
            ConnGrp.create_dataset("KxxGraph", data=Hamil.KxxGraph_)
            ConnGrp.create_dataset("KyyGraph", data=Hamil.KyyGraph_)
            ConnGrp.create_dataset("KzzGraph", data=Hamil.KzzGraph_)

        elif Para.parameters["Model"] == "AKLT":
            ConnGrp.create_dataset("Kxx1Graph", data=Hamil.Kxx1Graph_)
            ConnGrp.create_dataset("Kyy1Graph", data=Hamil.Kyy1Graph_)
            ConnGrp.create_dataset("Kzz1Graph", data=Hamil.Kzz1Graph_)
            ConnGrp.create_dataset("Kxx2Graph", data=Hamil.Kxx2Graph_)
            ConnGrp.create_dataset("Kyy2Graph", data=Hamil.Kyy2Graph_)
            ConnGrp.create_dataset("Kzz2Graph", data=Hamil.Kzz2Graph_)

    elif "Fermion" in Para.parameters["Dof"]:
        ConnGrp.create_dataset("tGraph", data=Hamil.tGraph_)
    else:
        raise ValueError("Para.Dof not valid")

    EigGrp = file.create_group("3.Eigen")
    EigGrp.create_dataset("Eigen Values", data=Eigen[0])
    EigGrp.create_dataset("Wavefunctions", data=Eigen[1])

    # if Para.Option is not None:
    #     if "EE" in Para.Option:
    #         EigGrp = file.create_group("4.Entanglment")
    #         EigGrp.create_dataset("ES", data=EntS)
    #         EigGrp.create_dataset("ESlog", data=EntS_log)
    #         EigGrp.create_dataset("EE", data=EE)

    file.close()
    toc = time.perf_counter()
    print(f"\nHDF5 time = {toc - tic:0.4f} sec")





def matprint(A):
    row = A.shape[0]
    col = A.shape[1]
    for i in range(row):
        print("\n")
        for j in range(col):
            print(A[i, j], end='\t')
    print("\n")


def vecprint(A):
    row = len(A)
    for i in range(row):
        print(A[i], end='\t')
    print("\n")


def matprintos(A, filename):
    file = open(filename, "w")
    row = A.shape[0]
    col = A.shape[1]
    for i in range(row):
        for j in range(col):
            file.write(str(A[i, j]) + "\t")
        file.write("\n")
    file.close()


def sort(evals, evecs):
    index_ascend = np.argsort(evals)
    evals_sorted = evals[index_ascend]
    evecs_sorted = evecs[:, index_ascend]
    return evals_sorted, evecs_sorted


class Logger:
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("logfile.log", "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        pass
