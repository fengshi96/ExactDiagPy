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


def readfArray(str, Complex = False):
    file = open(str, 'r')
    lines = file.readlines()
    file.close()

    # Determine shape:
    row = len(lines)
    testcol = lines[0].strip("\n").rstrip().split()
    col = len(testcol)  # rstip to rm whitespace at the end

    m = np.zeros((row, col))
    for i in range(row):
        line = lines[i].strip("\n").rstrip().split()
        # print(line)
        for j in range(col):
            val = float(line[j])
            m[i, j] = val
    return m


def printfArray(A, filename, transpose=False):
    file = open(filename, "w")
    try:
        col = A.shape[1]
    except IndexError:
        A = A.reshape(-1, 1)

    row = A.shape[0]
    col = A.shape[1]

    if transpose == False:
        for i in range(row):
            for j in range(col - 1):
                file.write(str(A[i, j]) + " ")
            file.write(str(A[i, col - 1]))  # to avoid whitespace at the end of line
            file.write("\n")
    elif transpose == True:
        for i in range(col):
            for j in range(row - 1):
                file.write(str(A[j, i]) + " ")
            file.write(str(A[row - 1, i]))
            file.write("\n")
    else:
        raise ValueError("3rd input must be Bool")
    file.close()

def matprint(A, delimiter='\t', nospacing=False):
    row = A.shape[0]
    col = A.shape[1]
    for i in range(row):
        if not nospacing:
            print("\n")
        for j in range(col):
            print(A[i, j], end=delimiter)
    print("\n")


def vecprint(A):
    row = len(A)
    for i in range(row):
        print(A[i], end='\t')
    print("\n")


def matprintos(A, filename, separation=0):
    if separation == 0:
        file = open(filename, "w")
        row = A.shape[0]
        col = A.shape[1]
        for i in range(row):
            for j in range(col):
                file.write(str(A[i, j]) + "\t")
            file.write("\n")
        file.close()
    else:
        file = open(filename, "w")
        row = A.shape[0]
        col = A.shape[1]

        segments = int(row / separation)
        if segments > 0:
            for c in range(segments):
                for i in range(separation):
                    for j in range(col):
                        file.write(str(A[c*separation+i, j]) + "\t")
                    file.write("\n")
                file.write("\n")
            file.close()
        else:
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
